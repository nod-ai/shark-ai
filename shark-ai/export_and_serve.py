import logging
from pathlib import Path
import subprocess
import sys
import os
import shutil
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter

fmt = logging.Formatter("[%(levelname)s] %(message)s")

stdout = logging.StreamHandler(stream=sys.stdout)
stdout.setFormatter(fmt)

logger = logging.getLogger("export_and_serve")
logger.addHandler(stdout)
logger.propagate = False


class CliParser(ArgumentParser):
    def print_help(self, file=None) -> None:
        if file is None:
            file = sys.stdout

        help_text = """usage: export_and_serve.py [-h] [-v] [-c] -w WEIGHT_FILE [-e] [-a ARTIFACT_DIR] [-s] [-b BATCH_SIZES] [-i IR] [-p {1,8}] export_dir

Utility script to combine shark-ai tools

positional arguments:
  export_dir                        the directory where the exported artifacts will be saved.

options:
  -h, --help                        show this help message and exit
  -v, --verbose                     set logging level to INFO. The default logging level is WARNING.
  -c, --compile                     compile the exported model as part of the pipeline. Default is FALSE.
  -w, --weight-file WEIGHT_FILE     the location of the GGUF/IRPA file(s) that contain the parameters.
  -e, --export                      export the model in tp1 mode.
  -a, --artifact-dir ARTIFACT_DIR   the location where the artifacts (sharded weights) should be saved. Defaults to EXPORT_DIR/artifacts/
  -s, --shard                       shard the weight file in tp8 mode and export to MLIR.
  -b, --batch-sizes BATCH_SIZES     batch sizes for export. Multiple batch sizes should be separated by a ','.
  -i, --ir IR                       location for the MLIR to be compiled, if compilation is done independently.
  -p, --tensor-parallel {1,8}       tensor parallel size. Used for independent compilation. Defaults to 1.
        """
        _ = file.write(help_text + "\n")


class BasePipeline:
    def __init__(
        self,
        compile: bool,
        shard: bool,
        export: bool,
        tp: int,
        batch_sizes: list[int],
        ir_path: Path | None,
        artifacts_dir: Path,
        weight_loc: Path,
        export_dir: Path,
    ):
        self.compile: bool = compile
        self.shard: bool = shard
        self.export: bool = export
        self.artifacts_dir: Path = artifacts_dir
        self.weight_loc: Path = weight_loc
        self.export_dir: Path = export_dir
        self.mlir_path: Path | None = None
        self.model_config_path: Path | None = None
        self.export_batch_sizes: list[int] = batch_sizes
        self.hip_target: str = "gfx942"
        self.hal_target_backend: str = "rocm"
        self.ir_path: Path | None = ir_path
        self.tp: int = tp

    def exec_pipeline(self):
        self.validate()
        if self.export:
            self.export_tp1()

        if self.shard:
            self.export_tp8()

        if self.compile:
            self.compile_model()

    def validate(self):
        if not self.shard and not self.export and not self.ir_path and self.compile:
            logger.error(
                "To run compilation, either TP1 export (-e) or TP8 export (-s) must be specified, or path to IR must be passed in"
            )
            exit(1)

        if not os.path.isfile(self.weight_loc):
            logger.error(f"File {str(self.weight_loc)} does not exist")
            raise ValueError(
                f"Invalid path to weight file {str(self.weight_loc)}. File does not exist."
            )

        if not os.path.isdir(self.export_dir):
            logger.warning(
                f"Export directory {str(self.export_dir)} does not exist, creating..."
            )
            try:
                Path.mkdir(self.export_dir, parents=True)
            except:
                logger.error("Failed to create export directory")
                raise

        if self.shard and not os.path.isdir(self.artifacts_dir):
            logger.warning(
                f"Artifacts directory {str(self.artifacts_dir)} does not exist, creating..."
            )
            try:
                Path.mkdir(self.artifacts_dir, parents=True)
            except:
                logger.critical("Failed to create artifacts directory")
                raise

        if self.export:
            self.mlir_path = Path(os.path.join(self.export_dir, "model.mlir"))
            self.model_config_path = Path(os.path.join(self.export_dir, "config.json"))

            logger.info("TP1 export is enabled")
            logger.info(f"TP1 MLIR will be saved to {self.mlir_path}")
            logger.info(f"TP1 model config will be saved to {self.model_config_path}")

        if self.shard:
            self.mlir_path = Path(os.path.join(self.export_dir, "model.mlir"))
            self.model_config_path = Path(
                os.path.join(self.export_dir, "config_tp8.json")
            )

            logger.info("TP8 sharding and export is enabled")
            logger.info(f"Artifacts directory set to {self.artifacts_dir}")
            logger.info(f"TP8 MLIR will be saved to {self.mlir_path}")
            logger.info(f"TP8 model config will be saved to {self.model_config_path}")

        if self.compile:
            logger.info("Compilation is enabled")
            if self.ir_path is not None:
                logger.info(f"IR for compilation located at {self.ir_path}")
                self.mlir_path = self.ir_path

        logger.info(f"Exported model will be saved to {self.export_dir}")

    def export_tp1(self) -> None:
        batch_size_string = ",".join([str(b) for b in self.export_batch_sizes])
        logger.info(f"Exporting with batch sizes {batch_size_string}")

        # TODO(vinayakdsci): Refactor this into a separate function.
        export_subp = subprocess.run(
            [
                "python3",
                "-m",
                "sharktank.examples.export_paged_llm_v1",
                f"--gguf-file={self.weight_loc}",
                f"--output-mlir={str(self.mlir_path)}",
                f"--output-config={self.model_config_path}",
                f"--bs={batch_size_string}",
            ],
            capture_output=True,
        )

        # Exit if the command fails, no point in throwing an exception.
        if export_subp.returncode != 0:
            logger.error("Failed to export model in TP1 mode")
            print(export_subp.stderr.decode(), file=sys.stderr)
            exit(export_subp.returncode)

        logger.info(
            f"Successfully exported and saved model MLIR and config to {self.export_dir}"
        )

    def export_tp8(self) -> None:
        shard_file = os.path.join(
            str(self.artifacts_dir),
            str(self.weight_loc).split("/")[-1][:-5] + "_tp8.irpa",
        )
        shard_subp = subprocess.run(
            [
                "python3",
                "-m",
                "sharktank.examples.sharding.shard_llm_dataset",
                f"--irpa-file={self.weight_loc}",
                f"--output-irpa={shard_file}",
                "--tensor-parallelism-size=8",
            ],
            capture_output=True,
        )

        if shard_subp.returncode != 0:
            logger.error("Failed to shard model to TP8")
            print(shard_subp.stderr.decode(), file=sys.stderr)
            exit(shard_subp.returncode)

        logger.info(
            f"Successfully sharded weight file; Unranked weight file located at {shard_file}"
        )

        # TODO(vinayakdsci): Refactor this into a separate function.
        logger.info("Starting TP8 model export")
        export_subp = subprocess.run(
            [
                "python3",
                "-m",
                "sharktank.examples.export_paged_llm_v1",
                f"--irpa-file={shard_file}",
                f"--output-mlir={str(self.mlir_path)}",
                f"--output-config={self.model_config_path}",
                "--bs=4",
            ],
            capture_output=True,
        )

        if export_subp.returncode != 0:
            logger.error("Failed to export model in TP8 mode")
            print(export_subp.stderr.decode(), file=sys.stderr)
            exit(export_subp.returncode)

        logger.info(
            f"Successfully exported and saved model MLIR and config to {self.export_dir}"
        )

    def compile_model(self) -> None:
        if not shutil.which("iree-compile"):
            logger.error("iree-compile not found in PATH")
            raise FileNotFoundError(
                "iree-compile binary is required to be in PATH for compilation."
            )

        if self.shard or self.tp == 8:
            self._compile_tp8()
        else:
            self._compile_tp1()

    def _compile_tp1(self):
        logger.info("Compiling unsharded IR")
        tp1_vmfb_path = os.path.join(self.export_dir, "model.vmfb")
        cmd = [
            "iree-compile",
            f"{self.mlir_path}",
            f"-o={tp1_vmfb_path}",
        ]

        # TODO(vinayakdsci): Add a flag to support backends other than rocm.
        cmd += [f"--iree-hal-target-backends={self.hal_target_backend}"]

        # TODO(vinayakdsci): Add a flag to support targets other than gfx942.
        cmd += [f"--iree-hip-target={self.hip_target}"]

        compile_subp = subprocess.run(
            cmd,
            capture_output=True,
        )

        if compile_subp.returncode != 0:
            logger.error("Failed to compile unsharded IR")
            print(compile_subp.stderr.decode(), file=sys.stderr)
            exit(compile_subp.returncode)

        logger.info(
            f"Successfully compiled and saved TP1 VMFB module to {tp1_vmfb_path}"
        )

    def _compile_tp8(self):
        logger.info("Compiling sharded IR")
        tp8_vmfb_path = os.path.join(self.export_dir, "model_tp8.vmfb")
        cmd = [
            "iree-compile",
            f"{self.mlir_path}",
            f"-o={tp8_vmfb_path}",
        ]

        cmd += [f"--iree-hal-target-device=hip[{idx}]" for idx in range(0, 8)]

        # TODO(vinayakdsci): Add a flag to support targets other than gfx942.
        cmd += [f"--iree-hip-target={self.hip_target}"]

        compile_subp = subprocess.run(
            cmd,
            capture_output=True,
        )

        if compile_subp.returncode != 0:
            logger.error("Failed to compile sharded IR")
            print(compile_subp.stderr.decode(), file=sys.stderr)
            exit(compile_subp.returncode)

        logger.info(
            f"Successfully compiled and saved TP8 VMFB module to {tp8_vmfb_path}"
        )


def main(args: Namespace) -> None:
    compile = args.compile
    artifacts_dir = (
        args.artifact_dir
        if args.artifact_dir is not None
        else Path(os.path.join(args.export_dir, "artifacts/"))
    )
    shard = args.shard
    export = args.export
    weight_loc = args.weight_file
    export_dir = args.export_dir
    ir_path = args.ir
    tp = args.tensor_parallel

    if ir_path is not None and not compile:
        logger.warning(
            "Path to IR was provided through -i, but compilation was not enabled."
        )

    batch_sizes = [1, 4]
    if args.batch_sizes:
        batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]

    if args.verbose:
        logger.setLevel(logging.INFO)

    base_pipeline = BasePipeline(
        compile,
        shard,
        export,
        tp,
        batch_sizes,
        ir_path,
        artifacts_dir,
        weight_loc,
        export_dir,
    )

    base_pipeline.exec_pipeline()


def _get_argparser() -> CliParser:
    msg = "Utility script to combine shark-ai tools"
    # The argparser should not print the usage when an error occurs.
    # We handle that ourselves.
    parser = CliParser(
        prog="export_and_serve.py",
        description=msg,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="set logging level to INFO. The default logging level is WARNING.",
    )
    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        default=False,
        help="compile the exported model as part of the pipeline. Default is FALSE.",
    )
    parser.add_argument(
        "export_dir",
        type=Path,
        help="the directory where the exported artifacts will be saved.",
    )
    parser.add_argument(
        "-w",
        "--weight-file",
        type=Path,
        help="the location of the GGUF/IRPA file(s) that contain the parameters.",
        required=True,
    )
    parser.add_argument(
        "-e",
        "--export",
        action="store_true",
        default=False,
        help="export the model in tp1 mode.",
    )
    parser.add_argument(
        "-a",
        "--artifact-dir",
        type=Path,
        help="the location where the artifacts (sharded weights) should be saved. Defaults to EXPORT_DIR/artifacts/",
    )
    parser.add_argument(
        "-s",
        "--shard",
        action="store_true",
        help="shard the weight file in tp8 mode and export to MLIR.",
    )
    parser.add_argument(
        "-b",
        "--batch-sizes",
        type=str,
        help="batch sizes for export. Multiple batch sizes should be separated by a ','.",
    )
    parser.add_argument(
        "-i",
        "--ir",
        type=Path,
        help="location for the MLIR to be compiled, if compilation is done independently.",
    )
    parser.add_argument(
        "-p",
        "--tensor-parallel",
        type=int,
        default=1,
        choices=[1, 8],
        help="tensor parallel size (required for independent compilation).",
    )
    return parser


if __name__ == "__main__":
    parser = _get_argparser()
    main(parser.parse_args(args=None if sys.argv[1:] else ["--help"]))
