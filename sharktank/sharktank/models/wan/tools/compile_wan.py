import logging
from pathlib import Path
import os

# Import the IREE compiler API
try:
    from iree.compiler.tools import compile_file
except ImportError:
    logging.error(
        "The 'iree-compiler' package is not installed. Please install it with 'pip install iree-compiler'."
    )
    exit()


# Define common compiler flags as a top-level constant for clarity and reusability.
COMMON_WAN_ROCM_FLAGS = [
    "--iree-hip-target=gfx942",
    "--iree-execution-model=async-external",
    "--iree-dispatch-creation-enable-fuse-horizontal-contractions=0",
    "--iree-flow-inline-constants-max-byte-length=16",
    "--iree-global-opt-propagate-transposes=1",
    "--iree-opt-const-eval=0",
    "--iree-opt-outer-dim-concat=1",
    "--iree-opt-aggressively-propagate-transposes=1",
    "--iree-dispatch-creation-enable-aggressive-fusion",
    "--iree-hal-force-indirect-command-buffers",
    "--iree-llvmgpu-enable-prefetch=1",
    "--iree-opt-data-tiling=0",
    "--iree-hal-memoization=1",
    # "--iree-opt-strip-assertions",
    "--iree-codegen-llvmgpu-early-tile-and-fuse-matmul=1",
    "--iree-stream-resource-memory-model=discrete",
    "--iree-vm-target-truncate-unsupported-floats=1",
    "--iree-opt-level=O3",
]


def get_compile_options(component, model_name, dims, dtype):
    """
    Generates the input filename and compiler arguments for a given component.

    This function centralizes the logic for creating compilation tasks, handling
    component-specific naming conventions.

    Args:
        component (str): The name of the component (e.g., 'clip', 't5').
        model_name (str): The base name of the model.
        dims (str): The dimensions string (e.g., '512x512').
        dtype (str): The data type (e.g., 'bf16').

    Returns:
        tuple[str, dict]: A tuple containing the input MLIR filename and a
                          dictionary of compiler arguments for that file.

    Raises:
        ValueError: If an unknown component is provided.
    """
    component_name_map = {
        "clip": f"clip_{dims}",
        "vae": f"vae_{dims}",
        "transformer": f"transformer_{dims}",
        "t5": "umt5xxl",
    }

    if component not in component_name_map:
        raise ValueError(f"Unknown component specified: '{component}'")

    component_part = component_name_map[component]
    base_filename = f"{model_name}_{component_part}_{dtype}"
    input_file = f"{base_filename}.mlir"
    output_file = f"{base_filename}_gfx942.vmfb"

    compile_args = {
        "target_backends": ["rocm"],
        "output_file": output_file,
        "extra_args": COMMON_WAN_ROCM_FLAGS.copy(),  # Use a copy to avoid mutation
    }

    return input_file, compile_args


def run_compilation(input_file, **kwargs):
    """
    Invokes the IREE compiler on a given input file using the Python API.

    Args:
        input_file (str): Path to the MLIR input file.
        **kwargs: Keyword arguments for compiler options.
    """
    is_debug = "--mlir-print-debuginfo=1" in kwargs.get("extra_args", [])
    prefix = "🛠️ [DEBUG] " if is_debug else "🚀"
    logging.info(f"{prefix} Starting compilation for '{input_file}'...")

    try:
        compile_file(input_file=input_file, **kwargs)
        logging.info(f"   Successfully compiled '{input_file}'")
    except Exception as e:
        # Log the detailed error here and re-raise it to be caught by the main loop.
        logging.error(f"   Error during compilation of '{input_file}': {e}")
        raise


def rerun_failed_jobs_with_debug(failures, compile_tasks):
    """
    Re-runs failed compilations with additional debug flags.

    Args:
        failures (list): A list of tuples (filename, error) for failed jobs.
        compile_tasks (dict): The original dictionary of compilation tasks.
    """
    print("\n" + "=" * 40)
    print("         🛠️ DEBUG RE-RUNNING FAILED JOBS 🛠️")
    print("=" * 40)

    debug_successes = []
    debug_failures = []

    for mlir_file, _ in failures:
        # Get a copy of the original args
        original_args = compile_tasks[mlir_file].copy()
        original_extra_args = original_args.get("extra_args", []).copy()

        # Create a unique directory for debug dumps
        debug_dump_dir = Path(f"debug_dump_{Path(mlir_file).stem}")
        debug_dump_dir.mkdir(exist_ok=True)
        logging.info(
            f"Dumping debug artifacts for '{mlir_file}' to '{debug_dump_dir}/'"
        )

        # Define and add new debug flags
        debug_flags = [
            f"--iree-hal-dump-executable-files-to={debug_dump_dir}",
            "--mlir-print-debuginfo=1",
            "--iree-opt-strip-assertions=0",
        ]

        # Combine args, ensuring no duplicate flags
        debug_extra_args = original_extra_args + debug_flags
        debug_command_args = {**original_args, "extra_args": debug_extra_args}

        try:
            run_compilation(input_file=mlir_file, **debug_command_args)
            debug_successes.append(mlir_file)
        except Exception as e:
            debug_failures.append((mlir_file, e))

    # --- Debug Re-run Report ---
    print("\n" + "=" * 40)
    print("         📊 DEBUG RE-RUN REPORT 📊")
    print("=" * 40)
    print(f"\nTotal files re-attempted: {len(failures)}")
    print(f"   ✅ Successes: {len(debug_successes)}")
    print(f"   ❌ Failures:  {len(debug_failures)}")
    if debug_failures:
        print("\n   Failures persisted even with debug flags.")
    print("=" * 40)


def main():
    """
    Main function to find and compile .mlir files, then report on successes and failures.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # --- Configuration ---
    model_name = "wan2_1"
    dims = "512x512"
    dtype = "bf16"
    components_to_compile = ["clip", "t5", "transformer", "vae"]

    # --- Task Generation ---
    # Dynamically build the compilation tasks using the helper function.
    compile_tasks = {}
    for component in components_to_compile:
        try:
            input_file, compile_args = get_compile_options(
                component, model_name, dims, dtype
            )
            compile_tasks[input_file] = compile_args
        except ValueError as e:
            logging.error(e)

    # --- File Discovery ---
    found_files = []
    for mlir_file in compile_tasks.keys():
        if Path(mlir_file).is_file():
            found_files.append(mlir_file)
        else:
            logging.warning(f"File not found, skipping: '{mlir_file}'.")

    if not found_files:
        logging.info("No .mlir files found to compile. Exiting.")
        return

    # --- Compilation Phase ---
    logging.info("-" * 20)
    logging.info(
        f"Found {len(found_files)} .mlir file(s) to process: {', '.join(found_files)}"
    )
    logging.info("-" * 20)

    successes = []
    failures = []  # Will store tuples of (filename, error_message)

    # Run compilation for each existing file, recording the outcome.
    for mlir_file in found_files:
        command_args = compile_tasks[mlir_file]
        output_vmfb = Path(command_args["output_file"])

        if output_vmfb.is_file():
            logging.info(
                f"✅ Skipping '{mlir_file}': Output '{output_vmfb}' already exists."
            )
            successes.append(mlir_file)
            continue

        try:
            run_compilation(input_file=mlir_file, **command_args)
            successes.append(mlir_file)
        except Exception as e:
            failures.append((mlir_file, e))

    # --- Final Report ---
    print("\n" + "=" * 40)
    print("           📊 COMPILATION REPORT 📊")
    print("=" * 40)
    print(f"\nTotal MLIR files processed: {len(found_files)}")
    print(f"   ✅ Artifacts ready: {len(successes)}")
    print(f"   ❌ Failures:        {len(failures)}")

    if failures:
        print("\n" + "-" * 40)
        print("                 Detailed Failures")
        print("-" * 40)
        for mlir_file, error in failures:
            print(f"\nFile: {mlir_file}")
            print(f"   └─ Error: {error}")

    print("\n" + "=" * 40)

    # --- Optional Debug Re-run ---
    if failures:
        print()  # Add a newline for spacing
        rerun_choice = input(
            "Would you like to re-run the failed jobs with debug options? (yes/no): "
        )
        if rerun_choice.lower().strip() in ["y", "yes"]:
            rerun_failed_jobs_with_debug(failures, compile_tasks)
        # Exit with a non-zero status code if there were any initial failures.
        exit(1)


if __name__ == "__main__":
    main()
