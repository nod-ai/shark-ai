# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
import pytest

from amdsharktank.utils import testing


@pytest.fixture(scope="module")
def temp_dir():
    with testing.temporary_directory(__name__) as td:
        yield Path(td)


@pytest.fixture(scope="module")
def punet_goldens(temp_dir):
    from huggingface_hub import hf_hub_download

    REPO_ID = "amd-shark/sharktank-goldens"
    REVISION = "1d4cb6c452d15a180c1928246848128cdff5ddc8"

    def download(filename):
        return hf_hub_download(
            repo_id=REPO_ID,
            subfolder="punet",
            filename=filename,
            revision=REVISION,
            cache_dir=temp_dir / "huggingface" / "hub",
        )

    return {
        "inputs.safetensors": download("classifier_free_guidance_inputs.safetensors"),
        "outputs.safetensors": download(
            "classifier_free_guidance_fp16_outputs.safetensors"
        ),
        "outputs_int8.safetensors": download(
            "classifier_free_guidance_int8_outputs.safetensors"
        ),
        "outputs_int8_emulated.safetensors": download(
            "classifier_free_guidance_int8_emulated_outputs.safetensors"
        ),
    }


################################################################################
# FP16 Dataset
################################################################################


@pytest.fixture(scope="module")
def sdxl_fp16_base_files(temp_dir):
    from huggingface_hub import hf_hub_download

    REPO_ID = "stabilityai/stable-diffusion-xl-base-1.0"
    REVISION = "76d28af79639c28a79fa5c6c6468febd3490a37e"

    def download(filename):
        return hf_hub_download(
            repo_id=REPO_ID,
            subfolder="unet",
            filename=filename,
            revision=REVISION,
            cache_dir=temp_dir / "huggingface" / "hub",
        )

    return {
        "config.json": download("config.json"),
        "params.safetensors": download("diffusion_pytorch_model.fp16.safetensors"),
    }


@pytest.fixture(scope="module")
def sdxl_fp16_dataset(sdxl_fp16_base_files, temp_dir):
    from amdsharktank.tools import import_hf_dataset

    dataset = temp_dir / "sdxl_fp16_dataset.irpa"
    import_hf_dataset.main(
        [
            f"--config-json={sdxl_fp16_base_files['config.json']}",
            f"--params={sdxl_fp16_base_files['params.safetensors']}",
            f"--output-irpa-file={dataset}",
        ]
    )
    yield dataset


################################################################################
# INT8 Dataset
################################################################################


@pytest.fixture(scope="module")
def sdxl_int8_base_files(temp_dir):
    from huggingface_hub import hf_hub_download

    REPO_ID = "amd-shark/sdxl-quant-int8"
    SUBFOLDER = "mi300_all_sym_8_step14_fp32"
    REVISION = "efda8afb35fd72c1769e02370b320b1011622958"

    def download(filename):
        return hf_hub_download(
            repo_id=REPO_ID,
            subfolder=SUBFOLDER,
            filename=filename,
            revision=REVISION,
            cache_dir=temp_dir / "huggingface" / "hub",
        )

    return {
        "config.json": download("config.json"),
        "params.safetensors": download("params.safetensors"),
        "quant_params.json": download("quant_params.json"),
    }


@pytest.fixture(scope="module")
def sdxl_int8_dataset(sdxl_int8_base_files, temp_dir):
    from amdsharktank.models.punet.tools import import_brevitas_dataset

    dataset = temp_dir / "sdxl_int8_dataset.irpa"
    import_brevitas_dataset.main(
        [
            f"--config-json={sdxl_int8_base_files['config.json']}",
            f"--params={sdxl_int8_base_files['params.safetensors']}",
            f"--quant-params={sdxl_int8_base_files['quant_params.json']}",
            f"--output-irpa-file={dataset}",
        ]
    )
    yield dataset


################################################################################
# Export fixtures
################################################################################


@pytest.fixture(scope="module")
def sdxl_fp16_export_mlir(sdxl_fp16_dataset, temp_dir):
    from amdsharktank.models.punet.tools import run_punet

    output_path = temp_dir / "sdxl_fp16_export_mlir.mlir"
    print(f"Exporting to {output_path}")
    run_punet.main(
        [
            f"--irpa-file={sdxl_fp16_dataset}",
            "--dtype=float16",
            f"--device=cpu",
            f"--export={output_path}",
        ]
    )
    return output_path


@pytest.mark.punet_quick_fp16
@pytest.mark.model_punet
@pytest.mark.export
def test_sdxl_export_fp16_mlir(sdxl_fp16_export_mlir):
    print(f"Exported: {sdxl_fp16_export_mlir}")


@pytest.fixture(scope="module")
def sdxl_int8_export_mlir(sdxl_int8_dataset, temp_dir):
    from amdsharktank.models.punet.tools import run_punet

    output_path = temp_dir / "sdxl_int8_export_mlir.mlir"
    print(f"Exporting to {output_path}")
    run_punet.main(
        [
            f"--irpa-file={sdxl_int8_dataset}",
            "--dtype=float16",
            f"--device=cpu",
            f"--export={output_path}",
        ]
    )
    return output_path


@pytest.mark.punet_quick_int8
@pytest.mark.model_punet
@pytest.mark.export
def test_sdxl_export_int8_mlir(sdxl_int8_export_mlir):
    print(f"Exported: {sdxl_int8_export_mlir}")


################################################################################
# Eager tests
################################################################################


@pytest.mark.model_punet
@pytest.mark.golden
def test_punet_eager_fp16_validation(punet_goldens, sdxl_fp16_dataset, temp_dir):
    from amdsharktank.models.punet.tools import run_punet

    device = testing.get_best_torch_device()
    output_path = (
        temp_dir / "test_punet_eager_fp16_validation_actual_outputs.safetensors"
    )
    print("Using torch device:", device)
    run_punet.main(
        [
            f"--irpa-file={sdxl_fp16_dataset}",
            "--dtype=float16",
            f"--device={device}",
            f"--inputs={punet_goldens['inputs.safetensors']}",
            f"--outputs={output_path}",
        ]
    )
    # TODO: re-enable golden checks once accuracy is pinned down
    # testing.assert_golden_safetensors(output_path, punet_goldens["outputs.safetensors"])


# Executes eagerly using custom integer kernels.
@pytest.mark.model_punet
@pytest.mark.expensive
@pytest.mark.golden
def test_punet_eager_int8_validation(punet_goldens, sdxl_int8_dataset, temp_dir):
    from amdsharktank.models.punet.tools import run_punet

    # Eager runtime issues keep this from producing reliable results on multi
    # GPU systems, so validate on CPU for now.
    # device = testing.get_best_torch_device()
    device = "cpu"
    output_path = (
        temp_dir / "test_punet_eager_int8_validation_actual_outputs.safetensors"
    )
    print("Using torch device:", device)
    run_punet.main(
        [
            f"--irpa-file={sdxl_int8_dataset}",
            "--dtype=float16",
            f"--device={device}",
            f"--inputs={punet_goldens['inputs.safetensors']}",
            f"--outputs={output_path}",
        ]
    )
    # TODO: re-enable golden checks once accuracy is pinned down
    # testing.assert_golden_safetensors(
    #     output_path, punet_goldens["outputs_int8.safetensors"]
    # )


# Executes using emulated fp kernels for key integer operations.
# Useful for speed/comparison.
@pytest.mark.model_punet
@pytest.mark.golden
def test_punet_eager_int8_emulated_validation(
    punet_goldens, sdxl_int8_dataset, temp_dir
):
    from amdsharktank.models.punet.tools import run_punet

    device = testing.get_best_torch_device()
    output_path = (
        temp_dir
        / "test_punet_eager_int8_emulated_validation_actual_outputs.safetensors"
    )
    print("Using torch device:", device)
    with testing.override_debug_flags(
        {
            "use_custom_iree_kernels": False,
        }
    ):
        run_punet.main(
            [
                f"--irpa-file={sdxl_int8_dataset}",
                "--dtype=float16",
                f"--device={device}",
                f"--inputs={punet_goldens['inputs.safetensors']}",
                f"--outputs={output_path}",
            ]
        )
    # TODO: re-enable golden checks once accuracy is pinned down
    # testing.assert_golden_safetensors(
    #     output_path, punet_goldens["outputs_int8_emulated.safetensors"]
    # )
