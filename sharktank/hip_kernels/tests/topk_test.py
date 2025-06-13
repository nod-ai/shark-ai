import iree.runtime
import numpy
import pytest
import subprocess
import torch


_topk_template = """
func.func @main(%arg0 : tensor<?x{d1}x{dtype}>, %arg1: tensor<?x{d1}xi32>) -> (tensor<?x{k}x{dtype}>, tensor<?x{k}xi32>) {
  %c0 = arith.constant 0 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x{d1}x{dtype}>

  %out_values = tensor.empty(%d0) : tensor<?x{k}x{dtype}>
  %out_indices = tensor.empty(%d0) : tensor<?x{k}xi32>

  %0:2 = iree_linalg_ext.topk
        dimension(1)
        ins(%arg0, %arg1: tensor<?x{d1}x{dtype}>, tensor<?x{d1}xi32>)
        outs(%out_values, %out_indices : tensor<?x{k}x{dtype}>, tensor<?x{k}xi32>) {
        ^bb0(%arg2: {dtype}, %arg3: {dtype}):
          %0 = arith.cmpf ogt, %arg2, %arg3 : {dtype}
          iree_linalg_ext.yield %0 : i1
        } -> tensor<?x{k}x{dtype}>, tensor<?x{k}xi32>
  return %0#0, %0#1 : tensor<?x{k}x{dtype}>, tensor<?x{k}xi32>
}
"""


def get_topk_template(k: int, dtype: str, d1):
    t = _topk_template
    t = t.replace("{k}", str(k))
    t = t.replace("{dtype}", str(dtype))
    t = t.replace("{d1}", str(d1))
    return t


@pytest.fixture(scope="session")
def hip_kernel_build(request):
    ret = request.config.option.hip_kernel_build
    if ret is None:
        raise ValueError("--hip-kernel-build not specified")
    return ret


@pytest.fixture(scope="session")
def test_device(request):
    ret = request.config.option.iree_hip_target
    if ret is None:
        raise ValueError("--test_device not specified")
    return ret


def test_topk_f16_test(tmp_path, test_device, hip_kernel_build):
    run_topk(
        tmp_path=tmp_path,
        test_device=test_device,
        hip_kernel_build=hip_kernel_build,
        k=8,
        bs=1,
        d1=64,
        dtype=torch.float16,
        dtype_mlir="f16",
    )


def test_topk_f32_test(tmp_path, test_device, hip_kernel_build):
    run_topk(
        tmp_path=tmp_path,
        test_device=test_device,
        hip_kernel_build=hip_kernel_build,
        k=8,
        bs=1,
        d1=64,
        dtype=torch.float32,
        dtype_mlir="f32",
    )


def run_topk(tmp_path, test_device, hip_kernel_build, k, bs, d1, dtype, dtype_mlir):
    filename = f"{tmp_path}/topk_{dtype_mlir}.mlir"
    with open(filename, "wt") as f:
        topk_mlir = get_topk_template(k=k, d1=d1, dtype=dtype_mlir)
        f.write(topk_mlir)
        f.close()

    command = [
        "iree-compile",
        filename,
        "--iree-hal-target-device=hip",
        f"--iree-hip-target={test_device}",
        "--mlir-print-ir-after=iree-preprocessing-transform-interpreter",
        f"--iree-preprocessing-transform-spec-filename={hip_kernel_build}/specs/topk_{dtype_mlir}_spec.mlir",
    ]
    process = subprocess.run(command, capture_output=True)
    stdout = process.stdout
    stderr = process.stderr.decode("utf-8")

    # Check that the substition worked:
    has_linalg_ext_topk = "iree_linalg_ext.topk" in stderr
    has_hal_disatch_extern = "hal.dispatch.extern" in stderr
    assert has_hal_disatch_extern
    assert not has_linalg_ext_topk

    hal_driver = iree.runtime.get_driver("hip")

    device_infos = hal_driver.query_available_devices()
    device = hal_driver.create_device(device_infos[0]["device_id"])
    config = iree.runtime.Config(device=device)
    vm_module = iree.runtime.VmModule.from_flatbuffer(
        config.vm_instance, stdout, warn_if_copy=False
    )
    bound_module = iree.runtime.load_vm_module(vm_module, config)

    arg0 = torch.arange(d1).repeat(bs, 1).to(dtype=dtype) * 17 % (bs * d1)
    arg1 = torch.arange(d1, dtype=torch.int32)[None, :].repeat(bs, 1)

    ref0, ref1 = torch.topk(torch.asarray(arg0), k=k)
    res0, res1 = bound_module.main(arg0, arg1)

    res0 = numpy.asarray(res0)
    res1 = numpy.asarray(res1)
    sort = numpy.argsort(res0, axis=-1)[:, ::-1]
    res0 = res0[numpy.arange(bs), sort]
    res1 = res1[numpy.arange(bs), sort]

    assert numpy.isclose(res1, ref1).all()
    assert numpy.isclose(res0, ref0).all()
