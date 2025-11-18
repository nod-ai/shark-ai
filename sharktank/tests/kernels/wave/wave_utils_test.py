from iree.compiler.ir import Context, Module
from iree.compiler.passmanager import PassManager
from amdsharktank.kernels.wave.utils import get_wave_module_body_asm
import unittest


def run_inliner_pass(mlir_str: str) -> str:
    with Context() as ctx:
        module = Module.parse(mlir_str)
        pm = PassManager.parse("builtin.module(inline)")
        pm.run(module.operation)
        return str(module)


class WaveUtilsTest(unittest.TestCase):
    def test_get_wave_module_body_asm(self):
        """
        This test shows why we must use `get_wave_module_body_asm()`:
        to convert the public `func.func` emitted by wave_compile into a
        `private` one so that it gets removed after inlining.

        If the public `func.func` from `wave_compile()` is retained
        after inlining, it causes duplicate/unused functions at
        IREE runtime during eager execution. Making it private allows
        the inline pass to remove it after use.
        """

        # Original wave_compile() result with a public func.func
        wave_compile_body = """
        stream.executable private @base_attention {
            builtin.module {
                func.func @base_attention(%arg0: !stream.binding) attributes {} {
                    return
                }
            }
        }
        func.func @wave_func_call(%arg0: tensor<4x32x128x128xf16>) -> tensor<4x32x128x128xf16> {
            %0 = flow.dispatch @base_attention::@base_attention(%arg0) : (tensor<4x32x128x128xf16>) -> tensor<4x32x128x128xf16>
            return %0 : tensor<4x32x128x128xf16>
        }
        """

        wave_compile_mlir = "module {" + wave_compile_body + "}"

        # util wrapper that calls @wave_func_call
        util_func_mlir = """
        util.func private @util_func_wave_bhsd_flash_attention(%arg0: tensor<4x32x128x128xf16>) -> tensor<4x32x128x128xf16> {
            %0 = func.call @wave_func_call(%arg0) : (tensor<4x32x128x128xf16>) -> tensor<4x32x128x128xf16>
            util.return %0 : tensor<4x32x128x128xf16>
        }
        """

        # In the public case, we use the original wave_compile_body
        public_wave_func_mlir = (
            """
        module {
            func.func @main(%arg0: !torch.vtensor<[4,32,128,128],f16>) -> !torch.vtensor<[4,32,128,128],f16> attributes {torch.assume_strict_symbolic_shapes} {
                %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4,32,128,128],f16> -> tensor<4x32x128x128xf16>
                %1 = util.call @util_func_wave_bhsd_flash_attention(%0) : (tensor<4x32x128x128xf16>) -> tensor<4x32x128x128xf16>
                %2 = torch_c.from_builtin_tensor %1 : tensor<4x32x128x128xf16> -> !torch.vtensor<[4,32,128,128],f16>
                return %2 : !torch.vtensor<[4,32,128,128],f16>
            }
        """
            + wave_compile_body
            + util_func_mlir
            + "}"
        )

        # In the private case, re-emit the wave_compile body with get_wave_module_body_asm()
        with Context() as ctx:
            asm_module = Module.parse(wave_compile_mlir)
            asm_body = get_wave_module_body_asm(asm_module)

        private_wave_func_mlir = (
            """
        module {
            func.func @main(%arg0: !torch.vtensor<[4,32,128,128],f16>) -> !torch.vtensor<[4,32,128,128],f16> attributes {torch.assume_strict_symbolic_shapes} {
                %0 = torch_c.to_builtin_tensor %arg0 : !torch.vtensor<[4,32,128,128],f16> -> tensor<4x32x128x128xf16>
                %1 = util.call @util_func_wave_bhsd_flash_attention(%0) : (tensor<4x32x128x128xf16>) -> tensor<4x32x128x128xf16>
                %2 = torch_c.from_builtin_tensor %1 : tensor<4x32x128x128xf16> -> !torch.vtensor<[4,32,128,128],f16>
                return %2 : !torch.vtensor<[4,32,128,128],f16>
            }
        """
            + asm_body
            + util_func_mlir
            + "}"
        )

        inlined_public = run_inliner_pass(public_wave_func_mlir)
        inlined_private = run_inliner_pass(private_wave_func_mlir)

        # In both cases, the main function remains
        self.assertIn("func.func @main", inlined_public)
        self.assertIn("func.func @main", inlined_private)

        # In both cases, the stream.executable remains
        self.assertIn(
            ("stream.executable private @base_attention"),
            inlined_public,
        )
        self.assertIn(
            ("stream.executable private @base_attention"),
            inlined_private,
        )

        # Public wave_func_call remains, which becomes a duplicate
        # of func.func @main after async invocations
        self.assertIn("func.func @wave_func_call", inlined_public)
        # Private wave_func_call gets removed
        self.assertNotIn("func.func private @wave_func_call", inlined_private)

        # Util function gets removed in both cases
        self.assertNotIn(
            "util.func private @util_func_wave_bhsd_flash_attention", inlined_private
        )
        self.assertNotIn(
            "util.func private @util_func_wave_bhsd_flash_attention", inlined_public
        )
