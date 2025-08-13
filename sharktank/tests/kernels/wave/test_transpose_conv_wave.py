import unittest
import torch
from sharktank.kernels.wave.transpose_conv import wave_transpose_conv
import iree.turbine.aot as aot


class transpose_conv_wave(unittest.TestCase):
    def test_transpose_conv_wave(self):
        class WaveTransposeConvModule(torch.nn.Module):
            def forward(
                self,
                x,
                we,
                out,
                upsamp_stride,
            ):
                return wave_transpose_conv(x, we, out, upsamp_stride)

        export = aot.export(
            WaveTransposeConvModule(),
            args=(
                torch.empty((1, 3, 17, 17), dtype=torch.float16),
                torch.empty((3, 3, 3, 3), dtype=torch.float16),
                torch.empty((1, 3, 32, 32), dtype=torch.float32),
                torch.tensor(2, dtype=torch.int32),
            ),
        )
        export.verify()
        mlir_asm = str(export.mlir_module)
        self.assertIn(
            ("func.func @main"),
            mlir_asm,
        )
        self.assertIn(
            ("stream.executable private @trans_conv"),
            mlir_asm,
        )
        self.assertIn(
            (
                "func.func private @wave_trans_conv_n_1_c_3_h_17_w_17_nf_3_cf_3_hf_3_wf_3_upStride_2"
            ),
            mlir_asm,
        )


if __name__ == "__main__":
    unittest.main()
