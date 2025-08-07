
import unittest
import torch
from sharktank.kernels.wave.transpose_conv import wave_transpose_conv
import iree.turbine.aot as aot


class transpose_conv_wave(unittest.TestCase):
    def test_transpose_conv_wave(self):
        class WaveTransposeConvModule(torch.nn.Module):
            def forward(
                self,
                x_shape,
                we_shape,
                out_shape,
                upsamp_stride,
            ):
                return wave_transpose_conv(x_shape, we_shape, out_shape, upsamp_stride)
            

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
        # mlir_asm = str(export.mlir_module)
        # print(mlir_asm)