from typing import Optional, Union

import torch

from brevitas.quant.scaled_int import (
    Int8WeightPerTensorFloat,
)
from brevitas.quant.experimental.float_quant_fnuz import (
    Fp8e4m3FNUZActPerTensorFloat,
)

string_to_torch = {
    "f8e4m3fnuz": torch.float8_e4m3fnuz,
    "float8_e4m3fnuz": torch.float8_e4m3fnuz,
    "int8": torch.int8,
}

torch_to_brevitas = {
    torch.float8_e4m3fnuz: Fp8e4m3FNUZActPerTensorFloat,
    torch.int8: Int8WeightPerTensorFloat,
}


class QuantizationConfig:
    def __init__(
        self, dtype: Optional[Union[torch.dtype, str]] = torch.float8_e4m3fnuz
    ):
        if isinstance(dtype, str):
            dtype = string_to_torch[dtype]
        self.dtype = dtype
        self.quantization_scheme = torch_to_brevitas[dtype]
