# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import pytest
from pathlib import Path
from sharktank.utils.iree import (
    run_iree_vs_torch_eager,
)
from sharktank.layers import LinearLayer, RMSNormLayer
from sharktank.types import Dataset, Theta
from sharktank.layers.configs import LlamaModelConfig
from sharktank.utils._iree_compile_flags_config import LLM_HIP_COMPILE_FLAGS
from sharktank.utils.testing import is_hip_condition, validate_and_get_irpa_path


class OutputLMHead(torch.nn.Module):
    """Standalone output_lm_head block extracted from PagedLlmModelV1"""

    def __init__(self, theta: Theta, config: LlamaModelConfig):
        super().__init__()
        self.config = config
        self.hp = config.hp

        # Output normalization layer
        self.output_norm = RMSNormLayer(
            theta("output_norm"), epsilon=self.hp.attention_layer_norm_rms_epsilon
        )

        # Output linear layer (language model head)
        self.output_lm_head = LinearLayer(
            theta("output"),
            matmul_kernel=config.matmul_kernel,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Apply normalization
        h_norm = self.output_norm(h)  # output fp16 && wieghts float32

        # Apply final linear transformation
        logits = self.output_lm_head(h_norm)  # output && weights fp16

        return logits


def create_output_lm_head_from_irpa(
    irpa_path: str,
) -> tuple[OutputLMHead, torch.Tensor]:
    """
    Create OutputLMHead module from IRPA file and generate sample input.

    Args:
        irpa_path: Path to the IRPA file

    Returns:
        Tuple of (OutputLMHead module, sample input tensor)
    """
    # Load dataset from IRPA file
    dataset = Dataset.load(Path(irpa_path))

    # Create model config from dataset
    llama_config = LlamaModelConfig.from_dataset(
        dataset=dataset,
        attention_kernel="torch",
        matmul_kernel="sharktank.asm;*",
        activation_dtype=torch.float16,
    )

    # Create the output LM head module
    output_lm_head = OutputLMHead(dataset.root_theta, llama_config)

    # Generate sample input tensor matching expected dimensions
    # Typical shape: [batch_size, seq_len, hidden_dim]
    # TODO: Check if there are other more suitable sizes to test.
    batch_size = 2
    seq_len = 8
    hidden_dim = (
        llama_config.hp.embedding_length
    )  # Use embedding_length instead of model_dim

    sample_input = torch.randn(
        batch_size, seq_len, hidden_dim, dtype=llama_config.activation_dtype
    )

    return output_lm_head, sample_input


# Test cases
@pytest.mark.skipif(f"not ({is_hip_condition})", reason="Test requires HIP device")
@pytest.mark.parametrize("dtype,atol", [(torch.float16, 1e-4)])
@pytest.mark.xfail(
    reason="Numerics Issue, see https://github.com/nod-ai/shark-ai/issues/2455"
)
def test_output_lm_head_iree_vs_eager(request, dtype, atol):
    """
    Test OutputLMHead module comparing IREE vs PyTorch eager execution.

    Use --parameters command line argument to specify the IRPA file path.
    """
    # Validate and get IRPA path
    irpa_path = validate_and_get_irpa_path(request)

    try:
        # Create module and sample input from IRPA
        module, sample_input = create_output_lm_head_from_irpa(irpa_path)
    except Exception as e:
        pytest.skip(f"Failed to load model from IRPA: {e}")

    # Convert to desired dtype
    # module = module.to(dtype)
    sample_input = sample_input.to(dtype)

    # Run IREE vs torch comparison
    run_iree_vs_torch_eager(
        module,
        input_args=(sample_input,),
        atol=atol,
        rtol=0,
        compile_flags=LLM_HIP_COMPILE_FLAGS,
        parameters_path=irpa_path,
    )


@pytest.mark.skipif(f"not ({is_hip_condition})", reason="Test requires HIP device")
@pytest.mark.xfail(
    reason="Numerics Issue, see https://github.com/nod-ai/shark-ai/issues/2455"
)
def test_output_lm_head_mock():
    """
    Mock test with synthetic weights for OutputLMHead functionality.
    Adding this test to work without requiring an IRPA file.
    """
    torch.manual_seed(42)

    # Mock configuration - provide all required parameters
    from sharktank.layers.configs import LlamaHParams

    # Create LlamaHParams with all required parameters
    hp = LlamaHParams(
        model_arch="llama",
        context_length=2048,
        embedding_length=512,  # hidden dimension
        block_count=6,
        feed_forward_length=2048,
        attention_head_count=8,
        attn_head_dim=64,
        attention_layer_norm_rms_epsilon=1e-6,
        attention_head_count_kv=8,
        vocab_size=32000,
    )

    # Create mock config
    config = LlamaModelConfig(
        hp=hp,
        activation_dtype=torch.float16,
        # attention_dtype=torch.float32,
    )

    # Create mock theta with synthetic weights
    from sharktank.types import DefaultPrimitiveTensor

    # Mock output_norm weights
    output_norm_weight = torch.randn(hp.embedding_length, dtype=torch.float32)

    # Mock output (lm_head) weights
    output_weight = torch.randn(hp.vocab_size, hp.embedding_length, dtype=torch.float16)

    # Create theta structure
    theta_dict = {
        "output_norm": {"weight": DefaultPrimitiveTensor(data=output_norm_weight)},
        "output": {"weight": DefaultPrimitiveTensor(data=output_weight)},
    }

    theta = Theta(theta_dict)

    # Create module
    module = OutputLMHead(theta, config)

    # Create sample input
    batch_size, seq_len = 2, 8
    sample_input = torch.randn(
        batch_size, seq_len, hp.embedding_length, dtype=torch.float32
    )
    # Run IREE vs torch comparison
    run_iree_vs_torch_eager(
        module,
        input_args=(sample_input,),
        atol=1e-4,
        rtol=0,
        compile_flags=LLM_HIP_COMPILE_FLAGS,
    )
