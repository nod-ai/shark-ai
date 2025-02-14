import logging
from pathlib import Path
import math
import pytest
import shortfin as sf
import shortfin.array as sfnp
from shortfin_apps.llm.components.service import (
    GenerateService,
    InferenceExecutorProcess,
)
from shortfin_apps.llm.components.messages import InferenceExecRequest, InferencePhase
from shortfin_apps.llm.components.config_struct import ModelParams, ServerParams
from shortfin_apps.llm.components.manager import SystemManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

art_path = Path("")
weights_path = art_path / "model.gguf"
vmfb_path = art_path / "model.vmfb"
config_path = art_path / "config.json"

# Create system manager
sysman = SystemManager(
    device="hip",
    device_ids=None,  # Full visibility
    async_allocs=True,
    amdgpu_allocators=None,
)

sysman.start()

model_params = ModelParams.load_json(config_path)

server_params = ServerParams()
server_params.prefix_sharing_algorithm = "default"  # Adjust as needed

service = GenerateService(
    name="notebook-service",
    sysman=sysman,
    tokenizer=None,  # No tokenizer needed for basic setup
    model_params=model_params,
    server_params=server_params,
    program_isolation="per_call",
)

# Load inference module and parameters
service.load_inference_module(vmfb_path)
service.load_inference_parameters(weights_path, parameter_scope="model")

# Start the service
service.start()

from shortfin_apps.llm.components.messages import InferenceExecRequest, InferencePhase

pytestmark = pytest.mark.parametrize(
    "model_artifacts,server",
    [
        ["llama3.1_8b", {"prefix_sharing": "none"}],
        ["llama3.1_8b", {"prefix_sharing": "trie"}],
    ],
    indirect=True,
)


def test_batch_sizes_same_inputs_same_outputs():
    """
    Tests submitting various numbers of inference requests with the same input token ids,
    and check to make sure no matter what the batch size is, we get the same output.
    """
    token_ids = [1, 2, 3]
    max_batch_size = 3
    reqs = []

    b = service.batcher

    # create
    for bi in range(max_batch_size):
        token_ids = [x for x in token_ids]  # copy list
        reqs[i] = InferenceExecRequest(
            phase=InferencePhase.PREFILL, input_token_ids=token_ids, rid=bi
        )

    for r in reqs:
        b.submit(r)

    # submit
    exec_process = InferenceExecutorProcess(
        service,
        InferencePhase.PREFILL,
        seq_stri=model_params.block_seq_stride,
        page_tables=service.page_cache.page_pool.page_tables,
    )
    for r in reqs:
        exec_process.exec_requests.append(r)

    exec_process.launch()

    # check to make sure the
    # run one batch


# Now the service is ready to use
# When done, you can shut down with:
# service.shutdown()
# sysman.shutdown()
