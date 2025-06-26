from sharktank.models.llama4 import testing
from sharktank.types import *
from sharktank.utils.evaluate import *

from sharktank.models.llama.testing import make_random_llama_theta
from sharktank.utils.testing import TempDirTestBase
from sharktank.utils.testing import is_mi300x, IreeVsEagerLLMTester, TempDirTestBase
import torch
import logging

import pytest

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

iree_compile_flags = []  # TODO; fill this if we need any flag


@pytest.mark.usefixtures("get_iree_flags", "device")
class TestLlama4IreeEager(TempDirTestBase):
    def helper_run(self, dtype, atol, rtol):
        seed = 1234
        random.seed(seed)
        torch.manual_seed(seed)
        config = testing.make_toy_model_config(dtype=dtype)
        theta = make_random_llama_theta(config=config)

        tester = IreeVsEagerLLMTester(
            work_dir=self._temp_dir,
            theta=theta,
            config=config,
            torch_device=self.device,
            iree_device=self.iree_device,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
            skip_decode=True,
        )
        tester.run_and_compare_iree_vs_eager(atol=atol, rtol=rtol)

    def testUnshardedToySizedModelIREEVsEager(self):
        for dtype, atol, rtol in [(torch.float16, 1e-1, 1e-1)]:
            self.helper_run(dtype=dtype, atol=atol, rtol=rtol)
