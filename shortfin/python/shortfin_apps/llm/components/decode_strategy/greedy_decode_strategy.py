import logging

import shortfin.array as sfnp

from .base_decode_strategy import DecodeStrategy, DecodeStrategyConfig
from ..messages import LlmInferenceExecRequest, InferencePhase


logger = logging.getLogger(__name__)


class GreedyDecodeStrategy(DecodeStrategy):
    def __init__(
        self,
        decode_strategy_config: DecodeStrategyConfig,
    ):
        self._decode_strategy_config = decode_strategy_config

    @property
    def decode_strategy_config(self):
        return self._decode_strategy_config

    async def decode(
        self,
        exec_req: LlmInferenceExecRequest,
    ):
        config = self.decode_strategy_config
        for _ in range(config.max_completion_tokens):
            exec_req.reset(InferencePhase.DECODE)
            config.batcher_callback(exec_req)
            await exec_req.done
            token = sfnp.argmax(exec_req.result_logits)
            token_int = token.items[0]
            if exec_req.llm_inference_metrics is not None:
                config.streaming_callback(token_int, exec_req.llm_inference_metrics)
            else:
                config.streaming_callback(token_int)
            if token_int == config.eos_token_id:
                break
            exec_req.input_token_ids.append(token_int)
            exec_req.output_token_ids.append(token_int)
            exec_req.start_position += 1

        exec_req.llm_inference_metrics.set_end_time()
