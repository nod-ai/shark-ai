import concurrent.futures
import json
import logging
import numpy as np
import pytest
import requests
from torch import Tensor

pytest.importorskip("sentence_transformers")

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from typing import Dict, List, Optional

from ...datasets import Dataset, DatasetRequest, DatasetTypes, AvailableDatasets
from ...model_management import AccuracyValidationException, ModelConfig
from ...server_management import ServerConfig


logger = logging.getLogger(__name__)


@dataclass
class SamplingParams:
    max_completion_tokens: int = 250
    temperature: float = 1.0


@dataclass
class IncorrectPrompt:
    prompt: str
    expected: str
    actual: str
    accuracy: float

    def __str__(self) -> str:
        return (
            f"{'-' * 80} Prompt {'-' * 80}\n{self.prompt}\n"
            f"{'-' * 80} Expected {'-' * 80}\n{self.expected}\n"
            f"{'-' * 80} Actual {'-' * 80}\n{self.actual}\n"
            f"Accuracy: {self.accuracy:.2f}%\n"
        )


@dataclass
class AccuracyResults:
    total_prompts: int
    total_correct: int
    total_incorrect: int
    accuracy: float
    incorrect_prompts: Optional[List[IncorrectPrompt]] = None

    def __str__(self) -> str:
        incorrect_prompts = "\n".join(str(p) for p in self.incorrect_prompts or [])
        return (
            "Results:\n"
            f"{'-' * 80} Incorrect Prompts {'-' * 80}\n"
            f"{incorrect_prompts}\n"
            f"\tTotal Prompts: {self.total_prompts}\n"
            f"\tTotal Correct: {self.total_correct}\n"
            f"\tTotal Incorrect: {self.total_incorrect}\n"
            f"\tAccuracy: {self.accuracy:.2f}%\n"
        )


BASIC = DatasetRequest(
    dataset_type=DatasetTypes.LOCAL,
    dataset=AvailableDatasets.BASIC,
    dataset_path=Path(__file__).parent / "datasets.json",
)

CHUNKED_PREFILL = DatasetRequest(
    dataset_type=DatasetTypes.LOCAL,
    dataset=AvailableDatasets.CHUNKED_PREFILL,
    dataset_path=Path(__file__).parent / "datasets.json",
)

PREFIX_MATCHING = DatasetRequest(
    dataset_type=DatasetTypes.LOCAL,
    dataset=AvailableDatasets.PREFIX_MATCHING,
    dataset_path=Path(__file__).parent / "datasets.json",
)

ALL = DatasetRequest(
    dataset_type=DatasetTypes.LOCAL,
    dataset=AvailableDatasets.ALL,
    dataset_path=Path(__file__).parent / "datasets.json",
)


ACCURACY_THRESHOLD = 0.99


def load_comparison_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    return model


class TestLLMAccuracy:
    def _check_health(self, base_url: str):
        resp = requests.get(f"{base_url}/health")
        assert resp.status_code == 200

    def _get_embeddings(
        self, comparison_model: SentenceTransformer, sentences: List[str]
    ) -> np.ndarray:
        return comparison_model.encode(sentences)

    def _compute_similarity(self, embedding_1: Tensor, embedding_2: Tensor):
        return util.pytorch_cos_sim(embedding_1, embedding_2).item()

    def _validate_response(
        self,
        comparison_model: SentenceTransformer,
        dataset: Dataset,
        results: List[Dict],
    ):
        total_prompts = dataset.size
        total_correct = 0
        total_incorrect = 0
        incorrect_prompts = []

        # Gather prompts and expected generations
        expected_generations = []
        prompts = []
        actual_generations = []
        for result in results:
            responses = result["responses"]
            for response in responses:
                prompt = response["prompt"]
                actual_generation = response["responses"][0]["text"]
                actual_generations.append(actual_generation)
                expected_generation = dataset.get_expected_generation(prompt)
                prompts.append(prompt)
                expected_generations.append(expected_generation)

        # Get embeddings for expected generations
        expected_embeddings = self._get_embeddings(
            comparison_model, expected_generations
        )
        # Get embeddings for actual generations
        actual_embeddings = self._get_embeddings(comparison_model, actual_generations)

        for index, (actual_generation, expected_generation) in enumerate(
            zip(actual_generations, expected_generations)
        ):
            prompt = prompts[index]
            expected_embedding = expected_embeddings[index]
            actual_embedding = actual_embeddings[index]

            accuracy = self._compute_similarity(expected_embedding, actual_embedding)
            if accuracy < ACCURACY_THRESHOLD:
                logger.error(
                    f"Mismatch for prompt: {prompt}\n, With Accuracy: {accuracy:.2f}\n"
                    f"{'-' * 80} Expected {'-' * 80}\n{expected_generation}\n"
                    f"{'-' * 80} Actual {'-' * 80}\n{actual_generation}\n"
                )
                total_incorrect += 1
                incorrect_prompts.append(
                    IncorrectPrompt(
                        prompt=prompt,
                        expected=expected_generation,
                        actual=actual_generation,
                        accuracy=accuracy,
                    )
                )
                continue

            total_correct += 1

        return AccuracyResults(
            total_prompts=total_prompts,
            total_correct=total_correct,
            total_incorrect=total_incorrect,
            accuracy=(total_correct / total_prompts) * 100,
            incorrect_prompts=incorrect_prompts,
        )

    def _send_request(
        self, base_url: str, prompts_batch: List[str], sampling_params: SamplingParams
    ):
        endpoint = f"{base_url}/generate"
        payload = {
            "text": prompts_batch,
            "sampling_params": asdict(sampling_params),
        }

        logger.info(f"Sending request with payload: {json.dumps(payload, indent=2)}")
        response = requests.post(endpoint, json=payload)
        response.raise_for_status()
        return json.loads(response.text)

    def _request_loop(
        self,
        base_url: str,
        dataset: Dataset,
        sampling_params: SamplingParams,
        num_workers,
    ):
        batch_responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch = {}
            for batch in dataset:
                future = executor.submit(
                    self._send_request, base_url, list(batch.keys()), sampling_params
                )
                future_to_batch[future] = batch

            for i, future in enumerate(
                concurrent.futures.as_completed(future_to_batch)
            ):
                batch = future_to_batch[future]
                try:
                    result_dict = future.result()
                    logger.info(f"Batch {i} - Prompts:")
                    for prompt in batch.keys():
                        logger.info(prompt)

                    logger.info(f"Batch {i} - Responses:")
                    logger.info(json.dumps(result_dict, indent=2))
                    batch_responses.append(result_dict)
                    print("-" * 80)
                except Exception as e:
                    logger.error(f"Batch {i} generated an exception: {e}")
                    print("-" * 80)

            return batch_responses

    @pytest.mark.parametrize(
        "dataset_request,batch_size,num_workers",
        [
            (ALL, 8, 4),
        ],
        ids=[
            "ALL-bs4-n2",
        ],
    )
    @pytest.mark.parametrize(
        "model_artifacts,server",
        [
            (
                ModelConfig.get(name="local_meta_llama3.1_8b_instruct"),
                {"prefix_sharing_algorithm": "none"},
            ),  # noqa: E501
        ],
        ids=[
            "meta_llama3.1_8b_instruct-no_prefix_sharing",
        ],
        indirect=True,
    )
    def test_accuracy(
        self,
        dataset_request: DatasetRequest,
        batch_size: int,
        num_workers: int,
        server,
    ):
        process, port, config = server
        assert process.poll() is None, "Server process terminated unexpectedly."

        base_url = f"http://localhost:{port}"
        self._check_health(base_url)

        dataset_instance = Dataset(dataset_request, batch_size=batch_size)
        sampling_params = SamplingParams()

        server_results = self._request_loop(
            base_url, dataset_instance, sampling_params, num_workers
        )

        # Kill the server before loading the comparison model to free up resources
        process.terminate()
        process.wait()

        comparison_model = load_comparison_model()
        accuracy_results = self._validate_response(
            comparison_model, dataset_instance, server_results
        )

        logger.info(
            f"Dataset: {dataset_request.dataset.name} | "
            f"Batch Size: {batch_size} | "
            f"Num Workers: {num_workers}\n"
            f"{accuracy_results}"
        )

        if accuracy_results.accuracy < 100.0:
            logger.error(f"Accuracy below 100%: {accuracy_results.accuracy:.2f}%")
            raise AccuracyValidationException(
                f"Accuracy below 100%: {str(accuracy_results)}%"
            )
