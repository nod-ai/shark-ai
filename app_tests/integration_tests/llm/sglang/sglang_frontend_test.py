# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import re
from typing import Tuple
import pytest

from ..model_management import AccuracyValidationException

pytest.importorskip("sglang")
import sglang as sgl
from sglang.lang.chat_template import get_chat_template

pytest.importorskip("sentence_transformers")
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

from ..device_settings import DeviceSettings

DEVICE_SETTINGS = DeviceSettings(
    compile_flags=(
        "--iree-hal-target-device=hip",
        "--iree-hip-target=gfx942",
    ),
    server_flags=("--device=hip",),
)

ACCEPTED_THRESHOLD = 0.7


def register_shortfin_backend(port):
    backend = sgl.Shortfin(
        chat_template=get_chat_template("llama-3-instruct"),
        base_url=f"http://localhost:{port}",
    )
    sgl.set_default_backend(backend)


def compute_similarity(model: SentenceTransformer, sentence_1: str, sentence_2: str):
    embeddings = model.encode([sentence_1, sentence_2])
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()


@sgl.function
def multi_turn_question(
    s,
    question_1: str,
    question_2: str,
    top_ps: Tuple[float | None, float | None] = (None, None),
    top_ks: Tuple[int | None, int | None] = (None, None),
):
    """Run a multi-turn question and answer session.

    This function allows for a multi-turn interaction where the user can send
    a prompt, receive a response from that prompt, then send a follow-up generation,
    that retains the context of the first response.

    Args:
        s (sgl.State): The state object to which the messages will be added.
        question_1 (str): The first question to be asked.
        question_2 (str): The second question to be asked, which will follow the first answer.
        top_ps (Tuple[float  |  None, float  |  None], optional): The top_p values for the two questions. Defaults to (None, None).
        top_ks (Tuple[int  |  None, int  |  None], optional): The top_k values for the two questions. Defaults to (None, None).
    """
    logger.info(f"Running multi-turn question with top_p: {top_ps} and top_k: {top_ks}")

    kwargs = [{"max_tokens": 50, "temperature": 1.0}] * 2

    for i, (top_p, top_k) in enumerate(zip(top_ps, top_ks)):
        if top_p is not None:
            kwargs[i]["top_p"] = top_p
        if top_k is not None:
            kwargs[i]["top_k"] = top_k

    s += sgl.user(question_1)
    s += sgl.assistant(
        sgl.gen(
            "answer_1",
            **kwargs[0],
        )
    )
    s += sgl.user(question_2)
    s += sgl.assistant(
        sgl.gen(
            "answer_2",
            **kwargs[1],
        )
    )


@sgl.function
def tip_suggestion(s):
    """Generate a forked process, but using a tip suggestion example.

    Forked processes send two requests in parallel, and concatenate the results
    as if it were a single response.

    Args:
        s (sgl.State): The state object to which the messages will be added.
    """
    s += (
        "Here are two tips for staying healthy: "
        "1. Balanced Diet. 2. Regular Exercise.\n\n"
    )

    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += f"Now, expand tip {i+1} into a paragraph:\n"
        f += sgl.gen(f"detailed_tip", max_tokens=50, temperature=1.0)

    s += "Tip 1:" + forks[0]["detailed_tip"] + "\n"
    s += "Tip 2:" + forks[1]["detailed_tip"] + "\n"
    s += "In summary" + sgl.gen("summary")


@pytest.mark.parametrize(
    "model_artifacts,start_server,load_comparison_model",
    [
        (
            {"device_settings": DEVICE_SETTINGS},
            {"device_settings": DEVICE_SETTINGS},
            None,
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "top_ps,top_ks",
    [
        ((None, None), (None, None)),  # Default values
        ((0.95, 0.95), (None, None)),  # top_p values only
        ((None, None), (8, 8)),  # top_k values only
        ((0.95, 0.95), (8, 8)),  # Both top_p and top_k values
    ],
    ids=["default", "top_p_only", "top_k_only", "top_p_and_top_k"],
)
def test_multi_turn_qa(
    model_artifacts, start_server, load_comparison_model, top_ps, top_ks
):
    server, port = start_server
    register_shortfin_backend(port)

    model = load_comparison_model

    question_1 = "Name the capital city of the USA."
    question_2 = "The Smithsonian is in this location."

    answer_1 = "The capital city of the United States of America is Washington, D.C. (short for District of Columbia).assistant\n\nWould you like to know more about Washington, D.C. or is there something else I can help you with?"
    answer_2 = "The Smithsonian Institution is indeed located in Washington, D.C. and is one of the world's largest and most comprehensive museums and research complexes. It was founded in 1846 and is named after British scientist James Smithson, who left a bequest to"

    logger.info("Testing multi-turn Q&A run...")
    state = multi_turn_question.run(
        question_1=question_1,
        question_2=question_2,
        top_ps=top_ps,
        top_ks=top_ks,
    )
    messages = state.messages()
    logger.info("Received messages from multi-turn call.")

    assert messages[0] == {
        "role": "user",
        "content": question_1,
    }
    assert messages[1]["role"] == "assistant"

    logger.info("Computing similarity between first question and first answer...")
    first_q_answer = messages[1]["content"]
    score_1 = compute_similarity(model, answer_1, first_q_answer)
    if not score_1 > ACCEPTED_THRESHOLD:
        raise AccuracyValidationException(
            f"Accuracy error between {answer_1} and {first_q_answer}:\n SCORE: {score_1}"
        )
    logger.info("Similarity passed")

    assert messages[2] == {
        "role": "user",
        "content": question_2,
    }
    assert messages[3]["role"] == "assistant"

    logger.info("Testing similarity between second question and second answer...")
    second_q_answer = messages[3]["content"]
    score_2 = compute_similarity(model, answer_2, second_q_answer)
    if not score_2 > ACCEPTED_THRESHOLD:
        raise AccuracyValidationException(
            f"Accuracy error between {answer_2} and {second_q_answer}:\n SCORE: {score_2}"
        )

    logger.info(
        "Similarity passed.\n"
        f"Expected answer 1: {answer_1}\n"
        f"Received answer 1: {first_q_answer}\n"
        f"Similarity score 1: {score_1}\n"
        f"Expected answer 2: {answer_2}\n"
        f"Received answer 2: {second_q_answer}\n"
        f"Similarity score 2: {score_2}"
    )


@pytest.mark.skip(reason="https://github.com/nod-ai/amdshark-ai/issues/2244")
@pytest.mark.parametrize(
    "model_artifacts,start_server,load_comparison_model",
    [
        (
            {"device_settings": DEVICE_SETTINGS},
            {"device_settings": DEVICE_SETTINGS},
            None,
        )
    ],
    indirect=True,
)
def test_stream_multi_turn_qa(model_artifacts, start_server, load_comparison_model):
    server, port = start_server
    register_shortfin_backend(port)

    def clean_message(message: str):
        """Remove chat tags from message before comparison.

        Args:
            message (str): Message to clean.

        Returns:
            str: Message without tags (i.e. <|start_header_id|>)
        """
        pattern = r"<\|.*?\|>"
        return re.sub(pattern, "", message)

    model = load_comparison_model
    question_1 = "Name the capital city of the USA."
    question_2 = "The Smithsonian is in this location."
    expected_answer_1 = "The capital city of the United States of America is Washington, D.C. (short for District of Columbia).assistant\n\nWould you like to know more about Washington, D.C. or is there something else I can help you with?"
    expected_answer_2 = "The Smithsonian Institution is indeed located in Washington, D.C. and is one of the world's largest and most comprehensive museums and research complexes. It was founded in 1846 and is named after British scientist James Smithson, who left a bequest to"

    logger.info("Testing multi-turn Q&A run w/ stream...")
    state = multi_turn_question.run(
        question_1=question_1,
        question_2=question_2,
        stream=True,
    )
    messages = ""
    for chunk in state.text_iter():
        messages += chunk
    logger.info("Received messages from multi-turn call.")

    logger.info("Computing similarity between expectation and result")
    expected_result = f"user: {question_1}\nassistant: {expected_answer_1}\nuser: {question_2}\nassistant: {expected_answer_2}"
    cleaned_messages = clean_message(messages)
    score = compute_similarity(model, cleaned_messages, expected_result)
    if not score > ACCEPTED_THRESHOLD:
        raise AccuracyValidationException(
            f"Accuracy error between {expected_result} and {messages}:\n SCORE: {score}"
        )

    logger.info(
        "Similarity passed.\n"
        f"Expected result: {expected_result}\n"
        f"Received messages: {cleaned_messages}\n"
        f"Similarity score: {score}"
    )


@pytest.mark.parametrize(
    "model_artifacts,start_server,load_comparison_model",
    [
        (
            {"device_settings": DEVICE_SETTINGS},
            {"device_settings": DEVICE_SETTINGS},
            None,
        )
    ],
    indirect=True,
)
def test_batch_multi_turn_qa(model_artifacts, start_server, load_comparison_model):
    server, port = start_server
    register_shortfin_backend(port)
    model = load_comparison_model

    question_1_1 = "Name the capital city of the USA."
    question_1_2 = "The Smithsonian is in this location."
    expected_answer_1_1 = "The capital city of the United States of America is Washington, D.C. (short for District of Columbia).assistant\n\nWould you like to know more about Washington, D.C. or is there something else I can help you with?"
    expected_answer_1_2 = "The Smithsonian Institution is indeed located in Washington, D.C. and is one of the world's largest and most comprehensive museums and research complexes. It was founded in 1846 and is named after British scientist James Smithson, who left a bequest to"

    question_2_1 = "Name the largest city in the USA."
    question_2_2 = "The Empire State Building is in this location."
    expected_answer_2_1 = "The largest city in the USA is New York City, with a population of over 8.4 million people, according to the United States Census Bureau (2020 estimates).assistant\n\nHowever, I should note that the largest city in the"
    expected_answer_2_2 = "That's correct, the iconic Empire State Building is located in Midtown Manhattan, New York City. It's one of the most recognizable landmarks in the world and a symbol of the city's grandeur and history.assistant\n\nAnd, by"

    logger.info("Testing batch multi-turn Q&A run...")
    states = multi_turn_question.run_batch(
        [
            {
                "question_1": question_1_1,
                "question_2": question_1_2,
            },
            {
                "question_1": question_2_1,
                "question_2": question_2_2,
            },
        ]
    )

    first_qa = states[0]
    second_qa = states[1]

    first_qa_messages = first_qa.messages()
    second_qa_messages = second_qa.messages()

    logger.info("Testing first batch of messages...")
    assert first_qa_messages[0] == {
        "role": "user",
        "content": question_1_1,
    }

    assert first_qa_messages[1]["role"] == "assistant"
    first_answer = first_qa_messages[1]["content"]
    expected_answer = expected_answer_1_1
    score = compute_similarity(model, expected_answer, first_answer)
    if not score > ACCEPTED_THRESHOLD:
        raise AccuracyValidationException(
            f"Accuracy error between {expected_answer} and {first_answer}:\n SCORE: {score}"
        )

    assert first_qa_messages[2] == {
        "role": "user",
        "content": question_1_2,
    }
    first_qa_messages[3]["role"] = "assistant"
    second_answer = first_qa_messages[3]["content"]
    expected_answer = expected_answer_1_2
    score = compute_similarity(model, expected_answer, second_answer)
    if not score > ACCEPTED_THRESHOLD:
        raise AccuracyValidationException(
            f"Accuracy error between {expected_answer} and {second_answer}:\n SCORE: {score}"
        )
    logger.info("First batch passed.")

    logger.info("Testing second batch of messages...")
    assert second_qa_messages[0] == {
        "role": "user",
        "content": question_2_1,
    }

    assert second_qa_messages[1]["role"] == "assistant"
    first_answer = second_qa_messages[1]["content"]
    expected_answer = expected_answer_2_1
    score = compute_similarity(model, expected_answer, first_answer)
    if not score > ACCEPTED_THRESHOLD:
        raise AccuracyValidationException(
            f"Accuracy error between {expected_answer} and {first_answer}:\n SCORE: {score}"
        )

    assert second_qa_messages[2] == {
        "role": "user",
        "content": question_2_2,
    }
    second_qa_messages[3]["role"] = "assistant"
    second_answer = second_qa_messages[3]["content"]
    expected_answer = expected_answer_2_2
    score = compute_similarity(model, expected_answer, second_answer)
    if not score > ACCEPTED_THRESHOLD:
        raise AccuracyValidationException(
            f"Accuracy error between {expected_answer} and {second_answer}:\n SCORE: {score}"
        )

    logger.info("Second batch passed.")
    logger.info(
        "Similarity passed.\n"
        f"Expected answer 1.1: {expected_answer_1_1}\n"
        f"Received answer 1.1: {first_answer}\n"
        f"Similarity score 1.1: {score}\n"
        f"Expected answer 1.2: {expected_answer_1_2}\n"
        f"Received answer 1.2: {second_answer}\n"
        f"Similarity score 1.2: {score}\n"
        f"Expected answer 2.1: {expected_answer_2_1}\n"
        f"Received answer 2.1: {first_answer}\n"
        f"Similarity score 2.1: {score}\n"
        f"Expected answer 2.2: {expected_answer_2_2}\n"
        f"Received answer 2.2: {second_answer}\n"
        f"Similarity score 2.2: {score}"
    )


@pytest.mark.parametrize(
    "model_artifacts,start_server,load_comparison_model",
    [
        (
            {"device_settings": DEVICE_SETTINGS},
            {"device_settings": DEVICE_SETTINGS},
            None,
        )
    ],
    indirect=True,
)
def test_fork(model_artifacts, start_server, load_comparison_model):
    server, port = start_server
    register_shortfin_backend(port)
    model = load_comparison_model

    logger.info("Testing fork...")
    state = tip_suggestion.run()
    result = state.text()
    logger.info("Fork response received.")

    logger.info("Computing similarity...")
    expected_answer = """Here are two tips for staying healthy: 1. Balanced Diet. 2. Regular Exercise.
    Tip 1:A balanced diet is essential for maintaining good health. It involves consuming a variety of foods from different food groups, including fruits, vegetables, whole grains, lean proteins, and healthy fats. A balanced diet provides the body with the necessary nutrients, vitamins, and
    Tip 2:Regular exercise is essential for maintaining a healthy body. It helps to improve cardiovascular health, increase strength and flexibility, and boost the immune system. Regular physical activity can also reduce the risk of chronic diseases such as heart disease, diabetes, and certain types of cancer
    In summary, a balanced diet and regular exercise are two of the most important tips for staying healthy. By following these tips, you can maintain a healthy body and reduce the risk of chronic diseases.
    """
    score = compute_similarity(model, result, expected_answer)
    if not score > ACCEPTED_THRESHOLD:
        raise AccuracyValidationException(
            f"Accuracy error between {expected_answer} and {result}:\n SCORE: {score}"
        )
    logger.info(
        "Similarity passed.\n"
        f"Expected answer: {expected_answer}\n"
        f"Received result: {result}\n"
        f"Similarity score: {score}"
    )
