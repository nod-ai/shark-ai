# Using `shortfin` with `sglang`

This doc includes basic steps for hooking up sglang with a running Shortfin server.

## Current Support Status

| Feature     | Description                                                     | Enabled | Reference                                                                                                    |
| ----------- | --------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------ |
| `gen`       | Generate shortfin completion, given a prompt                    | ✅       | [Shortfin Implementation](https://github.com/nod-ai/sglang/blob/main/python/sglang/lang/backend/shortfin.py) |
| `streaming` | Stream shortfin completion, given a prompt                      | ✅       | [Streaming](https://sgl-project.github.io/frontend/frontend.html#streaming)                                  |
| `run_batch` | Run batch of disjoint requests with continous batching          | ✅       | [Batching](https://sgl-project.github.io/frontend/frontend.html#batching)                                    |
| `fork`      | Generate sections of the same prompt in parallel                | ✅       | [Fork Docs](https://sgl-project.github.io/frontend/frontend.html#parallelism)                                |
| `choices`   | Given set of choices, generate response based on best log probs | ❌       | [Choices Methods](https://sgl-project.github.io/frontend/choices_methods.html#choices-methods-in-sglang)     |
| `image`     | Pass image as part of multi-modal prompt                        | ❌       | [sgl.image](https://sgl-project.github.io/frontend/frontend.html#multi-modality)                             |
| `regex`     | Specify regular expression as decoding constraint               | ❌       | [Regex](https://sgl-project.github.io/frontend/frontend.html#constrained-decoding)                           |

## Prerequisites

For this tutorial, you will need to meet the following prerequisites:

### Software

- Python >= 3.11
    - You can check out [pyenv](https://github.com/pyenv/pyenv)
    as a good tool to be able to manage multiple versions of python
    on the same system.

### Shortfin LLM Server

- A running `shortfin` LLM server. Directions on launching the llm server on one system can be found in [Llama end to end serving instructions](./llama_serving.md) and for launching
on a kubernetes cluster, see [Llama 8b GPU instructions on Kubernetes](./llama_serving_on_kubernetes.md)
  - We will use the shortfin server as the `backend` to generate completions
    from SGLang's `frontend language`. In this tutorial, you can think of
    `sglang` as the client and `shortfin` as the server.

After the `shortfin` LLM Server has started, we must obtain the base_url.
We will store this in our environment in order to send request to `shortfin`
 through the `sglang` client examples below.

```bash
export SHORTFIN_BASE_URL="SHORTFIN_BASE_URL" # example: http://localhost:8000
```

## Install sglang

### Install sglang inside of virtual environment

Currently, we have our SGLang integration located at this [forked repo](https://github.com/nod-ai/sglang).
We can use pip to install it in the same virtual environment that we used
to start our Shortfin LLM Server.

```bash
python -m venv --prompt amdshark-ai .venv
source .venv/bin/activate
pip install "git+https://github.com/nod-ai/sglang.git#subdirectory=python"
```

## Getting started

You can verify the installation/setup through the following examples:

- [Multi-Turn Q&A Example](#multi-turn-qa-example)
- [Streaming Example](#streaming-example)
- [Fork Example](#fork-example)
- [Multi-Turn Q&A Batching Example](#multi-turn-qa-batch-example)

In these examples, we will set our `max_tokens` to 50 when generating completions.
This details how many tokens we want to generate for each completion.

We can modify the arguments passed to `sgl.gen` to alter the outputs of our
`shortfin` LLM server. Specifically:

- `max_tokens` - The maximum number of tokens to generate for completion.
                 We may obtain longer responses by increasing this value,
                 and shorter responses by decreasing it.
- `temperature` - We can include a temperature parameter to control the
                  randomness of the generated completions. A higher value
                  will result in more randomness, while a lower value will
                  result in more deterministic completions.

## Multi-Turn Q&A example

Now that we have sglang installed, we can run an example to show a multi-turn
Q&A flow with the SGLang [Frontend Language](https://sgl-project.github.io/frontend/frontend.html):

### Open python interpreter

```bash
python
```

### Run example

You can copy and paste the following example into your interpreter:

```python
import os

import sglang as sgl

from sglang.lang.chat_template import get_chat_template

SHORTFIN_BASE_URL = os.environ["SHORTFIN_BASE_URL"]

backend = sgl.Shortfin(chat_template=get_chat_template("llama-3-instruct"), base_url=SHORTFIN_BASE_URL)

sgl.set_default_backend(backend)

@sgl.function
def multi_turn_question(s, question_1, question_2):
     s += sgl.user(question_1)
     s += sgl.assistant(sgl.gen("answer_1", max_tokens=50))
     s += sgl.user(question_2)
     s += sgl.assistant(sgl.gen("answer_2", max_tokens=50))

state = multi_turn_question.run(question_1="Name the capital city of the USA.", question_2="The Smithsonian is in this location.")

for m in state.messages():
    print(m["role"], m["content"])
```

## Streaming Example

We can stream our request for a more responsive feel. Let's invoke a `streaming` Q&A from our server:

```python
import os

import sglang as sgl
from sglang.lang.chat_template import get_chat_template

SHORTFIN_BASE_URL = os.environ["SHORTFIN_BASE_URL"]

backend = sgl.Shortfin(chat_template=get_chat_template("llama-3-instruct"), base_url=SHORTFIN_BASE_URL)

sgl.set_default_backend(backend)

@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=50))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=50))

question_1 = "Name the capital city of the USA."
question_2 = "The Smithsonian is in this location."

# Run the multi-turn question function with streaming enabled
state = multi_turn_question.run(
    question_1=question_1,
    question_2=question_2,
    stream=True,
)

# Collect messages from the streamed output
messages = ""

for chunk in state.text_iter():
    messages += chunk

print(messages)
```


## Fork example

We can also send different pieces of the same prompt in parallel using the `fork`
flow with the SGLang [Frontend Language](https://sgl-project.github.io/frontend/frontend.html):

```python
import os

import sglang as sgl

from sglang.lang.chat_template import get_chat_template

SHORTFIN_BASE_URL = os.environ["SHORTFIN_BASE_URL"]

backend = sgl.Shortfin(chat_template=get_chat_template("llama-3-instruct"), base_url=SHORTFIN_BASE_URL)

sgl.set_default_backend(backend)

@sgl.function
def tip_suggestion(s):
    s += (
        "Here are two tips for staying healthy: "
        "1. Balanced Diet. 2. Regular Exercise.\n\n"
    )
    forks = s.fork(2)
    for i, f in enumerate(forks):
        f += f"Now, expand tip {i+1} into a paragraph:\n"
        f += sgl.gen(f"detailed_tip", max_tokens=50, stop="\n\n")
    s += "Tip 1:" + forks[0]["detailed_tip"] + "\n"
    s += "Tip 2:" + forks[1]["detailed_tip"] + "\n"
    s += "In summary" + sgl.gen("summary")

state = tip_suggestion.run()

print(state.text())
```

## Multi-Turn Q&A Batch Example

With **Shortfin** + SGLang, we can also easily send requests as a batch.
Let's now invoke a `batched` Q&A flow with the SGLang [Batching](https://sgl-project.github.io/frontend/frontend.html#batching):

```python
import os

import sglang as sgl
from sglang.lang.chat_template import get_chat_template

SHORTFIN_BASE_URL = os.environ["SHORTFIN_BASE_URL"]

backend = sgl.Shortfin(chat_template=get_chat_template("llama-3-instruct"), base_url=SHORTFIN_BASE_URL)

# Set the default backend for sglang
sgl.set_default_backend(backend)

@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=50))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=50))

# Define the questions for the first and second sets
question_1_1 = "Name the capital city of the USA."
question_1_2 = "The Smithsonian is in this location."
question_2_1 = "Name the largest city in the USA."
question_2_2 = "The Empire State Building is in this location."

# Run the multi-turn question function in batch mode
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

# Extract responses from the states
first_qa = states[0]
second_qa = states[1]

first_qa_messages = first_qa.messages()
second_qa_messages = second_qa.messages()

# Print messages from the first QA session
for m in first_qa_messages:
    print(m["role"], m["content"])

# Print messages from the second QA session
for m in second_qa_messages:
    print(m["role"], m["content"])

```
