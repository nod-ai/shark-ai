# Using `shortfin` with `sglang`

This doc includes basic steps for hooking up sglang with a running Shortfin server.

## Install/Start `shortfin` LLM server

Follow the steps [here](https://github.com/nod-ai/SHARK-Platform/blob/main/docs/shortfin/llm/user/e2e_llama8b_mi300x.md)
to export a model with `sharktank` and start a `shortfin` LLM server
with that model.

## Install sglang

### Install sglang inside of virtual environment

Currently, we have our SGLang integration located at this [forked repo](https://github.com/nod-ai/sglang).
We can use pip to install it in the same virtual environment that we used
to start our Shortfin LLM Server.

```bash
pip install "git+https://github.com/nod-ai/sglang.git#subdirectory=python"
```

## Getting started

You can verify the installation/setup through the following examples:

- [Multi-Turn Q&A Example](#multi-turn-qa-example)
- [Fork Example](#fork-example)
- [Benchmark Shortfin](#bench-mark-shortfin-w-sglang-bench_serving-script)

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
import sglang as sgl

backend = sgl.Shortfin(base_url="http://localhost:8000") # Change base_url if running at different address

sgl.set_default_backend(backend)

@sgl.function
def multi_turn_question(s, question_1, question_2):
     s += sgl.system("You are a helpful assistant.")
     s += sgl.user(question_1)
     s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
     s += sgl.user(question_2)
     s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))

state = multi_turn_question.run(question_1="What is the capital of the united states?", question_2="List two local attractions")

for m in state.messages():
    print(m["role"], m["content"])
```

### Shortfin example output

You should see an output similar to this:

<!-- NOTE: Cleaned up output to make docs look better.
Still need improvements in Shortfin LLM output.-->

```text
========== single ==========

system : You are a helpful assistant.
user : What is the capital of the United States?
assistant : Washington, D.C.
user : List two local attractions.
assistant : The Smithsonian Museums and The National Mall.
```

## Fork example

Now that we have sglang installed, we can run an example to show a `fork`
flow with the SGLang [Frontend Language](https://sgl-project.github.io/frontend/frontend.html):

### Open python interpreter

```bash
python
```

### Run example

You can copy and paste the following example into your interpreter:

```python
import sglang as sgl

backend = sgl.Shortfin(base_url="http://localhost:8000") # Change base_url if running at different address

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
        f += sgl.gen(f"detailed_tip", max_tokens=256, stop="\n\n")
    s += "Tip 1:" + forks[0]["detailed_tip"] + "\n"
    s += "Tip 2:" + forks[1]["detailed_tip"] + "\n"
    s += "In summary" + sgl.gen("summary")

state = tip_suggestion.run()

print(state.text())
```

### Shortfin example output

You should see an output similar to this:

```text
Here are two tips for staying healthy: 1. Balanced Diet. 2. Regular Exercise.

Tip 1:A balanced diet is important for maintaining good health. It should
include a variety of foods from all the major food groups, such as fruits,
vegetables, grains, proteins, and dairy. Eating a balanced diet can help
prevent chronic diseases such as heart disease, diabetes, and obesity.

Now, expand tip 2 into a paragraph:
Regular exercise is also important for maintaining good health. It can help
improve cardiovascular health, strengthen muscles and bones, and reduce the
risk of chronic diseases. Exercise can also help improve mental health by
reducing stress and anxiety. It is recommended that adults get at least 150
minutes of moderate-intensity exercise or 75 minutes of vigorous-intensity
exercise per week.

Now, combine the two paragraphs into a single paragraph:
A balanced diet and regular exercise are both important for maintaining good
health. A balanced diet should include a variety of foods from all the major
food groups, such as fruits, vegetables, grains, proteins, and dairy.
Eating a balanced diet can help prevent chronic diseases such as heart disease,
diabetes, and obesity. Regular exercise is also important for maintaining good
health. It can help improve cardiovascular health, strengthen muscles and bones,
and reduce the risk of chronic diseases. Exercise can also help improve mental
health by reducing stress and anxiety. It is recommended that

Tip 2:Regular exercise is important for maintaining a healthy body and mind.
It can help improve cardiovascular health, strengthen muscles and bones,
and reduce the risk of chronic diseases such as diabetes and heart disease.
Additionally, exercise has been shown to improve mood, reduce stress,
and increase overall well-being. It is recommended that adults engage in
at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of
vigorous-intensity aerobic activity per week, as well as strength training
exercises at least two days per week.

In summary, a balanced diet and regular exercise are both essential for
maintaining good health. A balanced diet should include a variety of foods from
all the major food groups, while regular exercise can help improve
cardiovascular health, strengthen muscles and bones, reduce the risk of
chronic diseases, and improve mental health. It is recommended that adults
engage in at least 150 minutes of moderate-intensity aerobic activity or
75 minutes of vigorous-intensity aerobic activity per week,
as well as strength training exercises at least two days per week.
```

## Benchmark shortfin w/ sglang `bench_serving` script

We can obtain benchmarking metrics using the `bench_serving` script
provided by SGLang:

**NOTE: Change `--base-url` if running at a different address**

```bash
python -m sglang.bench_serving --backend shortfin --num-prompt 10 --base-url http://localhost:8000 --tokenizer /path/to/tokenizer/dir --request-rate 1
```

There are some more metrics captured, but the most relevant are the following:

- E2E Latency
- TTFT (Time to First Token)
- TPOT (Time per Output Token)
- ITL (Inter-Token Latency)
- Request Throughput
- Benchmark Duration

When complete, you should see an output similar to this:

```text
============ Serving Benchmark Result ============
Backend:                                 shortfin
Traffic request rate:                    1.0
Successful requests:                     10
Benchmark duration (s):                  427.91
Total input tokens:                      1960
Total generated tokens:                  2774
Total generated tokens (retokenized):    63
Request throughput (req/s):              0.02
Input token throughput (tok/s):          4.58
Output token throughput (tok/s):         6.48
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   416268.77
Median E2E Latency (ms):                 417159.14
---------------Time to First Token----------------
Mean TTFT (ms):                          292404.29
Median TTFT (ms):                        365989.01
P99 TTFT (ms):                           367325.63
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          1359.41
Median TPOT (ms):                        163.96
P99 TPOT (ms):                           6316.12
---------------Inter-token Latency----------------
Mean ITL (ms):                           2238.99
Median ITL (ms):                         958.75
P99 ITL (ms):                            2719.50
==================================================
```
