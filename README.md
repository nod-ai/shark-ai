# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/nod-ai/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                 |    Stmts |     Miss |   Cover |   Missing |
|----------------------------------------------------- | -------: | -------: | ------: | --------: |
| amdsharktuner/amdsharktuner/\_\_init\_\_.py          |        7 |        0 |    100% |           |
| amdsharktuner/amdsharktuner/candidate\_gen.py        |      178 |       74 |     58% |37, 51, 56, 62, 73, 78, 81-82, 85-88, 105-106, 116, 119, 132, 138, 153, 165, 178, 184, 191, 198, 206-207, 211, 217, 245-247, 267-272, 288-300, 317-352, 362-367, 373-389 |
| amdsharktuner/amdsharktuner/candidate\_ordering.py   |       95 |       11 |     88% |89-92, 109-110, 179-187 |
| amdsharktuner/amdsharktuner/common.py                |      246 |       10 |     96% |80, 85, 90, 92, 134, 289, 312-313, 417, 463 |
| amdsharktuner/amdsharktuner/constraint\_generator.py |      158 |        7 |     96% |301, 392, 394, 396, 509, 573, 582 |
| amdsharktuner/amdsharktuner/dispatch\_constraints.py |      274 |       13 |     95% |106-109, 149-158, 258, 359, 543, 613, 628 |
| amdsharktuner/amdsharktuner/dispatch\_parser.py      |      207 |        9 |     96% |36-38, 139, 144, 192-193, 291, 443 |
| amdsharktuner/amdsharktuner/libtuner.py              |      617 |      382 |     38% |109-111, 114, 117, 120, 125-128, 132, 136, 140, 151, 159, 169, 196, 200-209, 294-413, 417-468, 479-501, 508, 513-517, 521-541, 545-587, 591-679, 695-725, 729-733, 763-769, 778-912, 951-969, 987-1015, 1068, 1177-1261, 1271-1370 |
| amdsharktuner/amdsharktuner/merge\_td\_specs.py      |       32 |       32 |      0% |     19-79 |
| amdsharktuner/amdsharktuner/spec\_builder.py         |      127 |       23 |     82% |30-39, 76, 303-304, 314-342 |
| amdsharktuner/amdsharktuner/test\_utils.py           |       15 |        2 |     87% |     28-29 |
| amdsharktuner/boo\_tuner/\_\_init\_\_.py             |        0 |        0 |    100% |           |
| amdsharktuner/boo\_tuner/\_\_main\_\_.py             |        2 |        2 |      0% |       7-9 |
| amdsharktuner/boo\_tuner/boo\_tuner.py               |      208 |      139 |     33% |50-55, 59, 63, 67, 71, 75, 80, 89-131, 196-242, 255-370, 377-419, 423 |
| amdsharktuner/dispatch\_tuner/\_\_init\_\_.py        |        0 |        0 |    100% |           |
| amdsharktuner/dispatch\_tuner/\_\_main\_\_.py        |        2 |        2 |      0% |       7-9 |
| amdsharktuner/dispatch\_tuner/dispatch\_tuner.py     |       91 |       91 |      0% |     7-169 |
| amdsharktuner/model\_tuner/\_\_init\_\_.py           |        0 |        0 |    100% |           |
| amdsharktuner/model\_tuner/\_\_main\_\_.py           |        2 |        2 |      0% |       7-9 |
| amdsharktuner/model\_tuner/model\_tuner.py           |      127 |      127 |      0% |     7-244 |
| amdsharktuner/setup.py                               |       17 |       17 |      0% |      7-33 |
| amdsharktuner/tests/boo\_tuner\_test.py              |       72 |        0 |    100% |           |
| amdsharktuner/tests/candidate\_gen\_test.py          |      118 |        0 |    100% |           |
| amdsharktuner/tests/candidate\_ordering\_test.py     |       53 |        0 |    100% |           |
| amdsharktuner/tests/common\_test.py                  |      202 |        1 |     99% |       291 |
| amdsharktuner/tests/constraint\_generator\_test.py   |      173 |        0 |    100% |           |
| amdsharktuner/tests/dispatch\_constraints\_test.py   |      156 |        0 |    100% |           |
| amdsharktuner/tests/dispatch\_parser\_test.py        |      161 |        0 |    100% |           |
| amdsharktuner/tests/libtuner\_test.py                |      161 |        0 |    100% |           |
| amdsharktuner/tests/spec\_builder\_test.py           |      135 |        0 |    100% |           |
|                                            **TOTAL** | **3636** |  **944** | **74%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/nod-ai/shark-ai/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/nod-ai/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/nod-ai/shark-ai/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/nod-ai/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fnod-ai%2Fshark-ai%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/nod-ai/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.