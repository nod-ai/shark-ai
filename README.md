# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/nod-ai/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                          |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| sharktank/conftest.py                                                         |      149 |       14 |     91% |291, 298, 305, 339, 363, 368-371, 381, 386, 400, 424, 427-428 |
| sharktank/integration/models/punet/integration\_test.py                       |       94 |       57 |     39% |15-16, 21-31, 52-62, 70-80, 90-101, 110-121, 131-143, 150, 155-167, 174, 185-192, 210-220, 242-255 |
| sharktank/setup.py                                                            |       18 |       18 |      0% |      7-34 |
| sharktank/sharktank/\_\_init\_\_.py                                           |        4 |        1 |     75% |        15 |
| sharktank/sharktank/build/\_\_init\_\_.py                                     |        1 |        1 |      0% |         7 |
| sharktank/sharktank/build/actions.py                                          |       45 |       45 |      0% |     7-109 |
| sharktank/sharktank/evaluate/perplexity\_iree.py                              |      249 |      210 |     16% |71-92, 96-102, 107-129, 137-165, 173-210, 220-247, 252-297, 302-351, 354-409, 416-464, 477-532, 539-588, 592 |
| sharktank/sharktank/evaluate/perplexity\_torch.py                             |      181 |      144 |     20% |54-57, 61-67, 72-94, 113-138, 142-160, 164-228, 242-281, 295-351, 376-401, 405-445, 449 |
| sharktank/sharktank/examples/export\_paged\_llm\_v1.py                        |      114 |       36 |     68% |39, 51, 90-107, 192-193, 202-275, 279 |
| sharktank/sharktank/examples/paged\_llm\_v1.py                                |       63 |       54 |     14% |32-116, 122 |
| sharktank/sharktank/examples/pipeline/export\_ppffn\_net.py                   |       64 |        4 |     94% |139, 145, 172, 179 |
| sharktank/sharktank/examples/sharding/export\_ffn\_net.py                     |       59 |       13 |     78% |51-63, 82, 88, 113, 120 |
| sharktank/sharktank/examples/sharding/shard\_llm\_dataset.py                  |       23 |        3 |     87% |33, 36, 51 |
| sharktank/sharktank/kernels/\_\_init\_\_.py                                   |       13 |        0 |    100% |           |
| sharktank/sharktank/kernels/attention.py                                      |       22 |        0 |    100% |           |
| sharktank/sharktank/kernels/base.py                                           |       52 |        5 |     90% |136, 155-160 |
| sharktank/sharktank/kernels/batch\_matmul\_transpose\_b.py                    |       49 |        0 |    100% |           |
| sharktank/sharktank/kernels/bitcast.py                                        |       63 |       40 |     37% |58-69, 75-88, 97-108, 114-127, 136-139 |
| sharktank/sharktank/kernels/conv\_2d\_nchw\_fchw.py                           |       64 |        0 |    100% |           |
| sharktank/sharktank/kernels/einsum\_2args\_q4.py                              |      122 |        2 |     98% |   69, 179 |
| sharktank/sharktank/kernels/gemm\_fp4\_asm.py                                 |       16 |        2 |     88% |    49-187 |
| sharktank/sharktank/kernels/mlir\_kernel.py                                   |      204 |       18 |     91% |40, 43, 47, 112, 123, 129, 131, 220, 262, 269, 277, 321, 329, 369-374, 382 |
| sharktank/sharktank/kernels/mmt\_block\_scaled\_offset\_q4.py                 |       50 |        3 |     94% |     94-96 |
| sharktank/sharktank/kernels/mmt\_block\_scaled\_q8.py                         |       38 |        0 |    100% |           |
| sharktank/sharktank/kernels/mmt\_super\_block\_scaled\_offset\_q4.py          |       59 |        0 |    100% |           |
| sharktank/sharktank/kernels/mmtfp.py                                          |       41 |        2 |     95% |     68-69 |
| sharktank/sharktank/kernels/pooling\_nchw\_sum.py                             |       38 |        0 |    100% |           |
| sharktank/sharktank/kernels/rotary.py                                         |       31 |        0 |    100% |           |
| sharktank/sharktank/kernels/topk.py                                           |       30 |        0 |    100% |           |
| sharktank/sharktank/kernels/wave/attention.py                                 |       48 |        0 |    100% |           |
| sharktank/sharktank/kernels/wave/mxfp4\_gemm.py                               |      100 |       74 |     26% |42-113, 123-162, 186-235 |
| sharktank/sharktank/kernels/wave/utils.py                                     |       18 |        5 |     72% |     50-56 |
| sharktank/sharktank/layers/\_\_init\_\_.py                                    |       15 |        0 |    100% |           |
| sharktank/sharktank/layers/activations.py                                     |        3 |        0 |    100% |           |
| sharktank/sharktank/layers/base.py                                            |      177 |       27 |     85% |131, 206-209, 224, 242, 259-260, 269, 298, 366-374, 385-398, 400, 404-407, 411, 417, 424 |
| sharktank/sharktank/layers/causal\_llm.py                                     |       22 |        0 |    100% |           |
| sharktank/sharktank/layers/configs/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| sharktank/sharktank/layers/configs/config.py                                  |      170 |       15 |     91% |139, 196, 205-211, 219, 234, 248-254, 267, 269, 289, 313 |
| sharktank/sharktank/layers/configs/llm\_configs.py                            |      403 |       99 |     75% |186, 188, 190, 192, 194, 196, 198, 204, 206, 208, 210, 216, 220, 224, 231, 233, 237, 239, 250, 253, 256-259, 262-282, 285-292, 302-305, 311-314, 318-323, 335-338, 349-350, 361-362, 452-453, 458, 547, 553-561, 570-582, 621, 643, 667-671, 707-710, 714-718 |
| sharktank/sharktank/layers/conv.py                                            |      100 |       61 |     39% |48, 58, 61, 63, 80, 95-110, 113-143, 157-172, 175-205 |
| sharktank/sharktank/layers/ffn\_block.py                                      |       26 |        0 |    100% |           |
| sharktank/sharktank/layers/ffn\_moe\_block.py                                 |       83 |       25 |     70% |65-73, 203-237, 243-246, 253-259 |
| sharktank/sharktank/layers/latent\_attention\_block.py                        |       55 |        7 |     87% |42, 61, 66, 76, 84-85, 102 |
| sharktank/sharktank/layers/linear.py                                          |       43 |        4 |     91% |58, 69, 77, 85 |
| sharktank/sharktank/layers/mixture\_of\_experts\_block.py                     |       71 |        5 |     93% |48, 52, 60, 105-109 |
| sharktank/sharktank/layers/mmdit.py                                           |      103 |        0 |    100% |           |
| sharktank/sharktank/layers/modulation.py                                      |       21 |        0 |    100% |           |
| sharktank/sharktank/layers/norm.py                                            |       37 |        0 |    100% |           |
| sharktank/sharktank/layers/paged\_attention.py                                |      195 |       14 |     93% |141, 149-155, 218, 244, 420, 423, 427, 514-519, 528, 531 |
| sharktank/sharktank/layers/paged\_llama\_attention\_block.py                  |      116 |        9 |     92% |133, 143, 180, 182, 184, 246, 262-266 |
| sharktank/sharktank/layers/rotary\_embedding.py                               |       31 |        0 |    100% |           |
| sharktank/sharktank/layers/rotary\_embedding\_hf.py                           |      107 |        2 |     98% |   239-240 |
| sharktank/sharktank/layers/testing.py                                         |       44 |        1 |     98% |       302 |
| sharktank/sharktank/layers/token\_embedding.py                                |       12 |        0 |    100% |           |
| sharktank/sharktank/models/\_\_init\_\_.py                                    |        7 |        0 |    100% |           |
| sharktank/sharktank/models/clip/\_\_init\_\_.py                               |        2 |        0 |    100% |           |
| sharktank/sharktank/models/clip/clip.py                                       |      206 |       31 |     85% |80, 123, 131, 143, 159-162, 171, 249, 326, 337, 340, 343, 397, 412, 439, 454, 487, 490, 493, 544-557, 568-570 |
| sharktank/sharktank/models/clip/export.py                                     |       27 |       10 |     63% |40-43, 51-59 |
| sharktank/sharktank/models/clip/export\_toy\_text\_model\_iree\_test\_data.py |       11 |        1 |     91% |        29 |
| sharktank/sharktank/models/clip/testing.py                                    |       67 |        4 |     94% |   175-179 |
| sharktank/sharktank/models/deepseek/testing.py                                |       22 |        0 |    100% |           |
| sharktank/sharktank/models/deepseek/toy\_deepseek.py                          |       33 |        9 |     73% | 82-92, 96 |
| sharktank/sharktank/models/dummy/\_\_init\_\_.py                              |        1 |        0 |    100% |           |
| sharktank/sharktank/models/dummy/dummy.py                                     |       39 |        0 |    100% |           |
| sharktank/sharktank/models/flux/\_\_init\_\_.py                               |        1 |        0 |    100% |           |
| sharktank/sharktank/models/flux/compile.py                                    |        1 |        0 |    100% |           |
| sharktank/sharktank/models/flux/export.py                                     |       55 |       24 |     56% |35-36, 56, 80, 95-98, 104-127 |
| sharktank/sharktank/models/flux/export\_flux\_transformer\_mlir.py            |       13 |       13 |      0% |      7-38 |
| sharktank/sharktank/models/flux/flux.py                                       |      233 |       29 |     88% |82-91, 117-121, 129, 135, 137, 142, 147, 152, 218, 222, 235, 242, 268-279, 288, 407 |
| sharktank/sharktank/models/flux/testing.py                                    |       54 |       10 |     81% |31, 154, 209-227 |
| sharktank/sharktank/models/grok/testing.py                                    |       22 |        0 |    100% |           |
| sharktank/sharktank/models/grok/toy\_grok.py                                  |       31 |        6 |     81% | 66-71, 75 |
| sharktank/sharktank/models/llama4/testing.py                                  |       41 |        1 |     98% |        17 |
| sharktank/sharktank/models/llama/testing.py                                   |       58 |        0 |    100% |           |
| sharktank/sharktank/models/llama/toy\_llama.py                                |       57 |       12 |     79% |83, 85, 87, 91, 95, 100, 159-165, 169 |
| sharktank/sharktank/models/llm/\_\_init\_\_.py                                |        1 |        0 |    100% |           |
| sharktank/sharktank/models/llm/config.py                                      |       41 |        4 |     90% |     37-40 |
| sharktank/sharktank/models/llm/export.py                                      |       69 |       19 |     72% |23-28, 34, 79-81, 89-93, 98-101, 135, 143-146 |
| sharktank/sharktank/models/llm/llm.py                                         |       99 |        4 |     96% |183, 212, 249, 252 |
| sharktank/sharktank/models/llm/testing.py                                     |       27 |       27 |      0% |      1-97 |
| sharktank/sharktank/models/punet/config.py                                    |       84 |       34 |     60% |70-82, 87-91, 98-122, 126-130 |
| sharktank/sharktank/models/punet/layers.py                                    |      324 |      191 |     41% |135-180, 195-226, 258, 280-285, 303-330, 341-355, 366-388, 393-397, 400-410, 418-444, 452-499, 513-519, 524-529, 616-624, 627-631, 654-659, 668-695, 720-725, 728, 738-739, 742-744 |
| sharktank/sharktank/models/punet/sharding.py                                  |       31 |        0 |    100% |           |
| sharktank/sharktank/models/punet/testing.py                                   |       65 |        0 |    100% |           |
| sharktank/sharktank/models/punet/tools/sample\_data.py                        |       26 |       21 |     19% |15-20, 33-46, 50-53 |
| sharktank/sharktank/models/t5/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| sharktank/sharktank/models/t5/export.py                                       |       58 |       31 |     47% |42-46, 56-72, 97-105, 117-143 |
| sharktank/sharktank/models/t5/t5.py                                           |      344 |      103 |     70% |126, 160, 189, 236-240, 266-269, 272-284, 313, 326, 334-336, 347, 360, 436-448, 464-480, 517, 557-571, 591-597, 605-642, 649-655, 662, 710, 713, 719-753, 780, 787, 793, 801, 840-842, 850-861, 894-895, 901-905, 911, 926-927, 949-959, 985, 1013, 1018, 1023-1025, 1031, 1034 |
| sharktank/sharktank/models/t5/testing.py                                      |       22 |        0 |    100% |           |
| sharktank/sharktank/models/vae/config.py                                      |       39 |       13 |     67% |44-48, 54-62 |
| sharktank/sharktank/models/vae/layers.py                                      |       97 |        6 |     94% |48, 101, 103, 205, 231, 235 |
| sharktank/sharktank/models/vae/model.py                                       |       67 |        7 |     90% |24-25, 33, 63, 94, 108, 116 |
| sharktank/sharktank/models/vae/testing.py                                     |       14 |        0 |    100% |           |
| sharktank/sharktank/models/vae/tools/diffuser\_ref.py                         |       50 |       13 |     74% |39-60, 87, 104 |
| sharktank/sharktank/models/vae/tools/run\_vae.py                              |       75 |       47 |     37% |64-158, 162 |
| sharktank/sharktank/models/vae/tools/sample\_data.py                          |       14 |        5 |     64% |27-29, 39-40 |
| sharktank/sharktank/ops/\_\_init\_\_.py                                       |       13 |        0 |    100% |           |
| sharktank/sharktank/ops/\_registry.py                                         |      213 |       14 |     93% |132, 137, 267-270, 282, 319, 322-325, 338, 452, 474, 482, 529 |
| sharktank/sharktank/ops/attention\_impls.py                                   |      106 |        2 |     98% |  131, 144 |
| sharktank/sharktank/ops/cpu\_impls.py                                         |       20 |        1 |     95% |        43 |
| sharktank/sharktank/ops/custom\_impls.py                                      |      110 |       44 |     60% |63-67, 85, 101, 122-141, 159-187, 196, 200-203, 226, 228, 230 |
| sharktank/sharktank/ops/default\_impls.py                                     |      547 |      108 |     80% |122, 124, 156, 158, 160, 193, 195, 197, 252-260, 265-272, 284-292, 324, 326, 340-341, 356-363, 377-384, 403-421, 435, 445, 614, 633, 644-646, 685, 785, 856, 861, 866, 872, 904-911, 917, 1006, 1010, 1056, 1061-1078, 1083, 1088 |
| sharktank/sharktank/ops/qconv\_impls.py                                       |      123 |       31 |     75% |47, 53, 67-71, 88, 94, 109, 137-142, 168-177, 229, 252, 270-285, 298, 303, 310 |
| sharktank/sharktank/ops/qlinear\_impls.py                                     |       91 |       16 |     82% |41, 66, 85, 89, 103-106, 117-118, 144-145, 164, 167, 190-192, 211 |
| sharktank/sharktank/ops/quantized\_impls.py                                   |      222 |       13 |     94% |81, 89, 91-97, 99-106, 117-118, 142, 255-257, 394 |
| sharktank/sharktank/ops/shape.py                                              |       28 |        1 |     96% |        84 |
| sharktank/sharktank/ops/sharded\_impls.py                                     |      885 |       85 |     90% |228, 450, 471, 512-514, 521, 529, 544, 554-558, 568-573, 580-581, 588-589, 659-668, 718-726, 911, 963, 977, 979, 982, 987, 990, 1056-1058, 1112-1116, 1129, 1146, 1155, 1163-1165, 1191, 1207, 1217, 1241, 1267, 1269, 1279, 1281, 1346, 1396, 1503, 1533, 1538, 1735, 1745-1755, 1936-1937, 1961, 2024, 2033-2038, 2050, 2054, 2094-2095, 2100-2101 |
| sharktank/sharktank/ops/signatures.py                                         |      309 |       40 |     87% |125, 142, 170, 189, 222, 241, 259, 274, 293, 311, 326, 360, 376, 382, 398, 411, 453-459, 480, 533, 541, 574, 594, 607, 632, 659, 680, 703, 734, 764, 801, 819, 875, 918, 970, 1119 |
| sharktank/sharktank/ops/utils.py                                              |       86 |       11 |     87% |32, 37, 80, 221, 224, 227-237, 263 |
| sharktank/sharktank/pipelines/flux/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| sharktank/sharktank/pipelines/flux/flux\_pipeline.py                          |      137 |      109 |     20% |39-92, 120-132, 154-187, 209-227, 237, 243-245, 268-276, 294-316, 319, 338-367, 372-473, 477 |
| sharktank/sharktank/tools/convert\_dataset.py                                 |       27 |        1 |     96% |        51 |
| sharktank/sharktank/tools/import\_hf\_dataset.py                              |       16 |       10 |     38% | 33-54, 60 |
| sharktank/sharktank/tools/sharktank.py                                        |       37 |        3 |     92% |60, 65, 83 |
| sharktank/sharktank/transforms/dataset/\_\_init\_\_.py                        |        2 |        0 |    100% |           |
| sharktank/sharktank/transforms/dataset/dataset.py                             |       14 |        1 |     93% |        24 |
| sharktank/sharktank/transforms/dataset/sharding.py                            |       38 |       28 |     26% |32-34, 37-49, 54-68, 71 |
| sharktank/sharktank/types/\_\_init\_\_.py                                     |        6 |        0 |    100% |           |
| sharktank/sharktank/types/gguf\_interop/\_\_init\_\_.py                       |        2 |        0 |    100% |           |
| sharktank/sharktank/types/gguf\_interop/base.py                               |       70 |       50 |     29% |42-44, 48-61, 65-81, 99-104, 115-138, 142-163, 167-168 |
| sharktank/sharktank/types/gguf\_interop/layouts.py                            |      104 |       68 |     35% |47-49, 53-60, 64, 67, 107-110, 118-144, 157, 160, 170-217, 226-227, 230, 234, 237, 246-247, 250, 254, 257, 282-283, 287-300, 304, 307 |
| sharktank/sharktank/types/layout\_utils.py                                    |      111 |        8 |     93% |90, 125, 127, 204-207, 250, 259 |
| sharktank/sharktank/types/layouts.py                                          |      281 |       46 |     84% |117, 160-161, 166-167, 192-199, 290-302, 400, 404, 480, 488, 496, 504, 517, 530, 538, 550, 553, 556-564, 567-577, 684, 700, 705 |
| sharktank/sharktank/types/misc.py                                             |       55 |        1 |     98% |       122 |
| sharktank/sharktank/types/ocp\_floats.py                                      |       98 |       19 |     81% |93-118, 163, 290 |
| sharktank/sharktank/types/pipelining.py                                       |       65 |        6 |     91% |86-87, 103, 130, 165, 189 |
| sharktank/sharktank/types/quantizers.py                                       |      283 |       37 |     87% |130, 184-185, 188-189, 221, 260, 315-316, 325, 343, 349, 363, 369, 395, 397, 460, 462, 500-501, 558, 590, 592, 635, 658, 666, 696-697, 701, 713-723 |
| sharktank/sharktank/types/sharding.py                                         |      155 |       14 |     91% |34, 122-148, 263-264, 267, 321 |
| sharktank/sharktank/types/tensors.py                                          |      854 |      123 |     86% |73-78, 88-90, 161-167, 187-188, 193, 222, 230, 266, 294, 298, 339, 364, 382, 399-401, 411, 415, 423-424, 435-437, 451-453, 467-469, 477-479, 487-488, 518, 523-525, 535-537, 568-570, 600-601, 606, 637-639, 657-659, 668, 670, 722, 755-756, 761, 798, 802, 804, 839, 844, 851, 902-903, 926-927, 932-933, 1091-1095, 1163, 1210-1211, 1229, 1231, 1233, 1241-1243, 1248, 1372, 1385, 1387-1392, 1394, 1396, 1407, 1410, 1412, 1422, 1425, 1454, 1516-1518, 1523, 1548, 1583-1584, 1597, 1612, 1614, 1616, 1685, 1696-1697, 1705-1706, 1714-1715, 1721-1724, 1750-1751, 1878-1879, 1893, 1899, 1903-1904 |
| sharktank/sharktank/types/theta.py                                            |      377 |       50 |     87% |71, 79, 108, 139-149, 169, 181-182, 211-212, 218, 233, 346-350, 413, 474-475, 499-500, 515-516, 540, 546-547, 561-562, 579-580, 604-605, 626-627, 650, 664-666, 672, 758-760, 765 |
| sharktank/sharktank/utils/\_\_init\_\_.py                                     |        1 |        0 |    100% |           |
| sharktank/sharktank/utils/attention.py                                        |       41 |        1 |     98% |        92 |
| sharktank/sharktank/utils/azure.py                                            |       58 |       58 |      0% |     7-121 |
| sharktank/sharktank/utils/cli.py                                              |      116 |       72 |     38% |35-39, 73-190, 201-206, 217-229, 240-272, 287-308, 319, 332-335, 345, 358-370, 387-388, 390, 393-394, 406, 411-414, 423-441 |
| sharktank/sharktank/utils/create\_cache.py                                    |        7 |        1 |     86% |        12 |
| sharktank/sharktank/utils/debugging.py                                        |       91 |       29 |     68% |46-63, 67-74, 81-83, 127, 138 |
| sharktank/sharktank/utils/evaluate.py                                         |       59 |       34 |     42% |29-48, 55, 69-88, 119-120, 129-149 |
| sharktank/sharktank/utils/export.py                                           |       70 |        4 |     94% |140, 151, 179, 212 |
| sharktank/sharktank/utils/export\_artifacts.py                                |      176 |      129 |     27% |39-47, 61, 68, 75, 82, 89, 124-159, 170, 183-187, 213-233, 251-263, 267-273, 285-301, 325-367, 391-427, 451-464, 490-516, 538-544 |
| sharktank/sharktank/utils/hf.py                                               |       35 |       24 |     31% |26-54, 64-78 |
| sharktank/sharktank/utils/hf\_datasets.py                                     |       75 |       22 |     71% |37-55, 65, 73, 82-83, 88, 478-496, 500 |
| sharktank/sharktank/utils/io.py                                               |       39 |        9 |     77% |65-72, 83-86 |
| sharktank/sharktank/utils/iree.py                                             |      287 |       53 |     82% |187, 198-201, 299, 303, 307, 313-320, 326, 332, 338, 380, 498-499, 546-547, 556, 660-681, 696-703, 715-723, 746 |
| sharktank/sharktank/utils/llm\_artifacts.py                                   |       31 |        2 |     94% |    38, 43 |
| sharktank/sharktank/utils/llm\_utils.py                                       |      367 |       89 |     76% |48-55, 77-78, 81-86, 155, 159-161, 273-275, 303, 345, 376, 379-419, 432, 435-443, 469-472, 474-475, 477, 491, 495, 524-531, 534-547, 571-577, 595 |
| sharktank/sharktank/utils/load\_llm.py                                        |      182 |       84 |     54% |51-65, 70-82, 125, 170, 177, 180-189, 197, 199, 204-214, 224-242, 264-269, 293-370 |
| sharktank/sharktank/utils/logging.py                                          |        6 |        0 |    100% |           |
| sharktank/sharktank/utils/math.py                                             |       12 |        5 |     58% | 17, 25-28 |
| sharktank/sharktank/utils/misc.py                                             |       58 |       11 |     81% |35, 100, 105-115 |
| sharktank/sharktank/utils/patching.py                                         |       94 |       43 |     54% |56, 75-78, 87-93, 98, 108-133, 141-154, 157-168, 197, 231, 233, 238 |
| sharktank/sharktank/utils/random.py                                           |       23 |        0 |    100% |           |
| sharktank/sharktank/utils/testing.py                                          |      432 |      202 |     53% |157-273, 296-302, 313, 326-336, 349-373, 379-400, 416-425, 430-443, 447-451, 490-574, 609, 662-665, 694-700, 731, 755-763, 778, 783, 789-793, 801-804, 810-817, 825-829, 897, 938, 969, 985-994, 1019-1021, 1034, 1044, 1047, 1086 |
| sharktank/sharktank/utils/tokenizer.py                                        |       51 |       35 |     31% |34-38, 42-46, 50, 63-66, 69-72, 76, 80-81, 85-110 |
| sharktank/sharktank/utils/tree.py                                             |       71 |        2 |     97% |   81, 220 |
| sharktank/tests/evaluate/perplexity\_iree\_test.py                            |       48 |       30 |     38% |30-36, 44-58, 61-71, 81-86, 91-105, 109 |
| sharktank/tests/evaluate/perplexity\_torch\_test.py                           |       38 |       22 |     42% |29-34, 37-47, 50-66, 71-76, 80 |
| sharktank/tests/examples/main\_test.py                                        |       24 |        1 |     96% |        45 |
| sharktank/tests/examples/paged\_llm\_v1\_test.py                              |       16 |        5 |     69% |     29-33 |
| sharktank/tests/export\_ir/export\_test.py                                    |       38 |        0 |    100% |           |
| sharktank/tests/kernels/attention\_template\_test.py                          |       76 |        8 |     89% |23, 112-118, 138 |
| sharktank/tests/kernels/attention\_wave\_test.py                              |       23 |        2 |     91% |    25, 59 |
| sharktank/tests/kernels/batch\_matmul\_transpose\_b\_test.py                  |       85 |        6 |     93% |110-113, 126, 153 |
| sharktank/tests/kernels/conv\_2d\_nchw\_fchw\_test.py                         |       42 |        2 |     95% |    63, 91 |
| sharktank/tests/kernels/einsum\_q4\_test.py                                   |       69 |        3 |     96% |94, 120, 141 |
| sharktank/tests/kernels/gemm\_fp4\_asm\_test.py                               |       58 |       39 |     33% |26, 62-125 |
| sharktank/tests/kernels/mlir\_kernel\_test.py                                 |       21 |        0 |    100% |           |
| sharktank/tests/kernels/mmt\_block\_scaled\_offset\_q4\_test.py               |       46 |        3 |     93% |49, 79, 100 |
| sharktank/tests/kernels/mmt\_block\_scaled\_q8\_test.py                       |       43 |        3 |     93% |46, 74, 94 |
| sharktank/tests/kernels/mmt\_super\_block\_scaled\_offset\_q4\_test.py        |       71 |       20 |     72% |39-64, 97, 156, 174 |
| sharktank/tests/kernels/mmtfp\_test.py                                        |       60 |        4 |     93% |57, 81, 99, 125 |
| sharktank/tests/kernels/pooling\_nchw\_sum\_test.py                           |       42 |        2 |     95% |    58, 78 |
| sharktank/tests/kernels/rotary\_test.py                                       |       18 |        0 |    100% |           |
| sharktank/tests/kernels/topk\_test.py                                         |       31 |        0 |    100% |           |
| sharktank/tests/kernels/wave/mxfp4\_gemm\_test.py                             |       64 |       39 |     39% |33, 65-136 |
| sharktank/tests/kernels/wave/wave\_utils\_test.py                             |       30 |        0 |    100% |           |
| sharktank/tests/layers/base\_test.py                                          |       22 |        0 |    100% |           |
| sharktank/tests/layers/configs\_test.py                                       |       11 |        0 |    100% |           |
| sharktank/tests/layers/kv\_cache\_test.py                                     |       49 |        0 |    100% |           |
| sharktank/tests/layers/linear\_test.py                                        |       82 |        1 |     99% |       196 |
| sharktank/tests/layers/mixture\_of\_experts\_block\_test.py                   |       58 |        1 |     98% |       329 |
| sharktank/tests/layers/mmdit\_test.py                                         |       56 |        1 |     98% |        96 |
| sharktank/tests/layers/paged\_llama\_attention\_block\_test.py                |      278 |       47 |     83% |51-68, 79-154, 491-498, 737 |
| sharktank/tests/layers/rotary\_embedding\_hf\_test.py                         |      239 |       10 |     96% |303-304, 398-405 |
| sharktank/tests/layers/rotary\_embedding\_test.py                             |       61 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_conv2d\_with\_iree\_test.py                   |       78 |        0 |    100% |           |
| sharktank/tests/models/clip/clip\_test.py                                     |      251 |       53 |     79% |91, 96-111, 121, 131, 211-255, 300-327, 352-386, 395, 405 |
| sharktank/tests/models/deepseek/test\_deepseek.py                             |       34 |        3 |     91% |     71-82 |
| sharktank/tests/models/flux/flux\_test.py                                     |      155 |       49 |     68% |61-63, 82, 193-211, 220-238, 284, 291, 298-317, 327-353, 363, 372, 381-389, 393 |
| sharktank/tests/models/grok/test\_grok.py                                     |       25 |        0 |    100% |           |
| sharktank/tests/models/llama4/llama4\_test.py                                 |       61 |        8 |     87% |103-123, 140 |
| sharktank/tests/models/llama4/moe\_test.py                                    |       90 |        1 |     99% |       192 |
| sharktank/tests/models/llama/attention\_test.py                               |       67 |        1 |     99% |       190 |
| sharktank/tests/models/llama/benchmark\_amdgpu\_test.py                       |      110 |       69 |     37% |34, 37-48, 58-92, 95-105, 114-168, 193-208, 212-227, 231-250, 255-275, 282-352, 369-390, 396-414, 418 |
| sharktank/tests/models/llama/quantized\_test.py                               |       20 |        0 |    100% |           |
| sharktank/tests/models/llama/quark\_parity\_test.py                           |       55 |       40 |     27% |21-22, 29-101, 105 |
| sharktank/tests/models/llama/rot\_emb\_test.py                                |       37 |        1 |     97% |        81 |
| sharktank/tests/models/llama/test\_llama.py                                   |       37 |        3 |     92% |     72-83 |
| sharktank/tests/models/llama/toy\_llama\_test.py                              |       69 |        1 |     99% |        29 |
| sharktank/tests/models/punet/resnet\_test.py                                  |       42 |        1 |     98% |        93 |
| sharktank/tests/models/punet/sharded\_resnet\_block\_with\_iree\_test.py      |       43 |       12 |     72% |    76-113 |
| sharktank/tests/models/punet/up\_down\_block\_test.py                         |       49 |        1 |     98% |       149 |
| sharktank/tests/models/t5/t5\_test.py                                         |      269 |       59 |     78% |80-108, 146-174, 187-221, 266, 280, 289, 298, 307, 316, 325, 435-477, 522, 531, 540, 549, 558 |
| sharktank/tests/models/vae/vae\_test.py                                       |      213 |      114 |     46% |61-96, 102-111, 116-125, 129-224, 249-263, 268-281, 341-344, 349-440, 541-548, 557-561, 566-571, 577 |
| sharktank/tests/ops/ops\_test.py                                              |      622 |       30 |     95% |177-180, 245-251, 258-264, 271-278, 632-637, 1077 |
| sharktank/tests/ops/pipeline\_parallelized\_test.py                           |      153 |        4 |     97% |57, 181, 193, 203 |
| sharktank/tests/ops/qconv\_test.py                                            |       97 |       12 |     88% |192-228, 232 |
| sharktank/tests/ops/quantized\_test.py                                        |       85 |        0 |    100% |           |
| sharktank/tests/ops/sharded\_test.py                                          |     1337 |       19 |     99% |596-602, 684, 1915, 1918, 1922, 1945, 1949, 2123, 2132-2134, 2142, 2350 |
| sharktank/tests/ops/test\_attention\_ops.py                                   |       28 |        1 |     96% |       106 |
| sharktank/tests/pipelines/flux/flux\_pipeline\_test.py                        |       41 |       23 |     44% |25-27, 32-65, 77-121, 128, 135 |
| sharktank/tests/pytest\_fixtures\_test.py                                     |       19 |        0 |    100% |           |
| sharktank/tests/tools/convert\_dataset\_test.py                               |       22 |        0 |    100% |           |
| sharktank/tests/tools/convert\_to\_json\_test.py                              |       18 |        0 |    100% |           |
| sharktank/tests/tools/sharktank\_test.py                                      |       19 |        0 |    100% |           |
| sharktank/tests/transforms/dataset\_transforms\_test.py                       |       32 |        1 |     97% |        86 |
| sharktank/tests/types/dataset\_test.py                                        |      183 |       36 |     80% |239-259, 268-294, 303 |
| sharktank/tests/types/layout\_utils\_test.py                                  |       75 |        1 |     99% |       231 |
| sharktank/tests/types/layouts\_test.py                                        |       68 |        1 |     99% |       148 |
| sharktank/tests/types/misc\_test.py                                           |       14 |        0 |    100% |           |
| sharktank/tests/types/quantizers\_test.py                                     |      266 |        1 |     99% |       634 |
| sharktank/tests/types/tensors\_test.py                                        |      164 |        1 |     99% |       221 |
| sharktank/tests/utils/iree\_test.py                                           |       56 |        6 |     89% | 69-73, 93 |
| sharktank/tests/utils/misc\_test.py                                           |        9 |        0 |    100% |           |
| sharktank/tests/utils/patching\_test.py                                       |       44 |        0 |    100% |           |
| sharktank/tests/utils/testing\_test.py                                        |      137 |        4 |     97% |   395-408 |
| sharktank/tests/utils/tree\_test.py                                           |       20 |        0 |    100% |           |
|                                                                     **TOTAL** | **21370** | **4349** | **80%** |           |


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