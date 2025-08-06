# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/nod-ai/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                          |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| sharktank/conftest.py                                                         |      176 |       15 |     91% |353, 358, 365, 372, 447, 471, 476-479, 489, 494, 508, 532, 535-536 |
| sharktank/integration/models/punet/integration\_test.py                       |       94 |       57 |     39% |15-16, 21-31, 52-62, 70-80, 90-101, 110-121, 131-143, 150, 155-167, 174, 185-192, 210-220, 242-255 |
| sharktank/setup.py                                                            |       18 |       18 |      0% |      7-34 |
| sharktank/sharktank/\_\_init\_\_.py                                           |        4 |        1 |     75% |        15 |
| sharktank/sharktank/build/\_\_init\_\_.py                                     |        1 |        1 |      0% |         7 |
| sharktank/sharktank/build/actions.py                                          |       45 |       45 |      0% |     7-109 |
| sharktank/sharktank/evaluate/perplexity\_iree.py                              |      249 |      210 |     16% |70-91, 95-101, 106-128, 136-163, 171-208, 218-245, 250-295, 300-349, 352-407, 414-462, 475-529, 536-585, 589 |
| sharktank/sharktank/evaluate/perplexity\_torch.py                             |      181 |      144 |     20% |54-57, 61-67, 72-94, 113-139, 143-161, 165-229, 243-282, 296-352, 377-402, 406-446, 450 |
| sharktank/sharktank/examples/paged\_llm\_v1.py                                |       63 |       54 |     14% |32-116, 122 |
| sharktank/sharktank/examples/pipeline/export\_ppffn\_net.py                   |       83 |        5 |     94% |95, 144, 150, 174, 181 |
| sharktank/sharktank/examples/sharding/export\_ffn\_net.py                     |       59 |       13 |     78% |51-63, 82, 88, 113, 120 |
| sharktank/sharktank/examples/sharding/shard\_llm\_dataset.py                  |       23 |        3 |     87% |33, 36, 51 |
| sharktank/sharktank/kernels/\_\_init\_\_.py                                   |       13 |        0 |    100% |           |
| sharktank/sharktank/kernels/attention.py                                      |       22 |        2 |     91% |     40-75 |
| sharktank/sharktank/kernels/base.py                                           |       52 |        5 |     90% |136, 155-160 |
| sharktank/sharktank/kernels/batch\_matmul\_transpose\_b.py                    |       49 |        0 |    100% |           |
| sharktank/sharktank/kernels/bitcast.py                                        |       63 |       40 |     37% |58-69, 75-88, 97-108, 114-127, 136-139 |
| sharktank/sharktank/kernels/conv\_2d\_nchw\_fchw.py                           |       64 |        0 |    100% |           |
| sharktank/sharktank/kernels/einsum\_2args\_q4.py                              |      122 |        2 |     98% |   69, 179 |
| sharktank/sharktank/kernels/mlir\_kernel.py                                   |      204 |       18 |     91% |40, 43, 47, 112, 123, 129, 131, 220, 262, 269, 277, 321, 329, 369-374, 382 |
| sharktank/sharktank/kernels/mmt\_block\_scaled\_offset\_q4.py                 |       50 |        3 |     94% |     94-96 |
| sharktank/sharktank/kernels/mmt\_block\_scaled\_q8.py                         |       38 |        0 |    100% |           |
| sharktank/sharktank/kernels/mmt\_super\_block\_scaled\_offset\_q4.py          |       59 |        0 |    100% |           |
| sharktank/sharktank/kernels/mmtfp.py                                          |       41 |        2 |     95% |     68-69 |
| sharktank/sharktank/kernels/pooling\_nchw\_sum.py                             |       38 |        0 |    100% |           |
| sharktank/sharktank/kernels/rotary.py                                         |       31 |        0 |    100% |           |
| sharktank/sharktank/kernels/topk.py                                           |       30 |        0 |    100% |           |
| sharktank/sharktank/kernels/wave/attention.py                                 |       48 |        0 |    100% |           |
| sharktank/sharktank/kernels/wave/mxfp4\_gemm.py                               |       94 |        0 |    100% |           |
| sharktank/sharktank/kernels/wave/utils.py                                     |        9 |        0 |    100% |           |
| sharktank/sharktank/layers/\_\_init\_\_.py                                    |       15 |        0 |    100% |           |
| sharktank/sharktank/layers/activations.py                                     |        3 |        0 |    100% |           |
| sharktank/sharktank/layers/base.py                                            |      177 |       27 |     85% |131, 206-209, 224, 242, 259-260, 269, 298, 366-374, 385-398, 400, 404-407, 411, 417, 424 |
| sharktank/sharktank/layers/causal\_llm.py                                     |       70 |        6 |     91% |54, 58-63, 133 |
| sharktank/sharktank/layers/configs/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| sharktank/sharktank/layers/configs/config.py                                  |      172 |       16 |     91% |17, 139, 196, 205-211, 219, 234, 248-254, 267, 269, 289, 313 |
| sharktank/sharktank/layers/configs/llm\_configs.py                            |      402 |       78 |     81% |28-29, 214, 218, 222, 229, 231, 235, 237, 246, 249-252, 278-285, 295-298, 304-307, 311-316, 328-331, 342-343, 354-355, 446-447, 452, 543, 549-557, 566-578, 617, 639, 663-667, 703-706, 710-714 |
| sharktank/sharktank/layers/conv.py                                            |      100 |       61 |     39% |48, 58, 61, 63, 80, 95-110, 113-143, 157-172, 175-205 |
| sharktank/sharktank/layers/ffn\_block.py                                      |       26 |        0 |    100% |           |
| sharktank/sharktank/layers/ffn\_moe\_block.py                                 |       83 |       25 |     70% |65-73, 203-237, 243-246, 253-259 |
| sharktank/sharktank/layers/latent\_attention\_block.py                        |       55 |        4 |     93% |42, 62, 85-86 |
| sharktank/sharktank/layers/linear.py                                          |       42 |        4 |     90% |56, 67, 75, 83 |
| sharktank/sharktank/layers/mixture\_of\_experts\_block.py                     |       71 |        5 |     93% |48, 52, 60, 105-109 |
| sharktank/sharktank/layers/mmdit.py                                           |      103 |        0 |    100% |           |
| sharktank/sharktank/layers/modulation.py                                      |       21 |        0 |    100% |           |
| sharktank/sharktank/layers/norm.py                                            |       37 |        0 |    100% |           |
| sharktank/sharktank/layers/paged\_attention.py                                |      444 |       25 |     94% |142, 147, 155-161, 222, 677, 997, 1007-1008, 1015, 1018, 1022, 1024, 1026, 1045, 1051-1056, 1063-1069, 1073, 1076, 1078 |
| sharktank/sharktank/layers/paged\_llama\_attention\_block.py                  |      122 |       11 |     91% |116, 120-121, 134, 171, 173, 175, 243, 259-263 |
| sharktank/sharktank/layers/rotary\_embedding.py                               |       55 |        0 |    100% |           |
| sharktank/sharktank/layers/rotary\_embedding\_hf.py                           |       78 |        0 |    100% |           |
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
| sharktank/sharktank/models/llama/toy\_llama.py                                |       57 |        7 |     88% |100, 159-165, 169 |
| sharktank/sharktank/models/llm/\_\_init\_\_.py                                |        1 |        0 |    100% |           |
| sharktank/sharktank/models/llm/llm.py                                         |      135 |        4 |     97% |211, 273, 298, 301 |
| sharktank/sharktank/models/llm/testing.py                                     |       24 |        0 |    100% |           |
| sharktank/sharktank/models/punet/config.py                                    |       84 |       34 |     60% |70-82, 87-91, 98-122, 126-130 |
| sharktank/sharktank/models/punet/layers.py                                    |      324 |      191 |     41% |135-180, 195-226, 258, 280-285, 303-330, 341-355, 366-388, 393-397, 400-410, 418-444, 452-499, 513-519, 524-529, 616-624, 627-631, 654-659, 668-695, 720-725, 728, 738-739, 742-744 |
| sharktank/sharktank/models/punet/sharding.py                                  |       31 |        0 |    100% |           |
| sharktank/sharktank/models/punet/testing.py                                   |       65 |        0 |    100% |           |
| sharktank/sharktank/models/punet/tools/sample\_data.py                        |       26 |       21 |     19% |15-20, 33-46, 50-53 |
| sharktank/sharktank/models/t5/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| sharktank/sharktank/models/t5/export.py                                       |       58 |       31 |     47% |37-41, 51-67, 92-100, 112-143 |
| sharktank/sharktank/models/t5/t5.py                                           |      344 |      103 |     70% |126, 160, 189, 236-240, 266-269, 272-284, 313, 326, 334-336, 347, 360, 436-448, 464-480, 517, 557-571, 591-597, 605-642, 649-655, 662, 710, 713, 719-753, 780, 787, 793, 801, 840-842, 850-861, 894-895, 901-905, 911, 926-927, 949-959, 985, 1013, 1018, 1023-1025, 1031, 1034 |
| sharktank/sharktank/models/t5/testing.py                                      |       22 |        0 |    100% |           |
| sharktank/sharktank/models/vae/config.py                                      |       39 |       13 |     67% |44-48, 54-62 |
| sharktank/sharktank/models/vae/layers.py                                      |       97 |        6 |     94% |48, 101, 103, 205, 231, 235 |
| sharktank/sharktank/models/vae/model.py                                       |       67 |        7 |     90% |24-25, 33, 63, 94, 108, 116 |
| sharktank/sharktank/models/vae/testing.py                                     |       14 |        0 |    100% |           |
| sharktank/sharktank/models/vae/tools/diffuser\_ref.py                         |       50 |       13 |     74% |39-60, 87, 104 |
| sharktank/sharktank/models/vae/tools/run\_vae.py                              |       75 |       47 |     37% |64-158, 162 |
| sharktank/sharktank/models/vae/tools/sample\_data.py                          |       14 |        5 |     64% |27-29, 39-40 |
| sharktank/sharktank/ops/\_\_init\_\_.py                                       |       12 |        0 |    100% |           |
| sharktank/sharktank/ops/\_registry.py                                         |      147 |       10 |     93% |113, 118, 233-236, 247, 282, 285-288, 301 |
| sharktank/sharktank/ops/attention\_impls.py                                   |       48 |       33 |     31% |32-38, 48-64, 74-99 |
| sharktank/sharktank/ops/custom\_impls.py                                      |       89 |       28 |     69% |62-66, 84, 100, 121-140, 155, 159-162, 185, 187, 189 |
| sharktank/sharktank/ops/default\_impls.py                                     |      499 |       88 |     82% |119, 121, 153, 155, 157, 190, 192, 194, 257-260, 292, 294, 308-309, 324-331, 345-352, 366, 371-389, 403, 413, 597, 616, 627-629, 668, 718, 789, 794, 799, 805, 837-844, 850, 939, 943, 989, 994-1011, 1016, 1021 |
| sharktank/sharktank/ops/qconv\_impls.py                                       |      123 |       31 |     75% |47, 53, 67-71, 88, 94, 109, 137-142, 168-177, 229, 252, 270-285, 298, 303, 310 |
| sharktank/sharktank/ops/qlinear\_impls.py                                     |       91 |       16 |     82% |40, 65, 84, 88, 102-105, 116-117, 143-144, 162, 165, 188-190, 209 |
| sharktank/sharktank/ops/quantized\_impls.py                                   |      193 |       10 |     95% |83, 85-91, 93-100, 104, 209-211, 348 |
| sharktank/sharktank/ops/shape.py                                              |       28 |        1 |     96% |        84 |
| sharktank/sharktank/ops/sharded\_impls.py                                     |      858 |       61 |     93% |223, 445, 487-489, 496, 504, 519, 544, 555-556, 563-564, 634-643, 693-701, 885, 937, 950, 953, 958, 961, 1020-1022, 1076-1080, 1093, 1110, 1119, 1128, 1155, 1171, 1181, 1205, 1231, 1233, 1243, 1245, 1310, 1466, 1675, 1682, 1708, 1850-1851, 1875, 1938, 1951, 1964, 1968, 2008-2009, 2014-2015 |
| sharktank/sharktank/ops/signatures.py                                         |      693 |       97 |     86% |120, 137, 165, 182, 201, 234, 253, 271, 286, 305, 323, 338, 356, 369, 375, 391, 404, 420, 448-461, 489-495, 514, 529, 540, 559, 581, 622, 644, 652, 671, 696, 721, 740, 754, 782, 795, 820, 846, 864, 886, 903, 943, 962, 974, 992, 1007-1013, 1037, 1056, 1064, 1082, 1101, 1132, 1145, 1164, 1184, 1209, 1242, 1266, 1301, 1328, 1337, 1348, 1367, 1391, 1408, 1432, 1456, 1473, 1510, 1529, 1548, 1567, 1586, 1602, 1619, 1638, 1657, 1687, 1740, 1764, 1781, 1798, 1839 |
| sharktank/sharktank/ops/utils.py                                              |       55 |        3 |     95% |23, 28, 71 |
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
| sharktank/sharktank/types/layouts.py                                          |      272 |       46 |     83% |117, 160-161, 166-167, 192-199, 286-298, 395, 399, 475, 483, 491, 499, 512, 525, 533, 545, 548, 551-559, 562-572, 659, 678, 683 |
| sharktank/sharktank/types/misc.py                                             |       55 |        1 |     98% |       122 |
| sharktank/sharktank/types/ocp\_floats.py                                      |       98 |       19 |     81% |93-118, 163, 290 |
| sharktank/sharktank/types/pipelining.py                                       |       32 |        0 |    100% |           |
| sharktank/sharktank/types/quantizers.py                                       |      271 |       38 |     86% |130, 184-185, 188-189, 210, 222, 254, 309-310, 319, 334, 340, 354, 360, 386, 388, 451, 453, 491-492, 545, 577, 579, 622, 641, 649, 679-680, 684, 696-706 |
| sharktank/sharktank/types/sharding.py                                         |      157 |        7 |     96% |18, 34, 137-148, 321 |
| sharktank/sharktank/types/tensors.py                                          |      831 |      109 |     87% |73-78, 159-165, 185-186, 191, 220, 228, 259, 287, 291, 332, 357, 375, 392-394, 404, 408, 416-417, 428-430, 444-446, 470-472, 480-481, 511, 516-518, 528-530, 593-594, 599, 661, 663, 715, 748-749, 754, 788, 792, 794, 829, 838, 889-890, 913-914, 919-920, 1127, 1174-1175, 1193, 1195, 1197, 1205-1207, 1212, 1336, 1349, 1351-1356, 1358, 1360, 1371, 1374, 1376, 1386, 1389, 1418, 1480-1482, 1487, 1509, 1544-1545, 1558, 1573, 1575, 1577, 1646, 1657-1658, 1666-1667, 1675-1676, 1682-1685, 1711-1712, 1839-1840, 1860, 1864-1865 |
| sharktank/sharktank/types/theta.py                                            |      310 |       44 |     86% |69, 77, 106, 137-147, 167, 179-180, 209-210, 216, 231, 341-345, 405, 453-454, 477-478, 493-494, 514-515, 529-530, 547-548, 590-592, 596, 634-636, 641 |
| sharktank/sharktank/utils/\_\_init\_\_.py                                     |        1 |        0 |    100% |           |
| sharktank/sharktank/utils/azure.py                                            |       58 |       58 |      0% |     7-121 |
| sharktank/sharktank/utils/cli.py                                              |      109 |       66 |     39% |35-39, 72-178, 189-194, 205-217, 228-260, 275-296, 307, 320-323, 333, 346-358, 375-376, 378, 392, 397, 406-424 |
| sharktank/sharktank/utils/create\_cache.py                                    |        7 |        1 |     86% |        12 |
| sharktank/sharktank/utils/debugging.py                                        |       91 |       29 |     68% |46-63, 67-74, 81-83, 127, 138 |
| sharktank/sharktank/utils/evaluate.py                                         |       59 |       32 |     46% |29-48, 55, 69-88, 129-149 |
| sharktank/sharktank/utils/export.py                                           |       70 |        2 |     97% |  179, 212 |
| sharktank/sharktank/utils/export\_artifacts.py                                |      174 |      122 |     30% |22, 39-47, 61, 68, 75, 82, 89, 123-157, 211-231, 249-261, 265-271, 283-299, 323-362, 386-422, 446-459, 485-511, 533-539 |
| sharktank/sharktank/utils/hf.py                                               |       35 |       24 |     31% |26-54, 64-78 |
| sharktank/sharktank/utils/hf\_datasets.py                                     |       75 |       22 |     71% |37-55, 65, 73, 82-83, 88, 478-496, 500 |
| sharktank/sharktank/utils/io.py                                               |       39 |        9 |     77% |65-72, 83-86 |
| sharktank/sharktank/utils/iree.py                                             |      290 |       51 |     82% |37, 200-203, 301, 305, 309, 315-322, 328, 334, 340, 382, 500-501, 558, 662-683, 698-705, 717-725, 748 |
| sharktank/sharktank/utils/load\_llm.py                                        |      199 |       94 |     53% |50-64, 69-81, 124, 169, 176, 179-188, 196, 198, 203-213, 223-241, 274, 294-299, 323-438 |
| sharktank/sharktank/utils/logging.py                                          |        6 |        1 |     83% |        17 |
| sharktank/sharktank/utils/math.py                                             |       12 |        0 |    100% |           |
| sharktank/sharktank/utils/misc.py                                             |       58 |       11 |     81% |35, 100, 105-115 |
| sharktank/sharktank/utils/patching.py                                         |       96 |       44 |     54% |16, 56, 75-78, 87-93, 98, 108-133, 141-154, 157-168, 197, 231, 233, 238 |
| sharktank/sharktank/utils/random.py                                           |       23 |        0 |    100% |           |
| sharktank/sharktank/utils/testing.py                                          |      350 |      179 |     49% |147-263, 286-292, 303, 316-326, 339-363, 369-390, 406-415, 420-433, 437-441, 480-564, 599, 651-654, 719, 766, 771, 777-781, 789-792, 798-805, 813-817, 885 |
| sharktank/sharktank/utils/tokenizer.py                                        |       58 |       41 |     29% |34-38, 42-46, 50, 63-66, 69-72, 76, 80-81, 85-110, 114-119 |
| sharktank/sharktank/utils/tree.py                                             |       71 |        2 |     97% |   81, 220 |
| sharktank/tests/evaluate/perplexity\_iree\_test.py                            |      127 |       86 |     32% |36-42, 50-64, 67-77, 87-92, 97-103, 108-114, 119-133, 138-144, 150-156, 167-173, 179-185, 190-196, 207-214, 219-228, 232 |
| sharktank/tests/evaluate/perplexity\_torch\_test.py                           |       62 |       40 |     35% |32-37, 40-50, 53-69, 74-79, 84-89, 94-101, 106-112, 116 |
| sharktank/tests/examples/main\_test.py                                        |       24 |        1 |     96% |        45 |
| sharktank/tests/examples/paged\_llm\_v1\_test.py                              |       16 |        5 |     69% |     33-37 |
| sharktank/tests/export\_ir/export\_test.py                                    |       38 |        0 |    100% |           |
| sharktank/tests/kernels/attention\_template\_test.py                          |       76 |        8 |     89% |23, 112-118, 138 |
| sharktank/tests/kernels/attention\_wave\_test.py                              |       23 |        2 |     91% |    25, 59 |
| sharktank/tests/kernels/batch\_matmul\_transpose\_b\_test.py                  |       85 |        6 |     93% |110-113, 126, 153 |
| sharktank/tests/kernels/conv\_2d\_nchw\_fchw\_test.py                         |       42 |        2 |     95% |    63, 91 |
| sharktank/tests/kernels/einsum\_q4\_test.py                                   |       69 |        3 |     96% |94, 120, 141 |
| sharktank/tests/kernels/mlir\_kernel\_test.py                                 |       21 |        0 |    100% |           |
| sharktank/tests/kernels/mmt\_block\_scaled\_offset\_q4\_test.py               |       46 |        3 |     93% |49, 79, 100 |
| sharktank/tests/kernels/mmt\_block\_scaled\_q8\_test.py                       |       43 |        3 |     93% |46, 74, 94 |
| sharktank/tests/kernels/mmt\_super\_block\_scaled\_offset\_q4\_test.py        |       71 |       20 |     72% |39-64, 97, 156, 174 |
| sharktank/tests/kernels/mmtfp\_test.py                                        |       60 |        4 |     93% |57, 81, 99, 125 |
| sharktank/tests/kernels/pooling\_nchw\_sum\_test.py                           |       42 |        2 |     95% |    58, 78 |
| sharktank/tests/kernels/rotary\_test.py                                       |       18 |        0 |    100% |           |
| sharktank/tests/kernels/topk\_test.py                                         |       31 |        0 |    100% |           |
| sharktank/tests/kernels/wave/mxfp4\_gemm\_test.py                             |       25 |        2 |     92% |    27, 64 |
| sharktank/tests/kernels/wave/wave\_utils\_test.py                             |       30 |        0 |    100% |           |
| sharktank/tests/layers/base\_test.py                                          |       22 |        0 |    100% |           |
| sharktank/tests/layers/configs\_test.py                                       |       11 |        0 |    100% |           |
| sharktank/tests/layers/kv\_cache\_test.py                                     |       84 |        0 |    100% |           |
| sharktank/tests/layers/linear\_test.py                                        |       82 |        1 |     99% |       196 |
| sharktank/tests/layers/mixture\_of\_experts\_block\_test.py                   |       58 |        1 |     98% |       329 |
| sharktank/tests/layers/mmdit\_test.py                                         |       56 |        1 |     98% |        96 |
| sharktank/tests/layers/paged\_llama\_attention\_block\_test.py                |       53 |       14 |     74% |90-130, 134 |
| sharktank/tests/layers/pipelined\_paged\_attention\_test.py                   |       84 |        0 |    100% |           |
| sharktank/tests/layers/pipelined\_sharded\_paged\_attention\_test.py          |       86 |        0 |    100% |           |
| sharktank/tests/layers/rotary\_embedding\_hf\_test.py                         |       87 |        0 |    100% |           |
| sharktank/tests/layers/rotary\_embedding\_test.py                             |       61 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_conv2d\_with\_iree\_test.py                   |       78 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_paged\_kv\_cache\_test.py                     |       77 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_paged\_latent\_attention\_block\_test.py      |       73 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_paged\_llama\_attention\_block\_test.py       |       77 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_rotary\_embedding\_test.py                    |       29 |        0 |    100% |           |
| sharktank/tests/models/clip/clip\_test.py                                     |      251 |       53 |     79% |91, 96-111, 121, 131, 211-255, 300-327, 352-386, 395, 405 |
| sharktank/tests/models/deepseek/test\_deepseek.py                             |       47 |       13 |     72% |     76-99 |
| sharktank/tests/models/deepseek/test\_sharded.py                              |       87 |       12 |     86% |   169-204 |
| sharktank/tests/models/flux/flux\_test.py                                     |      155 |       59 |     62% |61-63, 82, 153-172, 176-182, 193-211, 220-238, 284, 291, 298-317, 327-353, 363, 372, 381-389, 393 |
| sharktank/tests/models/grok/test\_grok.py                                     |       25 |        0 |    100% |           |
| sharktank/tests/models/llama4/llama4\_test.py                                 |       60 |        8 |     87% |103-123, 137 |
| sharktank/tests/models/llama4/moe\_test.py                                    |       90 |        1 |     99% |       192 |
| sharktank/tests/models/llama/attention\_test.py                               |       59 |        1 |     98% |       189 |
| sharktank/tests/models/llama/benchmark\_amdgpu\_test.py                       |      135 |       85 |     37% |34, 37-48, 58-92, 95-105, 114-168, 193-208, 212-227, 231-250, 255-275, 282-352, 369-390, 396-414, 421-461, 476-491, 497-515, 519 |
| sharktank/tests/models/llama/quantized\_test.py                               |       20 |        0 |    100% |           |
| sharktank/tests/models/llama/quark\_parity\_test.py                           |       55 |       40 |     27% |21-22, 29-101, 105 |
| sharktank/tests/models/llama/rot\_emb\_test.py                                |       37 |        1 |     97% |        81 |
| sharktank/tests/models/llama/sharded\_llama\_test.py                          |      187 |       11 |     94% |170-191, 268, 325 |
| sharktank/tests/models/llama/test\_llama.py                                   |       48 |       11 |     77% |     77-97 |
| sharktank/tests/models/punet/resnet\_test.py                                  |       42 |        1 |     98% |        93 |
| sharktank/tests/models/punet/sharded\_resnet\_block\_with\_iree\_test.py      |       43 |       12 |     72% |    76-113 |
| sharktank/tests/models/punet/up\_down\_block\_test.py                         |       49 |        1 |     98% |       149 |
| sharktank/tests/models/t5/t5\_test.py                                         |      270 |       59 |     78% |80-108, 146-174, 187-221, 266, 280, 289, 298, 307, 316, 325, 435-477, 526, 535, 544, 553, 562 |
| sharktank/tests/models/vae/vae\_test.py                                       |      213 |      114 |     46% |61-96, 102-111, 116-125, 129-224, 249-263, 268-281, 341-344, 349-440, 541-548, 557-561, 566-571, 577 |
| sharktank/tests/ops/ops\_test.py                                              |      593 |       30 |     95% |163-166, 231-237, 244-250, 257-264, 618-623, 1024 |
| sharktank/tests/ops/pipeline\_parallelized\_test.py                           |      153 |        4 |     97% |57, 181, 193, 203 |
| sharktank/tests/ops/qconv\_test.py                                            |       97 |       12 |     88% |192-228, 232 |
| sharktank/tests/ops/quantized\_test.py                                        |       67 |        0 |    100% |           |
| sharktank/tests/ops/sharded\_test.py                                          |     1313 |       19 |     99% |554-560, 642, 1869, 1872, 1876, 1899, 1903, 2077, 2086-2088, 2096, 2264 |
| sharktank/tests/pipelines/flux/flux\_pipeline\_test.py                        |       41 |       23 |     44% |25-27, 32-65, 77-121, 128, 135 |
| sharktank/tests/pytest\_fixtures\_test.py                                     |       19 |        0 |    100% |           |
| sharktank/tests/tools/convert\_dataset\_test.py                               |       22 |        0 |    100% |           |
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
|                                                                     **TOTAL** | **21151** | **4054** | **81%** |           |


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