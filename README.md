# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/nod-ai/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                          |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| sharktank/conftest.py                                                         |      149 |        9 |     94% |291, 298, 305, 339, 363, 368-371, 400 |
| sharktank/integration/models/punet/integration\_test.py                       |       94 |       57 |     39% |15-16, 21-31, 52-62, 70-80, 90-101, 110-121, 131-143, 150, 155-167, 174, 185-192, 210-220, 242-255 |
| sharktank/setup.py                                                            |       18 |       18 |      0% |      7-34 |
| sharktank/sharktank/\_\_init\_\_.py                                           |        4 |        1 |     75% |        15 |
| sharktank/sharktank/build/\_\_init\_\_.py                                     |        1 |        1 |      0% |         7 |
| sharktank/sharktank/build/actions.py                                          |       45 |       45 |      0% |     7-109 |
| sharktank/sharktank/evaluate/perplexity\_iree.py                              |      247 |      208 |     16% |70-90, 94-100, 105-127, 135-171, 179-215, 225-251, 256-301, 306-355, 358-413, 420-468, 481-535, 542-591, 595 |
| sharktank/sharktank/evaluate/perplexity\_torch.py                             |      183 |      146 |     20% |53-55, 59-65, 70-92, 111-138, 142-159, 163-227, 241-280, 294-349, 373-396, 400-440, 444 |
| sharktank/sharktank/examples/export\_paged\_llm\_v1.py                        |      128 |       46 |     64% |40, 88-107, 207-298, 302 |
| sharktank/sharktank/examples/paged\_llm\_v1.py                                |       48 |       39 |     19% |32-118, 122 |
| sharktank/sharktank/examples/pipeline/export\_ppffn\_net.py                   |       66 |        4 |     94% |139, 145, 176, 183 |
| sharktank/sharktank/examples/sharding/export\_ffn\_net.py                     |       59 |       13 |     78% |51-63, 82, 88, 113, 120 |
| sharktank/sharktank/kernels/\_\_init\_\_.py                                   |       13 |        0 |    100% |           |
| sharktank/sharktank/kernels/attention.py                                      |       22 |        0 |    100% |           |
| sharktank/sharktank/kernels/base.py                                           |       52 |        5 |     90% |136, 155-160 |
| sharktank/sharktank/kernels/batch\_matmul\_transpose\_b.py                    |       49 |        0 |    100% |           |
| sharktank/sharktank/kernels/bitcast.py                                        |       63 |       40 |     37% |58-69, 75-88, 97-108, 114-127, 136-139 |
| sharktank/sharktank/kernels/conv\_2d\_nchw\_fchw.py                           |       64 |        0 |    100% |           |
| sharktank/sharktank/kernels/einsum\_2args\_q4.py                              |      122 |        2 |     98% |   69, 179 |
| sharktank/sharktank/kernels/gemm\_fp4\_asm.py                                 |       46 |       23 |     50% |30-46, 66-204, 218, 232, 251-257 |
| sharktank/sharktank/kernels/mlir\_kernel.py                                   |      204 |       18 |     91% |40, 43, 47, 112, 123, 129, 131, 220, 262, 269, 277, 321, 329, 369-374, 382 |
| sharktank/sharktank/kernels/mmt\_block\_scaled\_offset\_q4.py                 |       50 |        3 |     94% |     94-96 |
| sharktank/sharktank/kernels/mmt\_block\_scaled\_q8.py                         |       38 |        0 |    100% |           |
| sharktank/sharktank/kernels/mmt\_super\_block\_scaled\_offset\_q4.py          |       59 |        0 |    100% |           |
| sharktank/sharktank/kernels/mmtfp.py                                          |       41 |        2 |     95% |     68-69 |
| sharktank/sharktank/kernels/pooling\_nchw\_sum.py                             |       38 |        0 |    100% |           |
| sharktank/sharktank/kernels/rotary.py                                         |       31 |        0 |    100% |           |
| sharktank/sharktank/kernels/topk.py                                           |       30 |        0 |    100% |           |
| sharktank/sharktank/kernels/wave/attention.py                                 |       48 |        0 |    100% |           |
| sharktank/sharktank/kernels/wave/extend\_attention.py                         |       58 |       31 |     47% |62-97, 140-211 |
| sharktank/sharktank/kernels/wave/mxfp4\_gemm.py                               |       99 |       74 |     25% |41-112, 122-161, 185-234 |
| sharktank/sharktank/kernels/wave/utils.py                                     |      135 |      110 |     19% |68-74, 82-175, 213-253, 274-311 |
| sharktank/sharktank/layers/\_\_init\_\_.py                                    |       16 |        0 |    100% |           |
| sharktank/sharktank/layers/activations.py                                     |        3 |        0 |    100% |           |
| sharktank/sharktank/layers/base.py                                            |      177 |       27 |     85% |131, 206-209, 224, 242, 259-260, 269, 298, 366-374, 385-398, 400, 404-407, 411, 417, 424 |
| sharktank/sharktank/layers/causal\_llm.py                                     |       22 |        7 |     68% |     58-64 |
| sharktank/sharktank/layers/configs/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| sharktank/sharktank/layers/configs/config.py                                  |      170 |       15 |     91% |139, 196, 205-211, 219, 234, 248-254, 267, 269, 289, 313 |
| sharktank/sharktank/layers/configs/llm\_configs.py                            |      516 |      140 |     73% |44-46, 209, 211, 213, 215, 217, 219, 221, 227, 229, 231, 233, 241, 245, 249, 256, 258, 262, 264, 275, 278, 281-284, 287-307, 310-317, 327-330, 336-339, 343-348, 362-363, 374-375, 386-387, 417, 446-450, 454, 460, 540, 567-568, 573, 627-641, 645-648, 656-688, 732, 738-746, 755-767, 806, 828, 852-856, 892-895, 899-903 |
| sharktank/sharktank/layers/conv.py                                            |      100 |       61 |     39% |48, 58, 61, 63, 80, 95-110, 113-143, 157-172, 175-205 |
| sharktank/sharktank/layers/ffn\_block.py                                      |       26 |        0 |    100% |           |
| sharktank/sharktank/layers/ffn\_moe\_block.py                                 |       83 |       25 |     70% |65-73, 203-237, 243-246, 253-259 |
| sharktank/sharktank/layers/kv\_cache.py                                       |       16 |        0 |    100% |           |
| sharktank/sharktank/layers/latent\_attention\_block.py                        |       52 |        5 |     90% |42, 61, 66, 76, 96 |
| sharktank/sharktank/layers/linear.py                                          |       43 |        4 |     91% |55, 66, 74, 84 |
| sharktank/sharktank/layers/mixture\_of\_experts\_block.py                     |       71 |        4 |     94% |48, 52, 60, 105 |
| sharktank/sharktank/layers/mmdit.py                                           |      103 |        0 |    100% |           |
| sharktank/sharktank/layers/modulation.py                                      |       21 |        0 |    100% |           |
| sharktank/sharktank/layers/norm.py                                            |       37 |        0 |    100% |           |
| sharktank/sharktank/layers/paged\_attention.py                                |      266 |       18 |     93% |198, 352-355, 359, 379, 381, 419, 759-764, 920, 960-965 |
| sharktank/sharktank/layers/paged\_llama\_attention\_block.py                  |      144 |       17 |     88% |80, 136, 163-180, 186-187, 334, 336, 338, 460-462 |
| sharktank/sharktank/layers/rotary\_embedding.py                               |       56 |        3 |     95% |108, 115, 145 |
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
| sharktank/sharktank/models/deepseek/toy\_deepseek.py                          |       33 |        9 |     73% | 84-94, 98 |
| sharktank/sharktank/models/dummy/\_\_init\_\_.py                              |        1 |        0 |    100% |           |
| sharktank/sharktank/models/dummy/dummy.py                                     |       39 |        0 |    100% |           |
| sharktank/sharktank/models/flux/\_\_init\_\_.py                               |        1 |        0 |    100% |           |
| sharktank/sharktank/models/flux/compile.py                                    |        1 |        0 |    100% |           |
| sharktank/sharktank/models/flux/export.py                                     |       55 |       24 |     56% |35-36, 56, 80, 95-98, 104-127 |
| sharktank/sharktank/models/flux/export\_flux\_transformer\_mlir.py            |       13 |       13 |      0% |      7-38 |
| sharktank/sharktank/models/flux/flux.py                                       |      233 |       29 |     88% |82-91, 117-121, 129, 135, 137, 142, 147, 152, 218, 222, 235, 242, 268-279, 288, 407 |
| sharktank/sharktank/models/flux/testing.py                                    |       54 |       10 |     81% |31, 154, 209-227 |
| sharktank/sharktank/models/grok/testing.py                                    |       22 |        0 |    100% |           |
| sharktank/sharktank/models/grok/toy\_grok.py                                  |       31 |        6 |     81% | 67-72, 76 |
| sharktank/sharktank/models/llama4/testing.py                                  |       40 |        1 |     98% |        17 |
| sharktank/sharktank/models/llama/testing.py                                   |       58 |        0 |    100% |           |
| sharktank/sharktank/models/llama/toy\_llama.py                                |       51 |        6 |     88% |155-161, 165 |
| sharktank/sharktank/models/llm/\_\_init\_\_.py                                |        1 |        0 |    100% |           |
| sharktank/sharktank/models/llm/config.py                                      |       43 |        4 |     91% |     39-42 |
| sharktank/sharktank/models/llm/export.py                                      |       78 |       20 |     74% |25-30, 36, 71-73, 81-85, 90-93, 119, 127-130, 157 |
| sharktank/sharktank/models/llm/llm.py                                         |       94 |        4 |     96% |178, 202, 231, 234 |
| sharktank/sharktank/models/llm/testing.py                                     |       23 |       23 |      0% |      1-86 |
| sharktank/sharktank/models/punet/config.py                                    |       84 |       34 |     60% |70-82, 87-91, 98-122, 126-130 |
| sharktank/sharktank/models/punet/layers.py                                    |      324 |      191 |     41% |135-180, 195-226, 258, 280-285, 303-330, 341-355, 366-388, 393-397, 400-412, 420-445, 453-500, 514-520, 525-530, 617-625, 628-632, 655-660, 669-696, 721-726, 729, 739-740, 743-745 |
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
| sharktank/sharktank/ops/attention\_impls.py                                   |      113 |        5 |     96% |47-49, 140, 153 |
| sharktank/sharktank/ops/cpu\_impls.py                                         |       20 |        1 |     95% |        43 |
| sharktank/sharktank/ops/custom\_impls.py                                      |      115 |       44 |     62% |65-69, 87, 103, 124-143, 159-188, 218, 222-225, 248, 250, 252 |
| sharktank/sharktank/ops/default\_impls.py                                     |      588 |      114 |     81% |176-191, 220, 222, 254, 256, 258, 291, 293, 295, 350-358, 363-370, 382-390, 422, 424, 438-439, 454-465, 479-490, 509-527, 541, 551, 730, 749, 760-762, 801, 901, 972, 977, 982, 988, 1020-1031, 1037, 1126, 1130, 1181-1198, 1203, 1208 |
| sharktank/sharktank/ops/qconv\_impls.py                                       |      123 |       31 |     75% |47, 53, 67-71, 88, 94, 109, 137-142, 168-177, 229, 252, 270-285, 298, 303, 310 |
| sharktank/sharktank/ops/qlinear\_impls.py                                     |       91 |       16 |     82% |41, 72, 91, 95, 109-112, 123-124, 150-151, 170, 173, 196-198, 217 |
| sharktank/sharktank/ops/quantized\_impls.py                                   |      225 |       14 |     94% |81, 89, 91-97, 99-106, 117-118, 142, 255-257, 394, 542 |
| sharktank/sharktank/ops/shape.py                                              |       28 |        1 |     96% |        84 |
| sharktank/sharktank/ops/sharded\_impls.py                                     |      891 |       91 |     90% |228, 280-289, 472, 493, 534-536, 543, 551, 566, 576-580, 590-595, 602-603, 610-611, 681-690, 740-748, 892, 935, 987, 1001, 1003, 1006, 1011, 1014, 1080-1082, 1136-1140, 1153, 1170, 1179, 1187-1189, 1215, 1231, 1241, 1265, 1291, 1293, 1303, 1305, 1370, 1420, 1527, 1557, 1562, 1759, 1769-1779, 1960-1961, 1985, 2048, 2057-2062, 2074, 2078, 2118-2119, 2124-2125 |
| sharktank/sharktank/ops/signatures.py                                         |      325 |       41 |     87% |130, 147, 198, 224, 263, 296, 315, 333, 348, 367, 385, 400, 434, 450, 456, 472, 485, 527-533, 554, 607, 615, 648, 682, 695, 720, 747, 768, 791, 822, 852, 889, 907, 963, 1006, 1058, 1212 |
| sharktank/sharktank/ops/utils.py                                              |       86 |       11 |     87% |32, 37, 80, 221, 224, 227-237, 263 |
| sharktank/sharktank/pipelines/flux/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| sharktank/sharktank/pipelines/flux/flux\_pipeline.py                          |      137 |      109 |     20% |39-92, 120-132, 154-187, 209-227, 237, 243-245, 268-276, 294-316, 319, 338-367, 372-473, 477 |
| sharktank/sharktank/tools/convert\_dataset.py                                 |       27 |        1 |     96% |        51 |
| sharktank/sharktank/tools/import\_hf\_dataset.py                              |       16 |       10 |     38% | 33-54, 60 |
| sharktank/sharktank/transforms/dataset/\_\_init\_\_.py                        |        2 |        0 |    100% |           |
| sharktank/sharktank/transforms/dataset/dataset.py                             |       21 |        6 |     71% | 25, 43-47 |
| sharktank/sharktank/transforms/dataset/sharding.py                            |       38 |       28 |     26% |32-34, 37-49, 54-68, 71 |
| sharktank/sharktank/types/\_\_init\_\_.py                                     |        6 |        0 |    100% |           |
| sharktank/sharktank/types/gguf\_interop/\_\_init\_\_.py                       |        2 |        0 |    100% |           |
| sharktank/sharktank/types/gguf\_interop/base.py                               |       70 |       50 |     29% |42-44, 48-61, 65-81, 99-104, 115-138, 142-163, 167-168 |
| sharktank/sharktank/types/gguf\_interop/layouts.py                            |      104 |       68 |     35% |47-49, 53-60, 64, 67, 107-110, 118-144, 157, 160, 170-217, 226-227, 230, 234, 237, 246-247, 250, 254, 257, 282-283, 287-300, 304, 307 |
| sharktank/sharktank/types/layout\_utils.py                                    |      111 |        8 |     93% |90, 125, 127, 204-207, 250, 259 |
| sharktank/sharktank/types/layouts.py                                          |      281 |       46 |     84% |117, 160-161, 166-167, 192-199, 290-302, 400, 404, 480, 488, 496, 504, 517, 530, 538, 550, 553, 556-564, 567-577, 684, 700, 705 |
| sharktank/sharktank/types/misc.py                                             |       55 |        1 |     98% |       122 |
| sharktank/sharktank/types/ocp\_floats.py                                      |       98 |       19 |     81% |93-118, 163, 290 |
| sharktank/sharktank/types/pipelining.py                                       |       68 |        5 |     93% |90-91, 106, 168, 188 |
| sharktank/sharktank/types/quantizers.py                                       |      293 |       40 |     86% |131, 177-178, 181-182, 214, 253, 308-309, 318, 336, 342, 356, 362, 388, 390, 453, 455, 493-494, 551, 583, 585, 628, 651, 659, 689-690, 694, 706-716, 731, 743-750 |
| sharktank/sharktank/types/sharding.py                                         |      155 |       40 |     74% |33, 104-106, 109-150, 262-263, 266, 320, 363-366, 369-402, 407-408, 411, 450-451, 454, 466-475 |
| sharktank/sharktank/types/tensors.py                                          |      856 |      114 |     87% |73-78, 88-90, 161-167, 187-188, 193, 222, 230, 266, 294, 298, 339, 364, 382, 399-401, 411, 415, 423-424, 435-437, 451-453, 467-469, 477-479, 487-488, 518, 523-525, 535-537, 606, 668, 670, 722, 755-756, 761, 798, 802, 804, 839, 844, 851, 902-903, 926-927, 932-933, 1086, 1168, 1215-1216, 1234, 1236, 1238, 1246-1248, 1253, 1377, 1390, 1392-1397, 1399, 1401, 1412, 1415, 1417, 1427, 1430, 1459, 1521-1523, 1528, 1553, 1588-1589, 1602, 1617, 1619, 1621, 1690, 1701-1702, 1710-1711, 1719-1720, 1726-1729, 1755-1756, 1883-1884, 1898, 1904, 1908-1909 |
| sharktank/sharktank/types/theta.py                                            |      377 |       57 |     85% |71, 79, 108, 139-149, 169, 181-182, 211-212, 218, 230, 233, 346-350, 413, 474-475, 479-481, 491, 499-500, 504-506, 515-516, 540, 546-547, 561-562, 579-580, 604-605, 626-627, 650, 664-666, 672, 710, 758-760, 765 |
| sharktank/sharktank/utils/\_\_init\_\_.py                                     |        1 |        0 |    100% |           |
| sharktank/sharktank/utils/attention.py                                        |       16 |        5 |     69% |     70-77 |
| sharktank/sharktank/utils/azure.py                                            |       58 |       58 |      0% |     7-121 |
| sharktank/sharktank/utils/cli.py                                              |      118 |       73 |     38% |35-39, 73-190, 201-206, 217-229, 240-272, 287-308, 319, 332-335, 345, 358-370, 379, 395-396, 398, 401-402, 414, 419-422, 431-449 |
| sharktank/sharktank/utils/create\_cache.py                                    |       13 |        4 |     69% | 20, 31-33 |
| sharktank/sharktank/utils/debugging.py                                        |       91 |       29 |     68% |46-63, 67-74, 81-83, 127, 138 |
| sharktank/sharktank/utils/evaluate.py                                         |       59 |       34 |     42% |29-48, 55, 69-88, 119-120, 129-149 |
| sharktank/sharktank/utils/export.py                                           |       70 |        4 |     94% |140, 151, 179, 212 |
| sharktank/sharktank/utils/export\_artifacts.py                                |      177 |      130 |     27% |39-47, 61, 68, 75, 82, 89, 124-158, 169, 182-186, 213-233, 251-263, 267-273, 285-301, 325-367, 391-427, 451-464, 490-516, 538-544 |
| sharktank/sharktank/utils/functools.py                                        |       10 |        8 |     20% |     27-35 |
| sharktank/sharktank/utils/hf.py                                               |       78 |       56 |     28% |35-46, 50-52, 63-92, 103-120, 146-156, 160-161 |
| sharktank/sharktank/utils/hf\_datasets.py                                     |       75 |       22 |     71% |37-55, 65, 73, 82-83, 88, 478-496, 500 |
| sharktank/sharktank/utils/io.py                                               |       39 |        9 |     77% |65-72, 83-86 |
| sharktank/sharktank/utils/iree.py                                             |      287 |       56 |     80% |187, 198-201, 299, 303, 307, 313-320, 326, 332, 338, 380, 498-499, 546-547, 555-559, 660-681, 696-703, 715-723, 746 |
| sharktank/sharktank/utils/llm\_artifacts.py                                   |       31 |        2 |     94% |    37, 42 |
| sharktank/sharktank/utils/llm\_utils.py                                       |      367 |       88 |     76% |71-78, 100-101, 104-109, 190-192, 293-295, 323, 365, 396, 399-439, 452, 455-463, 488-491, 493-494, 496, 515, 519, 548-555, 558-571, 595-601, 619 |
| sharktank/sharktank/utils/load\_llm.py                                        |      173 |      136 |     21% |39-41, 45, 51-65, 70-82, 93-108, 120, 123, 138-162, 166, 173, 176-185, 190-197, 200-210, 220-238, 241-279, 282-352, 355-357 |
| sharktank/sharktank/utils/logging.py                                          |        6 |        1 |     83% |        17 |
| sharktank/sharktank/utils/math.py                                             |       12 |        5 |     58% | 17, 25-28 |
| sharktank/sharktank/utils/misc.py                                             |       58 |       11 |     81% |35, 100, 105-115 |
| sharktank/sharktank/utils/patching.py                                         |       94 |       43 |     54% |56, 75-78, 87-93, 98, 108-133, 141-154, 157-168, 197, 231, 233, 238 |
| sharktank/sharktank/utils/random.py                                           |       23 |        0 |    100% |           |
| sharktank/sharktank/utils/testing.py                                          |      431 |      203 |     53% |104-106, 157-268, 291-297, 308, 321-331, 344-368, 374-395, 411-420, 425-438, 442-446, 485-569, 604, 657-660, 697-703, 734, 758-766, 781, 786, 792-796, 804-807, 813-820, 828-832, 900, 941, 972, 988-997, 1022-1024, 1037, 1047, 1050, 1089 |
| sharktank/sharktank/utils/tokenizer.py                                        |       51 |       35 |     31% |34-38, 42-46, 50, 63-66, 69-72, 76, 80-81, 85-110 |
| sharktank/sharktank/utils/tree.py                                             |       71 |        2 |     97% |   81, 220 |
| sharktank/tests/evaluate/perplexity\_iree\_test.py                            |       46 |       28 |     39% |30-36, 43-55, 58-68, 78-83, 88-98, 102 |
| sharktank/tests/evaluate/perplexity\_torch\_test.py                           |       38 |       22 |     42% |29-34, 37-46, 49-65, 70-75, 79 |
| sharktank/tests/examples/main\_test.py                                        |       24 |        1 |     96% |        45 |
| sharktank/tests/examples/paged\_llm\_v1\_test.py                              |       16 |        5 |     69% |     29-33 |
| sharktank/tests/export\_ir/export\_test.py                                    |       37 |        0 |    100% |           |
| sharktank/tests/kernels/attention\_template\_test.py                          |       76 |        8 |     89% |23, 112-118, 138 |
| sharktank/tests/kernels/attention\_wave\_test.py                              |       23 |        2 |     91% |    25, 59 |
| sharktank/tests/kernels/batch\_matmul\_transpose\_b\_test.py                  |       94 |        6 |     94% |127-130, 143, 170 |
| sharktank/tests/kernels/conv\_2d\_nchw\_fchw\_test.py                         |       42 |        2 |     95% |    63, 91 |
| sharktank/tests/kernels/einsum\_q4\_test.py                                   |       69 |        3 |     96% |94, 120, 141 |
| sharktank/tests/kernels/gemm\_fp4\_asm\_test.py                               |       61 |       42 |     31% |26, 65-133 |
| sharktank/tests/kernels/mlir\_kernel\_test.py                                 |       21 |        0 |    100% |           |
| sharktank/tests/kernels/mmt\_block\_scaled\_offset\_q4\_test.py               |       46 |        3 |     93% |49, 79, 100 |
| sharktank/tests/kernels/mmt\_block\_scaled\_q8\_test.py                       |       43 |        3 |     93% |46, 74, 94 |
| sharktank/tests/kernels/mmt\_super\_block\_scaled\_offset\_q4\_test.py        |       71 |       20 |     72% |39-64, 97, 156, 174 |
| sharktank/tests/kernels/mmtfp\_test.py                                        |       60 |        4 |     93% |57, 81, 99, 125 |
| sharktank/tests/kernels/pooling\_nchw\_sum\_test.py                           |       42 |        2 |     95% |    58, 78 |
| sharktank/tests/kernels/rotary\_test.py                                       |       18 |        0 |    100% |           |
| sharktank/tests/kernels/topk\_test.py                                         |       31 |        0 |    100% |           |
| sharktank/tests/kernels/wave/extend\_attention\_test.py                       |       62 |       35 |     44% |35, 74-203 |
| sharktank/tests/kernels/wave/mxfp4\_gemm\_test.py                             |       64 |       39 |     39% |33, 65-136 |
| sharktank/tests/kernels/wave/wave\_utils\_test.py                             |       30 |        0 |    100% |           |
| sharktank/tests/layers/base\_test.py                                          |       22 |        0 |    100% |           |
| sharktank/tests/layers/configs\_test.py                                       |       14 |        0 |    100% |           |
| sharktank/tests/layers/kv\_cache\_test.py                                     |       85 |        0 |    100% |           |
| sharktank/tests/layers/linear\_test.py                                        |       92 |        1 |     99% |       219 |
| sharktank/tests/layers/mixture\_of\_experts\_block\_test.py                   |      130 |        0 |    100% |           |
| sharktank/tests/layers/mmdit\_test.py                                         |       56 |        1 |     98% |        96 |
| sharktank/tests/layers/paged\_llama\_attention\_block\_test.py                |      189 |       40 |     79% |56-73, 84-166, 466 |
| sharktank/tests/layers/rotary\_embedding\_hf\_test.py                         |      238 |       10 |     96% |305-306, 400-407 |
| sharktank/tests/layers/rotary\_embedding\_test.py                             |      131 |       46 |     65% |131-201, 219-227, 236-249 |
| sharktank/tests/layers/sharded\_conv2d\_with\_iree\_test.py                   |       79 |        0 |    100% |           |
| sharktank/tests/models/clip/clip\_test.py                                     |      251 |       53 |     79% |91, 96-111, 121, 131, 209-253, 298-325, 350-384, 393, 403 |
| sharktank/tests/models/deepseek/test\_deepseek.py                             |       30 |        3 |     90% |     68-79 |
| sharktank/tests/models/flux/flux\_test.py                                     |      166 |       76 |     54% |85-87, 91-92, 117, 162-217, 226-244, 253-273, 308, 316, 323-342, 352-381, 391, 400, 410-418, 422 |
| sharktank/tests/models/grok/test\_grok.py                                     |       25 |        0 |    100% |           |
| sharktank/tests/models/llama4/llama4\_test.py                                 |       46 |        1 |     98% |       100 |
| sharktank/tests/models/llama4/moe\_test.py                                    |       90 |        1 |     99% |       192 |
| sharktank/tests/models/llama/attention\_test.py                               |       71 |        1 |     99% |       203 |
| sharktank/tests/models/llama/benchmark\_amdgpu\_test.py                       |      110 |       69 |     37% |34, 37-48, 58-92, 95-105, 114-168, 193-208, 212-227, 231-250, 255-274, 281-351, 367-388, 392-410, 414 |
| sharktank/tests/models/llama/quantized\_theta\_test.py                        |       20 |        0 |    100% |           |
| sharktank/tests/models/llama/quark\_parity\_test.py                           |       55 |       40 |     27% |21-22, 29-101, 105 |
| sharktank/tests/models/llama/rot\_emb\_test.py                                |       37 |        1 |     97% |        81 |
| sharktank/tests/models/llama/test\_llama.py                                   |       44 |        7 |     84% |72-83, 88-98 |
| sharktank/tests/models/llama/toy\_llama\_test.py                              |       71 |        1 |     99% |        28 |
| sharktank/tests/models/punet/resnet\_test.py                                  |       42 |        1 |     98% |        93 |
| sharktank/tests/models/punet/sharded\_resnet\_block\_with\_iree\_test.py      |       43 |       12 |     72% |    76-113 |
| sharktank/tests/models/punet/up\_down\_block\_test.py                         |       49 |        1 |     98% |       149 |
| sharktank/tests/models/t5/t5\_test.py                                         |      269 |       59 |     78% |80-108, 146-174, 187-221, 266, 280, 289, 298, 307, 316, 325, 435-477, 522, 531, 540, 549, 558 |
| sharktank/tests/models/vae/vae\_test.py                                       |      226 |      125 |     45% |85-120, 126-135, 140-149, 155-250, 276-290, 295-308, 386-389, 396-487, 551-585, 588-595, 604-608, 613-618, 624 |
| sharktank/tests/ops/ops\_test.py                                              |      622 |       30 |     95% |177-180, 245-251, 258-264, 271-278, 632-637, 1077 |
| sharktank/tests/ops/pipeline\_parallelized\_test.py                           |      153 |        4 |     97% |57, 181, 193, 203 |
| sharktank/tests/ops/qconv\_test.py                                            |       97 |       12 |     88% |192-228, 232 |
| sharktank/tests/ops/quantized\_test.py                                        |       85 |        0 |    100% |           |
| sharktank/tests/ops/sharded\_test.py                                          |     1345 |       20 |     99% |596-602, 684, 1911, 1914, 1918, 1941, 1945, 2119, 2128-2130, 2138, 2165, 2364 |
| sharktank/tests/ops/test\_attention\_ops.py                                   |       28 |        1 |     96% |       106 |
| sharktank/tests/pipelines/flux/flux\_pipeline\_test.py                        |       41 |       23 |     44% |25-27, 32-65, 77-121, 128, 135 |
| sharktank/tests/pytest\_fixtures\_test.py                                     |       19 |        0 |    100% |           |
| sharktank/tests/tools/convert\_dataset\_test.py                               |       22 |        0 |    100% |           |
| sharktank/tests/tools/convert\_to\_json\_test.py                              |       18 |        0 |    100% |           |
| sharktank/tests/tools/sharktank\_test.py                                      |       19 |        0 |    100% |           |
| sharktank/tests/transforms/dataset\_transforms\_test.py                       |       34 |       25 |     26% | 23-84, 88 |
| sharktank/tests/types/dataset\_test.py                                        |      183 |       36 |     80% |239-259, 268-294, 303 |
| sharktank/tests/types/layout\_utils\_test.py                                  |       75 |        1 |     99% |       231 |
| sharktank/tests/types/layouts\_test.py                                        |       68 |        1 |     99% |       148 |
| sharktank/tests/types/quantizers\_test.py                                     |      266 |        1 |     99% |       634 |
| sharktank/tests/types/slice\_test.py                                          |       14 |        0 |    100% |           |
| sharktank/tests/types/tensors\_test.py                                        |      164 |        1 |     99% |       221 |
| sharktank/tests/utils/iree\_test.py                                           |       56 |        6 |     89% | 68-72, 92 |
| sharktank/tests/utils/misc\_test.py                                           |        9 |        0 |    100% |           |
| sharktank/tests/utils/patching\_test.py                                       |       44 |        0 |    100% |           |
| sharktank/tests/utils/testing\_test.py                                        |      132 |        0 |    100% |           |
| sharktank/tests/utils/tree\_test.py                                           |       20 |        0 |    100% |           |
|                                                                     **TOTAL** | **22084** | **4824** | **78%** |           |


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