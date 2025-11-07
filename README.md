# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/nod-ai/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                          |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| sharktank/conftest.py                                                         |      153 |       12 |     92% |297, 304, 311, 343, 354, 378, 383-386, 415, 442-443 |
| sharktank/integration/models/llama/llama\_integration\_test.py                |       32 |       22 |     31% |     42-86 |
| sharktank/integration/models/punet/integration\_test.py                       |       94 |       57 |     39% |15-16, 21-35, 56-70, 78-88, 98-113, 122-133, 143-155, 162, 167-179, 186, 197-204, 222-232, 254-267 |
| sharktank/setup.py                                                            |       18 |       18 |      0% |      7-34 |
| sharktank/sharktank/\_\_init\_\_.py                                           |        4 |        1 |     75% |        15 |
| sharktank/sharktank/build/\_\_init\_\_.py                                     |        1 |        1 |      0% |         7 |
| sharktank/sharktank/build/actions.py                                          |       45 |       45 |      0% |     7-109 |
| sharktank/sharktank/evaluate/perplexity\_iree.py                              |      248 |      208 |     16% |71-91, 95-101, 106-128, 136-172, 180-216, 226-252, 257-302, 307-356, 359-414, 421-469, 482-536, 543-592, 596 |
| sharktank/sharktank/evaluate/perplexity\_torch.py                             |      184 |      146 |     21% |54-56, 60-66, 71-93, 112-139, 143-160, 164-228, 242-281, 295-350, 374-397, 401-441, 445 |
| sharktank/sharktank/examples/export\_paged\_llm\_v1.py                        |      133 |       47 |     65% |44, 80-86, 96, 109-127, 234-322, 326 |
| sharktank/sharktank/examples/paged\_llm\_v1.py                                |       52 |       41 |     21% |38-132, 136 |
| sharktank/sharktank/examples/pipeline/export\_ppffn\_net.py                   |       66 |        4 |     94% |139, 145, 176, 183 |
| sharktank/sharktank/examples/sharding/export\_ffn\_net.py                     |       59 |       13 |     78% |51-63, 82, 88, 113, 120 |
| sharktank/sharktank/kernels/\_\_init\_\_.py                                   |       14 |        0 |    100% |           |
| sharktank/sharktank/kernels/attention.py                                      |       22 |        0 |    100% |           |
| sharktank/sharktank/kernels/base.py                                           |       52 |        5 |     90% |136, 155-160 |
| sharktank/sharktank/kernels/batch\_matmul\_transpose\_b.py                    |       49 |        0 |    100% |           |
| sharktank/sharktank/kernels/bitcast.py                                        |       63 |       40 |     37% |58-69, 75-88, 97-108, 114-127, 136-139 |
| sharktank/sharktank/kernels/conv\_2d\_nchw\_fchw.py                           |       64 |        0 |    100% |           |
| sharktank/sharktank/kernels/einsum\_2args\_q4.py                              |      122 |        2 |     98% |   69, 179 |
| sharktank/sharktank/kernels/gemm\_fp4.py                                      |       17 |        0 |    100% |           |
| sharktank/sharktank/kernels/gemm\_fp4\_asm.py                                 |       37 |       18 |     51% |29-45, 64-182, 195 |
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
| sharktank/sharktank/kernels/wave/templates/extend\_attention\_kernel.py       |      196 |      184 |      6% |    47-460 |
| sharktank/sharktank/kernels/wave/utils.py                                     |      140 |      114 |     19% |68-74, 82-175, 213-253, 274-311, 317-320 |
| sharktank/sharktank/layers/\_\_init\_\_.py                                    |       16 |        0 |    100% |           |
| sharktank/sharktank/layers/activations.py                                     |        3 |        0 |    100% |           |
| sharktank/sharktank/layers/base.py                                            |      177 |       27 |     85% |131, 206-209, 224, 242, 259-260, 269, 298, 366-374, 385-398, 400, 404-407, 411, 417, 424 |
| sharktank/sharktank/layers/causal\_llm.py                                     |       22 |        7 |     68% |     58-64 |
| sharktank/sharktank/layers/configs/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| sharktank/sharktank/layers/configs/config.py                                  |      170 |       15 |     91% |139, 196, 205-211, 219, 234, 248-254, 267, 269, 289, 313 |
| sharktank/sharktank/layers/configs/llm\_configs.py                            |      558 |      162 |     71% |44-46, 236, 238, 240, 242, 244, 246, 248, 254, 256, 258, 260, 268, 272, 276, 283, 285, 289, 291, 294, 296, 298, 302, 304, 306, 317, 320, 323-326, 329-349, 352-359, 364-387, 395-398, 404-407, 411-416, 430-431, 442-443, 454-455, 485, 516-520, 524, 530, 535, 553, 613, 645-646, 651, 705-719, 723-726, 734-766, 810, 816-824, 833-845, 884, 906, 930-934, 970-973, 977-981 |
| sharktank/sharktank/layers/conv.py                                            |      100 |       61 |     39% |48, 58, 61, 63, 80, 95-110, 113-143, 157-172, 175-205 |
| sharktank/sharktank/layers/ffn\_block.py                                      |       26 |        0 |    100% |           |
| sharktank/sharktank/layers/ffn\_moe\_block.py                                 |      105 |       29 |     72% |73-76, 79-87, 252-286, 292-295, 302-308 |
| sharktank/sharktank/layers/kv\_cache.py                                       |       16 |        0 |    100% |           |
| sharktank/sharktank/layers/latent\_attention\_block.py                        |       52 |        5 |     90% |42, 61, 66, 76, 96 |
| sharktank/sharktank/layers/linear.py                                          |       43 |        4 |     91% |55, 66, 74, 84 |
| sharktank/sharktank/layers/mixture\_of\_experts\_block.py                     |       86 |        6 |     93% |51, 55, 63, 98, 118, 227 |
| sharktank/sharktank/layers/mmdit.py                                           |      103 |        0 |    100% |           |
| sharktank/sharktank/layers/modulation.py                                      |       21 |        0 |    100% |           |
| sharktank/sharktank/layers/norm.py                                            |       37 |        0 |    100% |           |
| sharktank/sharktank/layers/paged\_attention.py                                |      272 |       15 |     94% |202, 356, 360-363, 367, 387, 389, 427, 767-772, 931 |
| sharktank/sharktank/layers/paged\_llama\_attention\_block.py                  |      167 |       26 |     84% |93-97, 121-122, 156, 183-203, 342-351, 383, 385, 387, 515-517 |
| sharktank/sharktank/layers/rotary\_embedding.py                               |       36 |        0 |    100% |           |
| sharktank/sharktank/layers/rotary\_embedding\_hf.py                           |      121 |        3 |     98% |104, 252-253 |
| sharktank/sharktank/layers/testing.py                                         |       67 |       21 |     69% |302, 372-384, 390, 434-473 |
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
| sharktank/sharktank/models/llama4/testing.py                                  |       41 |        1 |     98% |        18 |
| sharktank/sharktank/models/llama/testing.py                                   |       58 |        0 |    100% |           |
| sharktank/sharktank/models/llama/toy\_llama.py                                |       51 |        6 |     88% |156-162, 166 |
| sharktank/sharktank/models/llm/\_\_init\_\_.py                                |        1 |        0 |    100% |           |
| sharktank/sharktank/models/llm/config.py                                      |       43 |        4 |     91% |     39-42 |
| sharktank/sharktank/models/llm/export.py                                      |       78 |       20 |     74% |25-30, 36, 71-73, 81-85, 90-93, 119, 127-130, 157 |
| sharktank/sharktank/models/llm/llm.py                                         |       99 |        6 |     94% |178, 202, 231, 234, 369-370 |
| sharktank/sharktank/models/llm/testing.py                                     |       69 |       12 |     83% |   170-187 |
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
| sharktank/sharktank/ops/\_registry.py                                         |      218 |       15 |     93% |52, 142, 147, 277-280, 292, 329, 332-335, 348, 466, 488, 496, 543 |
| sharktank/sharktank/ops/attention\_impls.py                                   |      156 |       41 |     74% |49-51, 81, 154-156, 165, 182, 239, 255-302 |
| sharktank/sharktank/ops/cpu\_impls.py                                         |       20 |        1 |     95% |        43 |
| sharktank/sharktank/ops/custom\_impls.py                                      |      122 |       47 |     61% |66-70, 88, 104, 125-134, 151-170, 200-220, 229, 233-236, 259, 261, 263 |
| sharktank/sharktank/ops/default\_impls.py                                     |      615 |      116 |     81% |54, 147, 149, 181, 183, 185, 218, 220, 222, 246-253, 258-266, 271-277, 282-290, 295-302, 308-322, 354, 356, 370-371, 445, 466-487, 501, 511, 695, 706, 737, 748-750, 789, 895, 966, 971, 976, 982, 1050, 1145, 1149, 1216-1228, 1233, 1238, 1270 |
| sharktank/sharktank/ops/qconv\_impls.py                                       |      123 |       31 |     75% |47, 53, 67-71, 88, 94, 109, 137-142, 168-177, 229, 252, 270-285, 298, 303, 310 |
| sharktank/sharktank/ops/qlinear\_impls.py                                     |       91 |       16 |     82% |41, 72, 91, 95, 109-112, 123-124, 150-151, 170, 173, 196-198, 217 |
| sharktank/sharktank/ops/quantized\_impls.py                                   |      233 |       13 |     94% |81, 89, 91-97, 99-106, 117-118, 142, 255-257, 394 |
| sharktank/sharktank/ops/shape.py                                              |       28 |        1 |     96% |        84 |
| sharktank/sharktank/ops/sharded\_impls.py                                     |      925 |       85 |     91% |228, 466, 528-530, 537, 545, 560, 570-574, 584-589, 604-605, 693-702, 752-760, 904, 947, 999, 1013, 1015, 1018, 1023, 1026, 1092-1094, 1177-1181, 1194, 1211, 1220, 1228-1230, 1256, 1272, 1282, 1306, 1332, 1334, 1344, 1346, 1411, 1461, 1568, 1598, 1603, 1731, 1733, 1828, 1838-1848, 2029-2030, 2054, 2117, 2126-2131, 2143, 2147, 2187-2188, 2193-2194 |
| sharktank/sharktank/ops/signatures.py                                         |      360 |       43 |     88% |146, 163, 206, 231, 264, 283, 301, 316, 335, 353, 368, 408, 424, 430, 446, 459, 501-507, 528, 598, 606, 639, 659, 672, 697, 736, 757, 780, 811, 867, 904, 922, 938, 994, 1037, 1095, 1226, 1279, 1336 |
| sharktank/sharktank/ops/utils.py                                              |       93 |        5 |     95% |38, 43, 229, 234, 239 |
| sharktank/sharktank/pipelines/flux/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| sharktank/sharktank/pipelines/flux/flux\_pipeline.py                          |      138 |      109 |     21% |40-93, 121-133, 155-188, 210-228, 238, 244-246, 269-277, 295-317, 320, 339-368, 373-474, 478 |
| sharktank/sharktank/tools/convert\_dataset.py                                 |       27 |        1 |     96% |        51 |
| sharktank/sharktank/tools/e2e\_model\_test.py                                 |      275 |      255 |      7% |46-62, 75-541, 547-625, 638 |
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
| sharktank/sharktank/types/ocp\_floats.py                                      |       98 |       19 |     81% |93-118, 163, 292 |
| sharktank/sharktank/types/pipelining.py                                       |       66 |        2 |     97% |  163, 183 |
| sharktank/sharktank/types/quantizers.py                                       |      293 |       40 |     86% |131, 177-178, 181-182, 214, 253, 308-309, 318, 336, 342, 356, 362, 388, 390, 453, 455, 493-494, 551, 583, 585, 628, 651, 659, 689-690, 694, 706-716, 731, 743-750 |
| sharktank/sharktank/types/sharding.py                                         |      155 |       40 |     74% |33, 104-106, 109-150, 262-263, 266, 320, 363-366, 369-402, 407-408, 411, 450-451, 454, 466-475 |
| sharktank/sharktank/types/tensors.py                                          |      884 |      121 |     86% |73-78, 88-90, 161-167, 187-188, 193, 222, 230, 266, 294, 298, 339, 364, 382, 399-401, 409-411, 414, 419, 422, 426, 434-435, 451-453, 467-469, 483-485, 489, 492-494, 502-504, 512-513, 553, 570-572, 641, 687-689, 713, 715, 758, 765, 773, 806-807, 812, 849, 853, 855, 890, 895, 902, 953-954, 977-978, 983-984, 1112, 1140, 1165, 1167, 1169, 1171, 1236, 1283-1284, 1302-1304, 1309, 1433, 1446, 1448-1453, 1455, 1457, 1468, 1471, 1473, 1483, 1486, 1515, 1577-1579, 1584, 1644-1645, 1658, 1737, 1748-1749, 1757-1758, 1766-1767, 1773-1776, 1802-1803, 1930-1931, 1945, 1951, 1955-1956 |
| sharktank/sharktank/types/theta.py                                            |      377 |       57 |     85% |72, 80, 109, 140-150, 170, 182-183, 212-213, 219, 231, 234, 347-351, 414, 475-476, 480-482, 492, 500-501, 505-507, 516-517, 541, 547-548, 562-563, 580-581, 605-606, 627-628, 651, 665-667, 673, 711, 759-761, 766 |
| sharktank/sharktank/utils/\_\_init\_\_.py                                     |        1 |        0 |    100% |           |
| sharktank/sharktank/utils/attention.py                                        |       55 |       11 |     80% |124-131, 166-182 |
| sharktank/sharktank/utils/azure.py                                            |       58 |       58 |      0% |     7-122 |
| sharktank/sharktank/utils/cli.py                                              |      118 |       73 |     38% |35-39, 73-190, 201-206, 217-229, 240-272, 287-308, 319, 332-335, 345, 358-370, 379, 395-396, 398, 401-402, 414, 419-422, 431-449 |
| sharktank/sharktank/utils/create\_cache.py                                    |       13 |        4 |     69% | 20, 31-33 |
| sharktank/sharktank/utils/debugging.py                                        |       91 |       29 |     68% |46-63, 67-74, 81-83, 127, 138 |
| sharktank/sharktank/utils/e2e\_test\_utils.py                                 |       89 |       71 |     20% |32-40, 45-49, 52-70, 75-119, 122-126, 129-133 |
| sharktank/sharktank/utils/evaluate.py                                         |       59 |       34 |     42% |29-50, 57, 71-90, 121-122, 131-151 |
| sharktank/sharktank/utils/export.py                                           |       70 |        4 |     94% |140, 151, 179, 212 |
| sharktank/sharktank/utils/export\_artifacts.py                                |      177 |      130 |     27% |38-46, 60, 67, 74, 81, 88, 122-156, 167, 180-184, 210-230, 248-260, 264-270, 282-298, 322-364, 388-425, 449-462, 488-514, 536-542 |
| sharktank/sharktank/utils/functools.py                                        |       10 |        8 |     20% |     27-35 |
| sharktank/sharktank/utils/hf.py                                               |      109 |       74 |     32% |36-47, 51-53, 65-100, 115-157, 215, 237-247, 251-252 |
| sharktank/sharktank/utils/hf\_datasets.py                                     |       92 |       26 |     72% |40-43, 46-65, 68, 88, 103, 112-113, 118, 544-562, 566 |
| sharktank/sharktank/utils/io.py                                               |       39 |        9 |     77% |65-72, 83-86 |
| sharktank/sharktank/utils/iree.py                                             |      287 |       56 |     80% |187, 198-201, 299, 303, 307, 313-320, 326, 332, 338, 380, 498-499, 546-547, 555-559, 660-681, 696-703, 715-723, 746 |
| sharktank/sharktank/utils/llm\_artifacts.py                                   |       31 |        2 |     94% |    37, 42 |
| sharktank/sharktank/utils/llm\_scheduler.py                                   |       86 |        4 |     95% |34, 38, 42, 52 |
| sharktank/sharktank/utils/llm\_tasks.py                                       |      139 |        3 |     98% |57, 63, 68 |
| sharktank/sharktank/utils/llm\_utils.py                                       |      436 |       95 |     78% |102-116, 138-139, 142-145, 149-154, 205, 239-241, 590-591, 597, 606-607, 628, 631-671, 684, 687-695, 722-725, 727-728, 730, 749, 753, 788-795, 798-811, 839-845, 867 |
| sharktank/sharktank/utils/load\_llm.py                                        |      173 |      136 |     21% |39-41, 45, 51-65, 70-82, 93-108, 120, 123, 138-162, 166, 173, 176-185, 190-197, 200-210, 220-238, 241-279, 282-352, 355-357 |
| sharktank/sharktank/utils/logging.py                                          |        6 |        1 |     83% |        17 |
| sharktank/sharktank/utils/math.py                                             |       12 |        5 |     58% | 17, 25-28 |
| sharktank/sharktank/utils/misc.py                                             |       58 |        9 |     84% |35, 100, 108-114 |
| sharktank/sharktank/utils/patching.py                                         |      137 |       46 |     66% |138, 141, 166, 203-206, 215-221, 226, 236-261, 269-285, 288-299, 328, 362, 364 |
| sharktank/sharktank/utils/random.py                                           |       38 |        0 |    100% |           |
| sharktank/sharktank/utils/testing.py                                          |      444 |      207 |     53% |104-106, 157-268, 291-297, 308, 321-331, 344-368, 374-395, 411-420, 425-438, 442-446, 485-569, 604, 657-660, 697-703, 734, 758-766, 781, 786, 792-796, 804-807, 813-820, 828-832, 900, 949, 954-968, 1009, 1045-1047, 1058, 1064, 1074, 1077, 1119 |
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
| sharktank/tests/kernels/gemm\_fp4\_asm\_test.py                               |       57 |       38 |     33% |26, 61-120 |
| sharktank/tests/kernels/gemm\_fp4\_test.py                                    |      112 |       40 |     64% |59, 64, 135, 138, 161-222 |
| sharktank/tests/kernels/mlir\_kernel\_test.py                                 |       21 |        0 |    100% |           |
| sharktank/tests/kernels/mmt\_block\_scaled\_offset\_q4\_test.py               |       46 |        3 |     93% |49, 79, 100 |
| sharktank/tests/kernels/mmt\_block\_scaled\_q8\_test.py                       |       43 |        3 |     93% |46, 74, 94 |
| sharktank/tests/kernels/mmt\_super\_block\_scaled\_offset\_q4\_test.py        |       71 |       20 |     72% |39-64, 97, 156, 174 |
| sharktank/tests/kernels/mmtfp\_test.py                                        |       60 |        4 |     93% |57, 81, 99, 125 |
| sharktank/tests/kernels/pooling\_nchw\_sum\_test.py                           |       42 |        2 |     95% |    58, 78 |
| sharktank/tests/kernels/rotary\_test.py                                       |       18 |        0 |    100% |           |
| sharktank/tests/kernels/topk\_test.py                                         |       31 |        0 |    100% |           |
| sharktank/tests/kernels/wave/extend\_attention\_test.py                       |       90 |       56 |     38% |41, 80-209, 235-267 |
| sharktank/tests/kernels/wave/mxfp4\_gemm\_test.py                             |       64 |       39 |     39% |33, 66-137 |
| sharktank/tests/kernels/wave/wave\_utils\_test.py                             |       30 |        0 |    100% |           |
| sharktank/tests/layers/base\_test.py                                          |       22 |        0 |    100% |           |
| sharktank/tests/layers/configs\_test.py                                       |       14 |        0 |    100% |           |
| sharktank/tests/layers/kv\_cache\_test.py                                     |       85 |        0 |    100% |           |
| sharktank/tests/layers/linear\_test.py                                        |       92 |        1 |     99% |       219 |
| sharktank/tests/layers/mixture\_of\_experts\_block\_test.py                   |      286 |        1 |     99% |       849 |
| sharktank/tests/layers/mmdit\_test.py                                         |       56 |        1 |     98% |        96 |
| sharktank/tests/layers/paged\_llama\_attention\_block\_test.py                |      187 |       40 |     79% |54-71, 82-164, 464 |
| sharktank/tests/layers/rotary\_embedding\_hf\_test.py                         |      236 |       10 |     96% |303-304, 398-405 |
| sharktank/tests/layers/rotary\_embedding\_test.py                             |      131 |       46 |     65% |129-199, 217-225, 229-242 |
| sharktank/tests/layers/sharded\_conv2d\_with\_iree\_test.py                   |       79 |        0 |    100% |           |
| sharktank/tests/models/clip/clip\_test.py                                     |      251 |       53 |     79% |91, 96-111, 121, 131, 209-253, 298-325, 350-384, 393, 403 |
| sharktank/tests/models/deepseek/test\_deepseek.py                             |       31 |        3 |     90% |     69-80 |
| sharktank/tests/models/flux/flux\_test.py                                     |      166 |       76 |     54% |85-87, 91-92, 117, 162-217, 226-244, 253-273, 308, 316, 323-342, 352-381, 391, 400, 410-418, 422 |
| sharktank/tests/models/grok/test\_grok.py                                     |       25 |        0 |    100% |           |
| sharktank/tests/models/llama4/llama4\_test.py                                 |       41 |        1 |     98% |        98 |
| sharktank/tests/models/llama4/moe\_test.py                                    |       92 |        1 |     99% |       203 |
| sharktank/tests/models/llama/attention\_test.py                               |       63 |        1 |     98% |       194 |
| sharktank/tests/models/llama/benchmark\_amdgpu\_test.py                       |      110 |       69 |     37% |34, 37-49, 59-93, 96-106, 115-169, 194-209, 213-228, 232-251, 256-275, 282-352, 368-389, 393-411, 415 |
| sharktank/tests/models/llama/quantized\_theta\_test.py                        |       20 |        0 |    100% |           |
| sharktank/tests/models/llama/quark\_parity\_test.py                           |       55 |       40 |     27% |21-22, 29-101, 105 |
| sharktank/tests/models/llama/rot\_emb\_test.py                                |       37 |        1 |     97% |        81 |
| sharktank/tests/models/llama/test\_llama.py                                   |      114 |       16 |     86% |187-198, 203-213, 218-231 |
| sharktank/tests/models/llama/toy\_llama\_test.py                              |       85 |        1 |     99% |        41 |
| sharktank/tests/models/punet/resnet\_test.py                                  |       42 |        1 |     98% |        93 |
| sharktank/tests/models/punet/sharded\_resnet\_block\_with\_iree\_test.py      |       43 |       12 |     72% |    76-113 |
| sharktank/tests/models/punet/up\_down\_block\_test.py                         |       49 |        1 |     98% |       149 |
| sharktank/tests/models/t5/t5\_test.py                                         |      269 |       59 |     78% |80-108, 146-174, 187-221, 266, 280, 289, 298, 307, 316, 325, 435-477, 522, 531, 540, 549, 558 |
| sharktank/tests/models/vae/vae\_test.py                                       |      226 |      125 |     45% |85-120, 126-135, 140-149, 155-250, 276-290, 295-308, 386-389, 396-487, 551-585, 588-595, 604-608, 613-618, 624 |
| sharktank/tests/ops/ops\_test.py                                              |      854 |       30 |     96% |268-271, 386-392, 399-405, 412-419, 797-802, 1392 |
| sharktank/tests/ops/pipeline\_parallelized\_test.py                           |      153 |        4 |     97% |57, 181, 193, 203 |
| sharktank/tests/ops/qconv\_test.py                                            |       97 |       12 |     88% |192-228, 232 |
| sharktank/tests/ops/quantized\_test.py                                        |      103 |        0 |    100% |           |
| sharktank/tests/ops/shaping/expand\_op\_test.py                               |       17 |        1 |     94% |        63 |
| sharktank/tests/ops/shaping/flatten\_op\_test.py                              |       17 |        1 |     94% |        61 |
| sharktank/tests/ops/shaping/permute\_op\_test.py                              |       17 |        1 |     94% |        59 |
| sharktank/tests/ops/shaping/reshape\_op\_test.py                              |       17 |        1 |     94% |        63 |
| sharktank/tests/ops/shaping/squeeze\_op\_test.py                              |       17 |        1 |     94% |        63 |
| sharktank/tests/ops/shaping/transpose\_op\_test.py                            |       27 |        1 |     96% |        82 |
| sharktank/tests/ops/shaping/unflatten\_op\_test.py                            |       17 |        1 |     94% |        61 |
| sharktank/tests/ops/shaping/unsqueeze\_op\_test.py                            |       17 |        1 |     94% |        64 |
| sharktank/tests/ops/shaping/view\_op\_test.py                                 |       25 |        1 |     96% |        94 |
| sharktank/tests/ops/sharded\_test.py                                          |     1403 |       20 |     99% |605-611, 693, 1978, 1981, 1985, 2008, 2012, 2186, 2195-2197, 2205, 2232, 2450 |
| sharktank/tests/ops/test\_attention\_ops.py                                   |       28 |        1 |     96% |       106 |
| sharktank/tests/pipelines/flux/flux\_pipeline\_test.py                        |       41 |       23 |     44% |25-27, 32-65, 77-121, 128, 135 |
| sharktank/tests/pytest\_fixtures\_test.py                                     |       19 |        0 |    100% |           |
| sharktank/tests/tools/convert\_dataset\_test.py                               |       22 |        0 |    100% |           |
| sharktank/tests/tools/convert\_to\_json\_test.py                              |       18 |        0 |    100% |           |
| sharktank/tests/tools/list\_sfaetensors\_test.py                              |       13 |        0 |    100% |           |
| sharktank/tests/tools/sharktank\_test.py                                      |       19 |        0 |    100% |           |
| sharktank/tests/transforms/dataset\_transforms\_test.py                       |       34 |       25 |     26% | 23-84, 88 |
| sharktank/tests/types/dataset\_test.py                                        |      182 |       36 |     80% |238-258, 267-293, 302 |
| sharktank/tests/types/layout\_utils\_test.py                                  |       75 |        1 |     99% |       231 |
| sharktank/tests/types/layouts\_test.py                                        |       68 |        1 |     99% |       148 |
| sharktank/tests/types/quantizers\_test.py                                     |      266 |        1 |     99% |       634 |
| sharktank/tests/types/slice\_test.py                                          |       14 |        0 |    100% |           |
| sharktank/tests/types/tensors\_test.py                                        |      164 |        1 |     99% |       221 |
| sharktank/tests/utils/iree\_test.py                                           |       56 |        6 |     89% | 68-72, 92 |
| sharktank/tests/utils/llm\_basic\_scheduler\_test.py                          |       60 |        3 |     95% |31, 44, 109 |
| sharktank/tests/utils/llm\_chunk\_scheduler\_test.py                          |       63 |        3 |     95% |33, 46, 109 |
| sharktank/tests/utils/llm\_decode\_tasks\_test.py                             |       74 |        0 |    100% |           |
| sharktank/tests/utils/llm\_prefill\_tasks\_test.py                            |      140 |        0 |    100% |           |
| sharktank/tests/utils/llm\_utils\_test.py                                     |       85 |        0 |    100% |           |
| sharktank/tests/utils/misc\_test.py                                           |        9 |        0 |    100% |           |
| sharktank/tests/utils/patching\_test.py                                       |       81 |        0 |    100% |           |
| sharktank/tests/utils/random\_test.py                                         |       23 |        0 |    100% |           |
| sharktank/tests/utils/testing\_test.py                                        |      132 |        0 |    100% |           |
| sharktank/tests/utils/tree\_test.py                                           |       20 |        0 |    100% |           |
|                                                                     **TOTAL** | **24862** | **5568** | **78%** |           |


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