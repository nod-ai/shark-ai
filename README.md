# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/nod-ai/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                          |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| sharktank/conftest.py                                                         |      135 |       10 |     93% |342, 347, 354, 361, 429, 447, 452-455, 464 |
| sharktank/integration/models/punet/integration\_test.py                       |       94 |       57 |     39% |15-16, 21-31, 52-62, 70-80, 90-101, 110-121, 131-143, 150, 155-167, 174, 185-192, 210-220, 242-255 |
| sharktank/setup.py                                                            |       18 |       18 |      0% |      7-34 |
| sharktank/sharktank/\_\_init\_\_.py                                           |        4 |        1 |     75% |        15 |
| sharktank/sharktank/build/\_\_init\_\_.py                                     |        1 |        1 |      0% |         7 |
| sharktank/sharktank/build/actions.py                                          |       45 |       45 |      0% |     7-109 |
| sharktank/sharktank/evaluate/perplexity\_iree.py                              |      260 |      216 |     17% |68-86, 89-101, 105-120, 125-147, 157-183, 189-226, 236-261, 267-312, 317-366, 370-418, 425-471, 484-538, 545-594, 598 |
| sharktank/sharktank/evaluate/perplexity\_torch.py                             |      193 |      154 |     20% |53-54, 57-69, 73-88, 93-115, 134-160, 164-181, 186-250, 264-301, 315-371, 396-421, 425-465, 469 |
| sharktank/sharktank/examples/export\_paged\_llm\_v1.py                        |      204 |       79 |     61% |42, 47, 50, 60-65, 175, 244-246, 252-253, 270, 301, 304, 307-311, 317-320, 329-466, 485-491, 506, 520-521, 529-530 |
| sharktank/sharktank/examples/pipeline/export\_ppffn\_net.py                   |       83 |        5 |     94% |95, 149, 155, 179, 186 |
| sharktank/sharktank/examples/sharding/export\_ffn\_net.py                     |       59 |       13 |     78% |51-63, 82, 88, 113, 120 |
| sharktank/sharktank/examples/sharding/shard\_llm\_dataset.py                  |       23 |        3 |     87% |33, 36, 51 |
| sharktank/sharktank/kernels/\_\_init\_\_.py                                   |       13 |        0 |    100% |           |
| sharktank/sharktank/kernels/attention.py                                      |       22 |        2 |     91% |     40-75 |
| sharktank/sharktank/kernels/base.py                                           |       52 |        5 |     90% |136, 155-160 |
| sharktank/sharktank/kernels/batch\_matmul\_transpose\_b.py                    |       49 |        0 |    100% |           |
| sharktank/sharktank/kernels/bitcast.py                                        |       63 |       40 |     37% |58-69, 75-88, 97-108, 114-127, 136-139 |
| sharktank/sharktank/kernels/conv\_2d\_nchw\_fchw.py                           |       64 |        0 |    100% |           |
| sharktank/sharktank/kernels/einsum\_2args\_q4.py                              |      122 |        2 |     98% |   69, 179 |
| sharktank/sharktank/kernels/mlir\_kernel.py                                   |      204 |       20 |     90% |40, 43, 46-50, 112, 123, 129, 131, 220, 262, 269, 277, 321, 329, 369-374, 382 |
| sharktank/sharktank/kernels/mmt\_block\_scaled\_offset\_q4.py                 |       50 |        3 |     94% |     94-96 |
| sharktank/sharktank/kernels/mmt\_block\_scaled\_q8.py                         |       38 |        0 |    100% |           |
| sharktank/sharktank/kernels/mmt\_super\_block\_scaled\_offset\_q4.py          |       59 |        0 |    100% |           |
| sharktank/sharktank/kernels/mmtfp.py                                          |       41 |        2 |     95% |     68-69 |
| sharktank/sharktank/kernels/pooling\_nchw\_sum.py                             |       38 |        0 |    100% |           |
| sharktank/sharktank/kernels/rotary.py                                         |       31 |        0 |    100% |           |
| sharktank/sharktank/kernels/topk.py                                           |       30 |        0 |    100% |           |
| sharktank/sharktank/layers/\_\_init\_\_.py                                    |       15 |        0 |    100% |           |
| sharktank/sharktank/layers/activations.py                                     |        3 |        0 |    100% |           |
| sharktank/sharktank/layers/base.py                                            |      177 |       27 |     85% |131, 206-209, 224, 242, 259-260, 269, 298, 366-374, 385-398, 400, 404-407, 411, 417, 424 |
| sharktank/sharktank/layers/causal\_llm.py                                     |       69 |        5 |     93% |57-62, 132 |
| sharktank/sharktank/layers/configs/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| sharktank/sharktank/layers/configs/config.py                                  |      172 |       16 |     91% |17, 139, 196, 205-211, 219, 234, 248-254, 267, 269, 289, 313 |
| sharktank/sharktank/layers/configs/llm\_configs.py                            |      391 |       66 |     83% |28-29, 210, 214, 218, 230, 265-275, 283-286, 292-295, 301-304, 312, 315-316, 327-328, 429-430, 435, 517, 523-531, 540-552, 591, 613, 637-641, 677-680, 684-688 |
| sharktank/sharktank/layers/conv.py                                            |      100 |       61 |     39% |48, 58, 61, 63, 80, 95-110, 113-143, 157-172, 175-205 |
| sharktank/sharktank/layers/ffn\_block.py                                      |       26 |        0 |    100% |           |
| sharktank/sharktank/layers/ffn\_moe\_block.py                                 |       83 |       25 |     70% |65-73, 203-237, 243-246, 253-259 |
| sharktank/sharktank/layers/latent\_attention\_block.py                        |       55 |        4 |     93% |42, 62, 85-86 |
| sharktank/sharktank/layers/linear.py                                          |       42 |        4 |     90% |56, 67, 75, 83 |
| sharktank/sharktank/layers/mixture\_of\_experts\_block.py                     |       71 |        5 |     93% |48, 52, 60, 105-109 |
| sharktank/sharktank/layers/mmdit.py                                           |      102 |        0 |    100% |           |
| sharktank/sharktank/layers/modulation.py                                      |       21 |        0 |    100% |           |
| sharktank/sharktank/layers/norm.py                                            |       37 |        0 |    100% |           |
| sharktank/sharktank/layers/paged\_attention.py                                |      445 |       22 |     95% |136, 143-149, 210, 678, 998, 1008-1009, 1016, 1019, 1023, 1025, 1027, 1046, 1052-1057, 1062-1068, 1072 |
| sharktank/sharktank/layers/paged\_llama\_attention\_block.py                  |      122 |       11 |     91% |116, 120-121, 134, 171, 173, 175, 239, 255-259 |
| sharktank/sharktank/layers/rotary\_embedding.py                               |      146 |       18 |     88% |108, 124-131, 144-147, 200-201, 241-249 |
| sharktank/sharktank/layers/testing.py                                         |       44 |        1 |     98% |       299 |
| sharktank/sharktank/layers/token\_embedding.py                                |       12 |        0 |    100% |           |
| sharktank/sharktank/models/\_\_init\_\_.py                                    |        7 |        0 |    100% |           |
| sharktank/sharktank/models/clip/\_\_init\_\_.py                               |        2 |        0 |    100% |           |
| sharktank/sharktank/models/clip/clip.py                                       |      206 |       31 |     85% |80, 123, 131, 143, 159-162, 171, 249, 326, 337, 340, 343, 397, 412, 439, 454, 487, 490, 493, 544-557, 568-570 |
| sharktank/sharktank/models/clip/export.py                                     |       27 |       10 |     63% |40-43, 51-59 |
| sharktank/sharktank/models/clip/export\_toy\_text\_model\_iree\_test\_data.py |       11 |        1 |     91% |        29 |
| sharktank/sharktank/models/clip/testing.py                                    |       67 |        4 |     94% |   175-179 |
| sharktank/sharktank/models/deepseek/testing.py                                |       22 |        0 |    100% |           |
| sharktank/sharktank/models/deepseek/toy\_deepseek.py                          |       35 |        9 |     74% | 83-93, 97 |
| sharktank/sharktank/models/dummy/\_\_init\_\_.py                              |        1 |        0 |    100% |           |
| sharktank/sharktank/models/dummy/dummy.py                                     |       40 |        0 |    100% |           |
| sharktank/sharktank/models/flux/\_\_init\_\_.py                               |        1 |        0 |    100% |           |
| sharktank/sharktank/models/flux/compile.py                                    |        1 |        0 |    100% |           |
| sharktank/sharktank/models/flux/export.py                                     |       55 |       24 |     56% |35-36, 56, 80, 95-98, 104-127 |
| sharktank/sharktank/models/flux/export\_flux\_transformer\_mlir.py            |       13 |       13 |      0% |      7-38 |
| sharktank/sharktank/models/flux/flux.py                                       |      233 |       29 |     88% |82-91, 117-121, 129, 135, 137, 142, 147, 152, 218, 222, 235, 242, 268-279, 288, 407 |
| sharktank/sharktank/models/flux/testing.py                                    |       53 |       10 |     81% |31, 154, 209-227 |
| sharktank/sharktank/models/grok/testing.py                                    |       22 |        0 |    100% |           |
| sharktank/sharktank/models/grok/toy\_grok.py                                  |       31 |        6 |     81% | 66-71, 75 |
| sharktank/sharktank/models/llama4/testing.py                                  |       42 |        1 |     98% |        17 |
| sharktank/sharktank/models/llama/testing.py                                   |       37 |        0 |    100% |           |
| sharktank/sharktank/models/llama/toy\_llama.py                                |       29 |        6 |     79% | 60-66, 70 |
| sharktank/sharktank/models/llm/\_\_init\_\_.py                                |        1 |        0 |    100% |           |
| sharktank/sharktank/models/llm/llm.py                                         |      136 |        4 |     97% |212, 272, 297, 300 |
| sharktank/sharktank/models/llm/testing.py                                     |       13 |       13 |      0% |      1-32 |
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
| sharktank/sharktank/ops/\_\_init\_\_.py                                       |       11 |        0 |    100% |           |
| sharktank/sharktank/ops/\_registry.py                                         |      144 |       12 |     92% |108, 113, 160-167, 229-232, 243, 278, 297 |
| sharktank/sharktank/ops/attention\_impls.py                                   |       48 |       29 |     40% |32-38, 42-53, 58-79, 86, 97 |
| sharktank/sharktank/ops/custom\_impls.py                                      |       68 |       15 |     78% |56-60, 78, 94, 112, 116-119, 142, 144, 146 |
| sharktank/sharktank/ops/default\_impls.py                                     |      469 |       97 |     79% |112, 114, 146, 148, 150, 183, 185, 187, 250-253, 285, 287, 301-302, 317-324, 338-345, 359, 364-382, 396, 406, 562, 581, 592-594, 633, 683, 710-721, 731, 736, 741, 747, 779-786, 792, 881, 885, 926-943, 948, 953 |
| sharktank/sharktank/ops/qconv\_impls.py                                       |      123 |       31 |     75% |47, 53, 67-71, 88, 94, 109, 137-142, 168-177, 229, 252, 270-285, 298, 303, 310 |
| sharktank/sharktank/ops/qlinear\_impls.py                                     |       91 |       16 |     82% |40, 65, 84, 88, 102-105, 116-117, 143-144, 162, 165, 188-190, 209 |
| sharktank/sharktank/ops/shape.py                                              |       23 |        1 |     96% |        76 |
| sharktank/sharktank/ops/sharded\_impls.py                                     |      824 |       59 |     93% |220, 442, 484-486, 493, 501, 516, 541, 560-561, 631-640, 690-698, 791, 838, 860, 873, 919, 932, 935, 940, 943, 1002-1004, 1064, 1081, 1090, 1099, 1126, 1140, 1150, 1174, 1201, 1203, 1213, 1215, 1280, 1436, 1645, 1652, 1771-1772, 1796, 1859, 1872, 1882, 1886, 1926-1927, 1932-1933 |
| sharktank/sharktank/ops/signatures.py                                         |      724 |      146 |     80% |101, 114, 120, 131, 142, 159, 164, 176, 195, 228, 247, 265, 280, 299, 317, 332, 350, 363, 369, 385, 398, 414, 428, 442-455, 467, 483-489, 495, 508, 523, 528-534, 540, 553, 559, 575, 616, 638, 646, 665, 673, 690, 698, 715, 721, 734, 748, 776, 789, 814, 840, 858, 864, 880, 897, 918, 937, 945, 956, 968, 986, 994, 1001-1007, 1013, 1026, 1034, 1052, 1058, 1071, 1081, 1102, 1115, 1133, 1137, 1146, 1157, 1171, 1182, 1193, 1215, 1223, 1239, 1255, 1274, 1284, 1301, 1310, 1321, 1327, 1340, 1351, 1364, 1370, 1381, 1389, 1405, 1413, 1429, 1435, 1446, 1471, 1483, 1489, 1502, 1508, 1521, 1527, 1540, 1546, 1559, 1565, 1576, 1582, 1595, 1601, 1614, 1626, 1644, 1658, 1697, 1703, 1716, 1722, 1733, 1739, 1750, 1764, 1791 |
| sharktank/sharktank/ops/utils.py                                              |       47 |        1 |     98% |        53 |
| sharktank/sharktank/pipelines/flux/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| sharktank/sharktank/pipelines/flux/flux\_pipeline.py                          |      137 |      109 |     20% |39-92, 120-132, 154-187, 209-227, 237, 243-245, 268-276, 294-316, 319, 338-367, 372-473, 477 |
| sharktank/sharktank/tools/import\_hf\_dataset.py                              |       16 |       10 |     38% | 33-54, 60 |
| sharktank/sharktank/tools/sharktank.py                                        |       37 |        3 |     92% |60, 65, 83 |
| sharktank/sharktank/transforms/dataset/\_\_init\_\_.py                        |        2 |        0 |    100% |           |
| sharktank/sharktank/transforms/dataset/dataset.py                             |        7 |        1 |     86% |        23 |
| sharktank/sharktank/transforms/dataset/sharding.py                            |       38 |       28 |     26% |32-34, 37-49, 54-68, 71 |
| sharktank/sharktank/types/\_\_init\_\_.py                                     |        5 |        0 |    100% |           |
| sharktank/sharktank/types/gguf\_interop/\_\_init\_\_.py                       |        2 |        0 |    100% |           |
| sharktank/sharktank/types/gguf\_interop/base.py                               |       70 |       50 |     29% |42-44, 48-61, 65-81, 99-104, 115-138, 142-163, 167-168 |
| sharktank/sharktank/types/gguf\_interop/layouts.py                            |      104 |       68 |     35% |47-49, 53-60, 64, 67, 107-110, 118-144, 157, 160, 170-217, 226-227, 230, 234, 237, 246-247, 250, 254, 257, 282-283, 287-300, 304, 307 |
| sharktank/sharktank/types/layout\_utils.py                                    |       83 |        3 |     96% |   144-147 |
| sharktank/sharktank/types/layouts.py                                          |      214 |       45 |     79% |103, 146-147, 152-153, 158-159, 184-191, 275, 278-290, 353, 433, 441, 449, 457, 470, 483, 491, 503, 506, 509-517, 520-530 |
| sharktank/sharktank/types/pipelining.py                                       |       35 |        0 |    100% |           |
| sharktank/sharktank/types/quantizers.py                                       |      202 |       38 |     81% |64-72, 83, 137, 229, 260-261, 264-265, 281-287, 298, 318-319, 330, 408-409, 418, 433, 439, 466-467, 471, 483-493 |
| sharktank/sharktank/types/sharding.py                                         |      158 |        8 |     95% |18, 34, 100, 137-148, 321 |
| sharktank/sharktank/types/tensors.py                                          |      808 |      129 |     84% |71-76, 86, 92, 102, 112, 149-155, 175-176, 181, 210, 218, 239, 244, 249, 271, 275, 289, 300, 311, 332-333, 339, 346, 375, 379, 387-388, 399-401, 415-417, 441-443, 451-452, 482, 487-489, 494-496, 499-501, 593-595, 599, 601, 641, 653, 685-686, 691, 725, 729, 731, 755, 765, 774, 828-829, 852-853, 858-859, 955, 963, 1034, 1054-1057, 1087-1088, 1106, 1108, 1110, 1115-1117, 1122, 1245, 1258, 1260-1265, 1267, 1269, 1280, 1283, 1285, 1295, 1298, 1327, 1386-1388, 1393, 1401, 1416-1419, 1445-1446, 1474, 1476, 1478, 1547, 1558-1559, 1567-1568, 1576-1577, 1583-1586, 1612-1613, 1735-1736, 1756, 1760-1761 |
| sharktank/sharktank/types/theta.py                                            |      310 |       45 |     85% |69, 77, 106, 110, 137-147, 167, 179-180, 209-210, 216, 231, 341-345, 405, 453-454, 477-478, 493-494, 514-515, 529-530, 547-548, 590-592, 596, 634-636, 641 |
| sharktank/sharktank/utils/\_\_init\_\_.py                                     |        1 |        0 |    100% |           |
| sharktank/sharktank/utils/azure.py                                            |       58 |       58 |      0% |     7-121 |
| sharktank/sharktank/utils/cli.py                                              |      104 |       29 |     72% |178, 188-200, 248-269, 293-296, 319-325, 342-343, 345, 359, 364, 373-391 |
| sharktank/sharktank/utils/create\_cache.py                                    |        7 |        1 |     86% |        12 |
| sharktank/sharktank/utils/debugging.py                                        |       91 |       29 |     68% |46-63, 67-74, 81-83, 127, 138 |
| sharktank/sharktank/utils/evaluate.py                                         |       60 |       33 |     45% |32-51, 55, 69-88, 121-140 |
| sharktank/sharktank/utils/export.py                                           |       70 |        2 |     97% |  179, 212 |
| sharktank/sharktank/utils/export\_artifacts.py                                |      157 |       47 |     70% |17, 56, 63, 77, 84, 157-176, 205, 207, 226-238, 273, 277, 279, 312, 317, 353-368, 386-399, 410-412, 416-453 |
| sharktank/sharktank/utils/hf.py                                               |       35 |       24 |     31% |26-54, 64-78 |
| sharktank/sharktank/utils/hf\_datasets.py                                     |       75 |       22 |     71% |37-55, 65, 73, 82-83, 88, 478-496, 500 |
| sharktank/sharktank/utils/io.py                                               |       39 |        9 |     77% |65-72, 83-86 |
| sharktank/sharktank/utils/iree.py                                             |      267 |       65 |     76% |35, 81-83, 86-112, 153-156, 233, 240, 254, 258, 262, 268-275, 279, 281, 287, 293, 331, 499, 603-624, 639-646, 658-666, 689 |
| sharktank/sharktank/utils/load\_llm.py                                        |      187 |       88 |     53% |47-60, 97, 140, 145, 148-157, 165, 167, 172-182, 192-209, 258-263, 287-402 |
| sharktank/sharktank/utils/logging.py                                          |        6 |        1 |     83% |        17 |
| sharktank/sharktank/utils/math.py                                             |       12 |        0 |    100% |           |
| sharktank/sharktank/utils/misc.py                                             |       27 |        1 |     96% |        64 |
| sharktank/sharktank/utils/patching.py                                         |       97 |       45 |     54% |16, 56, 60, 75-78, 87-93, 98, 108-133, 141-154, 157-168, 197, 231, 233, 238 |
| sharktank/sharktank/utils/testing.py                                          |      218 |       97 |     56% |121-130, 135-148, 152-156, 195-279, 305, 318-336, 383, 403, 408, 418, 475-478, 484-491, 499-503 |
| sharktank/sharktank/utils/tokenizer.py                                        |       59 |       40 |     32% |34-38, 42-46, 50, 54, 59-62, 65-68, 72, 76-77, 81-103, 107-112 |
| sharktank/sharktank/utils/tree.py                                             |       47 |        2 |     96% |   51, 150 |
| sharktank/tests/evaluate/perplexity\_iree\_test.py                            |      107 |       71 |     34% |36-42, 50-64, 67-77, 87-92, 97-103, 108-114, 119-133, 139-145, 151-157, 162-167, 178-184, 189-195, 199 |
| sharktank/tests/evaluate/perplexity\_torch\_test.py                           |       71 |       46 |     35% |31-36, 39-49, 52-68, 73-78, 84-90, 95-100, 105-112, 117-123, 127 |
| sharktank/tests/examples/main\_test.py                                        |       24 |        1 |     96% |        45 |
| sharktank/tests/export\_ir/export\_test.py                                    |       38 |        0 |    100% |           |
| sharktank/tests/kernels/attention\_template\_test.py                          |       72 |        2 |     97% |  113, 130 |
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
| sharktank/tests/layers/base\_test.py                                          |       22 |        0 |    100% |           |
| sharktank/tests/layers/configs\_test.py                                       |       11 |        0 |    100% |           |
| sharktank/tests/layers/kv\_cache\_test.py                                     |       83 |        0 |    100% |           |
| sharktank/tests/layers/linear\_test.py                                        |       82 |        1 |     99% |       196 |
| sharktank/tests/layers/mixture\_of\_experts\_block\_test.py                   |       58 |        1 |     98% |       326 |
| sharktank/tests/layers/mmdit\_test.py                                         |       56 |        1 |     98% |        96 |
| sharktank/tests/layers/paged\_llama\_attention\_block\_test.py                |       53 |       14 |     74% |90-130, 134 |
| sharktank/tests/layers/pipelined\_paged\_attention\_test.py                   |       83 |        0 |    100% |           |
| sharktank/tests/layers/pipelined\_sharded\_paged\_attention\_test.py          |       85 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_conv2d\_with\_iree\_test.py                   |       78 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_paged\_kv\_cache\_test.py                     |       76 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_paged\_latent\_attention\_block\_test.py      |       71 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_paged\_llama\_attention\_block\_test.py       |       75 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_rotary\_embedding\_test.py                    |       27 |        0 |    100% |           |
| sharktank/tests/models/clip/clip\_test.py                                     |      250 |       53 |     79% |90, 95-110, 120, 130, 210-254, 299-326, 351-385, 394, 404 |
| sharktank/tests/models/deepseek/test\_deepseek.py                             |       23 |        0 |    100% |           |
| sharktank/tests/models/deepseek/test\_sharded.py                              |       91 |       11 |     88% |   172-205 |
| sharktank/tests/models/flux/flux\_test.py                                     |      155 |       72 |     54% |61-63, 67-68, 82, 127-182, 193-211, 220-238, 284, 291, 298-317, 327-353, 363, 372, 381-389, 393 |
| sharktank/tests/models/grok/test\_grok.py                                     |       25 |        0 |    100% |           |
| sharktank/tests/models/llama4/llama4\_test.py                                 |       44 |        0 |    100% |           |
| sharktank/tests/models/llama4/moe\_test.py                                    |       90 |        1 |     99% |       191 |
| sharktank/tests/models/llama/attention\_test.py                               |       59 |        1 |     98% |       176 |
| sharktank/tests/models/llama/benchmark\_amdgpu\_test.py                       |      156 |      108 |     31% |33, 36-45, 55-89, 92-118, 131-227, 252-258, 262-268, 273-279, 286-392, 409-417, 423-429, 436-503, 518-524, 530-536, 540 |
| sharktank/tests/models/llama/quark\_parity\_test.py                           |       55 |       40 |     27% |21-22, 29-101, 105 |
| sharktank/tests/models/llama/rot\_emb\_test.py                                |       37 |        1 |     97% |        81 |
| sharktank/tests/models/llama/sharded\_llama\_test.py                          |      171 |        2 |     99% |  276, 335 |
| sharktank/tests/models/llama/test\_llama.py                                   |       25 |        0 |    100% |           |
| sharktank/tests/models/punet/resnet\_test.py                                  |       41 |        1 |     98% |        94 |
| sharktank/tests/models/punet/sharded\_resnet\_block\_with\_iree\_test.py      |       45 |       12 |     73% |    78-115 |
| sharktank/tests/models/punet/up\_down\_block\_test.py                         |       48 |        1 |     98% |       148 |
| sharktank/tests/models/t5/t5\_test.py                                         |      269 |       59 |     78% |80-108, 146-174, 187-221, 266, 280, 289, 298, 307, 316, 325, 435-477, 526, 535, 544, 553, 562 |
| sharktank/tests/models/vae/vae\_test.py                                       |      210 |      111 |     47% |61-96, 102-111, 116-125, 129-224, 249-263, 268-281, 356-447, 548-555, 564-568, 573-578, 584 |
| sharktank/tests/ops/ops\_test.py                                              |      466 |       30 |     94% |154-157, 222-228, 235-241, 248-255, 391-396, 722 |
| sharktank/tests/ops/pipeline\_parallelized\_test.py                           |      152 |        4 |     97% |56, 184, 196, 206 |
| sharktank/tests/ops/qconv\_test.py                                            |       94 |       12 |     87% |200-236, 240 |
| sharktank/tests/ops/sharded\_test.py                                          |     1356 |       20 |     99% |552-558, 640, 1777, 1780, 1784, 1807, 1811, 1981, 1985, 1994-1996, 2004, 2118 |
| sharktank/tests/pipelines/flux/flux\_pipeline\_test.py                        |       41 |       23 |     44% |25-27, 32-65, 77-121, 128, 135 |
| sharktank/tests/tools/sharktank\_test.py                                      |       19 |        0 |    100% |           |
| sharktank/tests/transforms/dataset\_transforms\_test.py                       |       32 |        1 |     97% |        86 |
| sharktank/tests/types/dataset\_test.py                                        |       94 |        1 |     99% |       178 |
| sharktank/tests/types/layout\_utils\_test.py                                  |       33 |        1 |     97% |        98 |
| sharktank/tests/types/layouts\_test.py                                        |       57 |        1 |     98% |       129 |
| sharktank/tests/types/quantizers\_test.py                                     |      124 |        1 |     99% |       248 |
| sharktank/tests/types/tensors\_test.py                                        |      164 |        1 |     99% |       221 |
| sharktank/tests/utils/iree\_test.py                                           |       56 |        6 |     89% | 69-73, 93 |
| sharktank/tests/utils/patching\_test.py                                       |       44 |        0 |    100% |           |
| sharktank/tests/utils/testing\_test.py                                        |       63 |        3 |     95% |   226-238 |
| sharktank/tests/utils/tree\_test.py                                           |       20 |        0 |    100% |           |
|                                                                     **TOTAL** | **19255** | **3867** | **80%** |           |


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