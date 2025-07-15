# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/nod-ai/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                          |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| sharktank/conftest.py                                                         |      165 |       15 |     91% |352, 357, 364, 371, 446, 470, 475-478, 488, 493, 507, 531, 534-535 |
| sharktank/integration/models/punet/integration\_test.py                       |       94 |       57 |     39% |15-16, 21-31, 52-62, 70-80, 90-101, 110-121, 131-143, 150, 155-167, 174, 185-192, 210-220, 242-255 |
| sharktank/setup.py                                                            |       18 |       18 |      0% |      7-34 |
| sharktank/sharktank/\_\_init\_\_.py                                           |        4 |        1 |     75% |        15 |
| sharktank/sharktank/build/\_\_init\_\_.py                                     |        1 |        1 |      0% |         7 |
| sharktank/sharktank/build/actions.py                                          |       45 |       45 |      0% |     7-109 |
| sharktank/sharktank/evaluate/perplexity\_iree.py                              |      263 |      219 |     17% |70-91, 94-106, 110-125, 130-152, 161-188, 196-233, 243-270, 276-321, 326-375, 379-427, 434-482, 495-551, 558-607, 611 |
| sharktank/sharktank/evaluate/perplexity\_torch.py                             |      197 |      158 |     20% |54-57, 60-72, 76-91, 96-118, 137-163, 167-185, 190-254, 268-307, 321-377, 402-427, 431-471, 475 |
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
| sharktank/sharktank/layers/\_\_init\_\_.py                                    |       16 |        0 |    100% |           |
| sharktank/sharktank/layers/activations.py                                     |        3 |        0 |    100% |           |
| sharktank/sharktank/layers/base.py                                            |      177 |       27 |     85% |131, 206-209, 224, 242, 259-260, 269, 298, 366-374, 385-398, 400, 404-407, 411, 417, 424 |
| sharktank/sharktank/layers/causal\_llm.py                                     |       69 |        6 |     91% |53, 57-62, 132 |
| sharktank/sharktank/layers/configs/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| sharktank/sharktank/layers/configs/config.py                                  |      172 |       16 |     91% |17, 139, 196, 205-211, 219, 234, 248-254, 267, 269, 289, 313 |
| sharktank/sharktank/layers/configs/llm\_configs.py                            |      402 |       78 |     81% |28-29, 214, 218, 222, 229, 231, 235, 237, 246, 249-252, 278-285, 295-298, 304-307, 311-316, 328-331, 342-343, 354-355, 446-447, 452, 543, 549-557, 566-578, 617, 639, 663-667, 703-706, 710-714 |
| sharktank/sharktank/layers/conv.py                                            |      100 |       61 |     39% |48, 58, 61, 63, 80, 95-110, 113-143, 157-172, 175-205 |
| sharktank/sharktank/layers/ffn\_block.py                                      |       26 |        0 |    100% |           |
| sharktank/sharktank/layers/ffn\_moe\_block.py                                 |       83 |       25 |     70% |65-73, 203-237, 243-246, 253-259 |
| sharktank/sharktank/layers/latent\_attention\_block.py                        |       55 |        4 |     93% |42, 62, 85-86 |
| sharktank/sharktank/layers/linear.py                                          |       42 |        4 |     90% |56, 67, 75, 83 |
| sharktank/sharktank/layers/mixture\_of\_experts\_block.py                     |       71 |        5 |     93% |48, 52, 60, 105-109 |
| sharktank/sharktank/layers/mmdit.py                                           |      102 |        0 |    100% |           |
| sharktank/sharktank/layers/modulation.py                                      |       21 |        0 |    100% |           |
| sharktank/sharktank/layers/norm.py                                            |       37 |        0 |    100% |           |
| sharktank/sharktank/layers/paged\_attention.py                                |      482 |       24 |     95% |136, 143-149, 210, 757, 1125, 1135-1136, 1143, 1146, 1150, 1152, 1154, 1173, 1179-1184, 1189-1195, 1199, 1202, 1204 |
| sharktank/sharktank/layers/paged\_llama\_attention\_block.py                  |      122 |       11 |     91% |116, 120-121, 134, 171, 173, 175, 243, 259-263 |
| sharktank/sharktank/layers/rotary\_embedding.py                               |      174 |       15 |     91% |67-69, 94, 112, 126-132, 151-159, 265-266, 309 |
| sharktank/sharktank/layers/rotary\_embedding\_hf.py                           |       54 |        0 |    100% |           |
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
| sharktank/sharktank/models/llama/testing.py                                   |       35 |        0 |    100% |           |
| sharktank/sharktank/models/llama/toy\_llama.py                                |       28 |        6 |     79% | 66-72, 76 |
| sharktank/sharktank/models/llm/\_\_init\_\_.py                                |        1 |        0 |    100% |           |
| sharktank/sharktank/models/llm/llm.py                                         |      135 |        4 |     97% |213, 275, 300, 303 |
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
| sharktank/sharktank/ops/\_\_init\_\_.py                                       |       12 |        0 |    100% |           |
| sharktank/sharktank/ops/\_registry.py                                         |      144 |        8 |     94% |108, 113, 229-232, 243, 278, 297 |
| sharktank/sharktank/ops/attention\_impls.py                                   |       48 |       29 |     40% |32-38, 42-53, 58-79, 86, 97 |
| sharktank/sharktank/ops/custom\_impls.py                                      |       86 |       22 |     74% |62-66, 84, 100, 119, 123-134, 149, 153-156, 179, 181, 183 |
| sharktank/sharktank/ops/default\_impls.py                                     |      495 |       88 |     82% |116, 118, 150, 152, 154, 187, 189, 191, 254-257, 289, 291, 305-306, 321-328, 342-349, 363, 368-386, 400, 410, 594, 613, 624-626, 665, 715, 785, 790, 795, 801, 833-840, 846, 935, 939, 985, 990-1007, 1012, 1017 |
| sharktank/sharktank/ops/qconv\_impls.py                                       |      123 |       31 |     75% |47, 53, 67-71, 88, 94, 109, 137-142, 168-177, 229, 252, 270-285, 298, 303, 310 |
| sharktank/sharktank/ops/qlinear\_impls.py                                     |       91 |       16 |     82% |40, 65, 84, 88, 102-105, 116-117, 143-144, 162, 165, 188-190, 209 |
| sharktank/sharktank/ops/quantized\_impls.py                                   |       21 |        7 |     67% |24-25, 32, 41, 49-52 |
| sharktank/sharktank/ops/shape.py                                              |       23 |        1 |     96% |        76 |
| sharktank/sharktank/ops/sharded\_impls.py                                     |      825 |       59 |     93% |220, 442, 484-486, 493, 501, 516, 541, 552-553, 560-561, 631-640, 690-698, 860, 873, 919, 932, 935, 940, 943, 1002-1004, 1064, 1081, 1090, 1099, 1126, 1140, 1150, 1174, 1201, 1203, 1213, 1215, 1280, 1436, 1645, 1652, 1771-1772, 1796, 1859, 1872, 1885, 1889, 1929-1930, 1935-1936 |
| sharktank/sharktank/ops/signatures.py                                         |      724 |      146 |     80% |101, 114, 120, 131, 142, 159, 164, 176, 195, 228, 247, 265, 280, 299, 317, 332, 350, 363, 369, 385, 398, 414, 428, 442-455, 467, 483-489, 495, 508, 523, 528-534, 540, 553, 559, 575, 616, 638, 646, 665, 673, 690, 698, 715, 721, 734, 748, 776, 789, 814, 840, 858, 864, 880, 897, 918, 937, 945, 956, 968, 986, 994, 1001-1007, 1013, 1026, 1034, 1052, 1058, 1071, 1081, 1102, 1115, 1133, 1137, 1146, 1157, 1171, 1182, 1193, 1215, 1223, 1239, 1255, 1274, 1284, 1301, 1310, 1321, 1327, 1340, 1351, 1364, 1370, 1381, 1389, 1405, 1413, 1429, 1435, 1446, 1471, 1483, 1489, 1502, 1508, 1521, 1527, 1540, 1546, 1559, 1565, 1576, 1582, 1595, 1601, 1614, 1626, 1644, 1658, 1697, 1705, 1721, 1727, 1738, 1744, 1755, 1769, 1796 |
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
| sharktank/sharktank/types/layout\_utils.py                                    |      111 |        8 |     93% |90, 125, 127, 204-207, 250, 259 |
| sharktank/sharktank/types/layouts.py                                          |      268 |       51 |     81% |113, 156-157, 162-163, 188-195, 282-294, 332, 337, 342, 391, 395, 471, 479, 487, 495, 508, 521, 529, 541, 544, 547-555, 558-568, 634, 642, 645, 662, 667 |
| sharktank/sharktank/types/ocp\_floats.py                                      |       79 |       18 |     77% |89-114, 159 |
| sharktank/sharktank/types/pipelining.py                                       |       42 |        0 |    100% |           |
| sharktank/sharktank/types/quantizers.py                                       |      329 |       59 |     82% |75-83, 94, 148, 240, 271-272, 275-276, 292-298, 309, 329-330, 341, 419-420, 429, 444, 450, 464, 470, 495, 497, 556, 558, 588, 592, 607-608, 625, 652, 661, 692, 694, 727, 731, 735, 758, 776, 783, 813-814, 818, 830-840 |
| sharktank/sharktank/types/sharding.py                                         |      158 |        8 |     95% |18, 34, 100, 137-148, 321 |
| sharktank/sharktank/types/tensors.py                                          |      841 |      152 |     82% |72-77, 87, 93, 103, 115, 126, 158-164, 184-185, 190, 219, 227, 248, 253, 258, 286, 290, 304, 315, 326, 347-348, 354, 372, 401, 405, 413-414, 425-427, 441-443, 467-469, 477-478, 508, 513-515, 520-522, 525-527, 569, 573, 590-591, 596, 642-644, 648, 650, 690, 702, 734-735, 740, 774, 778, 780, 804, 814, 823, 877-878, 901-902, 907-908, 1006, 1014, 1028-1032, 1106-1116, 1146-1147, 1165, 1167, 1169, 1174-1176, 1181, 1304, 1317, 1319-1324, 1326, 1328, 1339, 1342, 1344, 1354, 1357, 1386, 1427, 1448-1450, 1455, 1475-1485, 1511-1512, 1525, 1540, 1542, 1544, 1613, 1624-1625, 1633-1634, 1642-1643, 1649-1652, 1678-1679, 1806-1807, 1827, 1831-1832 |
| sharktank/sharktank/types/theta.py                                            |      310 |       45 |     85% |69, 77, 106, 110, 137-147, 167, 179-180, 209-210, 216, 231, 341-345, 405, 453-454, 477-478, 493-494, 514-515, 529-530, 547-548, 590-592, 596, 634-636, 641 |
| sharktank/sharktank/utils/\_\_init\_\_.py                                     |        1 |        0 |    100% |           |
| sharktank/sharktank/utils/azure.py                                            |       58 |       58 |      0% |     7-121 |
| sharktank/sharktank/utils/cli.py                                              |      109 |       66 |     39% |35-39, 72-178, 189-194, 205-217, 228-260, 275-296, 307, 320-323, 333, 346-358, 375-376, 378, 392, 397, 406-424 |
| sharktank/sharktank/utils/create\_cache.py                                    |        7 |        1 |     86% |        12 |
| sharktank/sharktank/utils/debugging.py                                        |       91 |       29 |     68% |46-63, 67-74, 81-83, 127, 138 |
| sharktank/sharktank/utils/evaluate.py                                         |       63 |       33 |     48% |32-51, 55, 69-88, 126-145 |
| sharktank/sharktank/utils/export.py                                           |       70 |        2 |     97% |  179, 212 |
| sharktank/sharktank/utils/export\_artifacts.py                                |      179 |      126 |     30% |22, 39-47, 61, 68, 75, 82, 89, 123-157, 211-231, 249-261, 265-283, 295-311, 335-374, 398-434, 458-471, 497-523, 545-547 |
| sharktank/sharktank/utils/hf.py                                               |       35 |       24 |     31% |26-54, 64-78 |
| sharktank/sharktank/utils/hf\_datasets.py                                     |       75 |       22 |     71% |37-55, 65, 73, 82-83, 88, 478-496, 500 |
| sharktank/sharktank/utils/io.py                                               |       39 |        9 |     77% |65-72, 83-86 |
| sharktank/sharktank/utils/iree.py                                             |      291 |       51 |     82% |37, 199-202, 279, 286, 300, 304, 308, 314-321, 327, 333, 339, 381, 552, 656-677, 692-699, 711-719, 742 |
| sharktank/sharktank/utils/load\_llm.py                                        |      198 |       93 |     53% |50-64, 69-81, 124, 169, 176, 179-188, 196, 198, 203-213, 223-240, 273, 293-298, 322-437 |
| sharktank/sharktank/utils/logging.py                                          |        6 |        1 |     83% |        17 |
| sharktank/sharktank/utils/math.py                                             |       12 |        0 |    100% |           |
| sharktank/sharktank/utils/misc.py                                             |       34 |        2 |     94% |    20, 73 |
| sharktank/sharktank/utils/patching.py                                         |       97 |       45 |     54% |16, 56, 60, 75-78, 87-93, 98, 108-133, 141-154, 157-168, 197, 231, 233, 238 |
| sharktank/sharktank/utils/random.py                                           |        8 |        0 |    100% |           |
| sharktank/sharktank/utils/testing.py                                          |      305 |      184 |     40% |146-262, 285-291, 302, 315-325, 338-362, 368-389, 405-414, 419-432, 436-440, 479-563, 590, 603, 626-644, 692, 737, 742, 748-752, 760-763, 769-776, 784-788 |
| sharktank/sharktank/utils/tokenizer.py                                        |       64 |       43 |     33% |34-38, 42-46, 50, 54, 58, 63-66, 69-72, 76, 80-81, 85-110, 114-119 |
| sharktank/sharktank/utils/tree.py                                             |       47 |        2 |     96% |   51, 150 |
| sharktank/tests/evaluate/perplexity\_iree\_test.py                            |      124 |       83 |     33% |36-42, 50-64, 67-77, 87-92, 97-103, 108-114, 119-133, 138-144, 150-156, 167-173, 179-185, 190-195, 206-212, 217-223, 227 |
| sharktank/tests/evaluate/perplexity\_torch\_test.py                           |       62 |       40 |     35% |32-37, 40-50, 53-69, 74-79, 84-89, 94-101, 106-112, 116 |
| sharktank/tests/examples/main\_test.py                                        |       24 |        1 |     96% |        45 |
| sharktank/tests/examples/paged\_llm\_v1\_test.py                              |       16 |        5 |     69% |     33-37 |
| sharktank/tests/export\_ir/export\_test.py                                    |       38 |        0 |    100% |           |
| sharktank/tests/kernels/attention\_template\_test.py                          |       72 |        2 |     97% |  113, 130 |
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
| sharktank/tests/layers/kv\_cache\_test.py                                     |      137 |        0 |    100% |           |
| sharktank/tests/layers/linear\_test.py                                        |       82 |        1 |     99% |       196 |
| sharktank/tests/layers/mixture\_of\_experts\_block\_test.py                   |       58 |        1 |     98% |       329 |
| sharktank/tests/layers/mmdit\_test.py                                         |       56 |        1 |     98% |        96 |
| sharktank/tests/layers/paged\_llama\_attention\_block\_test.py                |       53 |       14 |     74% |90-130, 134 |
| sharktank/tests/layers/pipelined\_paged\_attention\_test.py                   |      102 |        0 |    100% |           |
| sharktank/tests/layers/pipelined\_sharded\_paged\_attention\_test.py          |      104 |        0 |    100% |           |
| sharktank/tests/layers/rotary\_embedding\_hf\_test.py                         |       85 |        0 |    100% |           |
| sharktank/tests/layers/rotary\_embedding\_test.py                             |       60 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_conv2d\_with\_iree\_test.py                   |       78 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_paged\_kv\_cache\_test.py                     |       93 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_paged\_latent\_attention\_block\_test.py      |       71 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_paged\_llama\_attention\_block\_test.py       |       76 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_rotary\_embedding\_test.py                    |       27 |        0 |    100% |           |
| sharktank/tests/models/clip/clip\_test.py                                     |      251 |       53 |     79% |89, 94-109, 119, 129, 209-253, 298-325, 350-384, 393, 403 |
| sharktank/tests/models/deepseek/test\_deepseek.py                             |       47 |       13 |     72% |     76-99 |
| sharktank/tests/models/deepseek/test\_sharded.py                              |       87 |       12 |     86% |   169-204 |
| sharktank/tests/models/flux/flux\_test.py                                     |      155 |       72 |     54% |61-63, 67-68, 82, 127-182, 193-211, 220-238, 284, 291, 298-317, 327-353, 363, 372, 381-389, 393 |
| sharktank/tests/models/grok/test\_grok.py                                     |       25 |        0 |    100% |           |
| sharktank/tests/models/llama4/llama4\_test.py                                 |       61 |        8 |     87% |105-125, 139 |
| sharktank/tests/models/llama4/moe\_test.py                                    |       90 |        1 |     99% |       192 |
| sharktank/tests/models/llama/attention\_test.py                               |       59 |        1 |     98% |       180 |
| sharktank/tests/models/llama/benchmark\_amdgpu\_test.py                       |      135 |       85 |     37% |34, 37-48, 58-92, 95-105, 114-168, 193-208, 212-227, 231-250, 255-275, 282-352, 369-390, 396-414, 421-461, 476-491, 497-515, 519 |
| sharktank/tests/models/llama/quark\_parity\_test.py                           |       55 |       40 |     27% |21-22, 29-101, 105 |
| sharktank/tests/models/llama/rot\_emb\_test.py                                |       37 |        1 |     97% |        82 |
| sharktank/tests/models/llama/sharded\_llama\_test.py                          |      171 |        2 |     99% |  278, 337 |
| sharktank/tests/models/llama/test\_llama.py                                   |       43 |        6 |     86% |     78-92 |
| sharktank/tests/models/punet/resnet\_test.py                                  |       41 |        1 |     98% |        94 |
| sharktank/tests/models/punet/sharded\_resnet\_block\_with\_iree\_test.py      |       43 |       12 |     72% |    76-113 |
| sharktank/tests/models/punet/up\_down\_block\_test.py                         |       48 |        1 |     98% |       148 |
| sharktank/tests/models/t5/t5\_test.py                                         |      270 |       59 |     78% |79-107, 145-173, 186-220, 265, 279, 288, 297, 306, 315, 324, 434-476, 525, 534, 543, 552, 561 |
| sharktank/tests/models/vae/vae\_test.py                                       |      210 |      111 |     47% |61-96, 102-111, 116-125, 129-224, 249-263, 268-281, 356-447, 548-555, 564-568, 573-578, 584 |
| sharktank/tests/ops/ops\_test.py                                              |      588 |       31 |     95% |159-162, 227-233, 240-246, 253-260, 583-588, 952, 987 |
| sharktank/tests/ops/pipeline\_parallelized\_test.py                           |      152 |        4 |     97% |56, 184, 196, 206 |
| sharktank/tests/ops/qconv\_test.py                                            |       94 |       12 |     87% |200-236, 240 |
| sharktank/tests/ops/sharded\_test.py                                          |     1357 |       20 |     99% |553-559, 641, 1778, 1781, 1785, 1808, 1812, 1982, 1986, 1995-1997, 2005, 2119 |
| sharktank/tests/pipelines/flux/flux\_pipeline\_test.py                        |       41 |       23 |     44% |25-27, 32-65, 77-121, 128, 135 |
| sharktank/tests/pytest\_fixtures\_test.py                                     |       19 |        0 |    100% |           |
| sharktank/tests/tools/sharktank\_test.py                                      |       19 |        0 |    100% |           |
| sharktank/tests/transforms/dataset\_transforms\_test.py                       |       32 |        1 |     97% |        86 |
| sharktank/tests/types/dataset\_test.py                                        |       94 |        1 |     99% |       178 |
| sharktank/tests/types/layout\_utils\_test.py                                  |       75 |        1 |     99% |       233 |
| sharktank/tests/types/layouts\_test.py                                        |       57 |        1 |     98% |       129 |
| sharktank/tests/types/quantizers\_test.py                                     |      226 |        1 |     99% |       493 |
| sharktank/tests/types/tensors\_test.py                                        |      164 |        1 |     99% |       221 |
| sharktank/tests/utils/iree\_test.py                                           |       56 |        6 |     89% | 69-73, 93 |
| sharktank/tests/utils/patching\_test.py                                       |       44 |        0 |    100% |           |
| sharktank/tests/utils/testing\_test.py                                        |       88 |        4 |     95% |   279-292 |
| sharktank/tests/utils/tree\_test.py                                           |       20 |        0 |    100% |           |
|                                                                     **TOTAL** | **20695** | **4158** | **80%** |           |


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