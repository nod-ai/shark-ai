# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/nod-ai/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| amdsharktank/amdsharktank/\_\_init\_\_.py                                           |        4 |        1 |     75% |        15 |
| amdsharktank/amdsharktank/build/\_\_init\_\_.py                                     |        1 |        1 |      0% |         7 |
| amdsharktank/amdsharktank/build/actions.py                                          |       45 |       45 |      0% |     7-109 |
| amdsharktank/amdsharktank/evaluate/perplexity\_iree.py                              |      248 |      208 |     16% |71-91, 95-101, 106-128, 136-172, 180-216, 226-252, 257-302, 307-356, 359-414, 421-469, 482-536, 543-592, 596 |
| amdsharktank/amdsharktank/evaluate/perplexity\_torch.py                             |      184 |      146 |     21% |54-56, 60-66, 71-93, 112-139, 143-160, 164-228, 242-281, 295-350, 374-397, 401-441, 445 |
| amdsharktank/amdsharktank/examples/export\_paged\_llm\_v1.py                        |      133 |       47 |     65% |48, 84-90, 100, 113-131, 238-326, 330 |
| amdsharktank/amdsharktank/examples/paged\_llm\_v1.py                                |       52 |       41 |     21% |38-132, 136 |
| amdsharktank/amdsharktank/examples/pipeline/export\_ppffn\_net.py                   |       66 |        4 |     94% |139, 145, 176, 183 |
| amdsharktank/amdsharktank/examples/sharding/export\_ffn\_net.py                     |       59 |       13 |     78% |51-63, 82, 88, 113, 120 |
| amdsharktank/amdsharktank/kernels/\_\_init\_\_.py                                   |       14 |        0 |    100% |           |
| amdsharktank/amdsharktank/kernels/attention.py                                      |       22 |        0 |    100% |           |
| amdsharktank/amdsharktank/kernels/base.py                                           |       52 |        5 |     90% |136, 155-160 |
| amdsharktank/amdsharktank/kernels/batch\_matmul\_transpose\_b.py                    |       49 |        0 |    100% |           |
| amdsharktank/amdsharktank/kernels/bitcast.py                                        |       63 |       40 |     37% |58-69, 75-88, 97-108, 114-127, 136-139 |
| amdsharktank/amdsharktank/kernels/conv\_2d\_nchw\_fchw.py                           |       64 |        0 |    100% |           |
| amdsharktank/amdsharktank/kernels/einsum\_2args\_q4.py                              |      122 |        2 |     98% |   69, 179 |
| amdsharktank/amdsharktank/kernels/gemm\_fp4.py                                      |       17 |        0 |    100% |           |
| amdsharktank/amdsharktank/kernels/gemm\_fp4\_asm.py                                 |       37 |       18 |     51% |29-45, 64-182, 195 |
| amdsharktank/amdsharktank/kernels/mlir\_kernel.py                                   |      204 |       18 |     91% |40, 43, 47, 112, 123, 129, 131, 220, 262, 269, 277, 321, 329, 369-374, 382 |
| amdsharktank/amdsharktank/kernels/mmt\_block\_scaled\_offset\_q4.py                 |       50 |        3 |     94% |     94-96 |
| amdsharktank/amdsharktank/kernels/mmt\_block\_scaled\_q8.py                         |       38 |        0 |    100% |           |
| amdsharktank/amdsharktank/kernels/mmt\_super\_block\_scaled\_offset\_q4.py          |       59 |        0 |    100% |           |
| amdsharktank/amdsharktank/kernels/mmtfp.py                                          |       41 |        2 |     95% |     68-69 |
| amdsharktank/amdsharktank/kernels/pooling\_nchw\_sum.py                             |       38 |        0 |    100% |           |
| amdsharktank/amdsharktank/kernels/rotary.py                                         |       31 |        0 |    100% |           |
| amdsharktank/amdsharktank/kernels/topk.py                                           |       30 |        0 |    100% |           |
| amdsharktank/amdsharktank/kernels/wave/attention.py                                 |       48 |        0 |    100% |           |
| amdsharktank/amdsharktank/kernels/wave/extend\_attention.py                         |       58 |       31 |     47% |62-97, 140-211 |
| amdsharktank/amdsharktank/kernels/wave/mxfp4\_gemm.py                               |       99 |       74 |     25% |41-112, 122-161, 185-234 |
| amdsharktank/amdsharktank/kernels/wave/templates/extend\_attention\_kernel.py       |      196 |      184 |      6% |    47-460 |
| amdsharktank/amdsharktank/kernels/wave/utils.py                                     |      140 |      114 |     19% |68-74, 82-175, 213-253, 274-311, 317-320 |
| amdsharktank/amdsharktank/layers/\_\_init\_\_.py                                    |       16 |        0 |    100% |           |
| amdsharktank/amdsharktank/layers/activations.py                                     |        3 |        0 |    100% |           |
| amdsharktank/amdsharktank/layers/base.py                                            |      177 |       27 |     85% |131, 206-209, 224, 242, 259-260, 269, 298, 366-374, 385-398, 400, 404-407, 411, 417, 424 |
| amdsharktank/amdsharktank/layers/causal\_llm.py                                     |       22 |        7 |     68% |     58-64 |
| amdsharktank/amdsharktank/layers/configs/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| amdsharktank/amdsharktank/layers/configs/config.py                                  |      170 |       15 |     91% |139, 196, 205-211, 219, 234, 248-254, 267, 269, 289, 313 |
| amdsharktank/amdsharktank/layers/configs/llm\_configs.py                            |      558 |      162 |     71% |47-49, 239, 241, 243, 245, 247, 249, 251, 257, 259, 261, 263, 271, 275, 279, 286, 288, 292, 294, 297, 299, 301, 305, 307, 309, 320, 323, 326-329, 332-352, 355-362, 367-390, 398-401, 407-410, 414-419, 433-434, 445-446, 457-458, 488, 519-523, 527, 533, 538, 556, 616, 648-649, 654, 708-722, 726-729, 737-769, 813, 819-827, 836-848, 887, 909, 933-937, 973-976, 980-984 |
| amdsharktank/amdsharktank/layers/conv.py                                            |      100 |       61 |     39% |48, 58, 61, 63, 80, 95-110, 113-143, 157-172, 175-205 |
| amdsharktank/amdsharktank/layers/ffn\_block.py                                      |       26 |        0 |    100% |           |
| amdsharktank/amdsharktank/layers/ffn\_moe\_block.py                                 |      105 |       29 |     72% |73-76, 79-87, 252-286, 292-295, 302-308 |
| amdsharktank/amdsharktank/layers/kv\_cache.py                                       |       16 |        0 |    100% |           |
| amdsharktank/amdsharktank/layers/latent\_attention\_block.py                        |       52 |        5 |     90% |42, 61, 66, 76, 96 |
| amdsharktank/amdsharktank/layers/linear.py                                          |       43 |        4 |     91% |55, 66, 74, 84 |
| amdsharktank/amdsharktank/layers/mixture\_of\_experts\_block.py                     |       86 |        6 |     93% |51, 55, 63, 98, 118, 227 |
| amdsharktank/amdsharktank/layers/mmdit.py                                           |      103 |        0 |    100% |           |
| amdsharktank/amdsharktank/layers/modulation.py                                      |       21 |        0 |    100% |           |
| amdsharktank/amdsharktank/layers/norm.py                                            |       37 |        0 |    100% |           |
| amdsharktank/amdsharktank/layers/paged\_attention.py                                |      272 |       15 |     94% |202, 356, 360-363, 367, 387, 389, 427, 767-772, 931 |
| amdsharktank/amdsharktank/layers/paged\_llama\_attention\_block.py                  |      167 |       26 |     84% |93-97, 121-122, 156, 183-203, 342-351, 383, 385, 387, 515-517 |
| amdsharktank/amdsharktank/layers/rotary\_embedding.py                               |       36 |        0 |    100% |           |
| amdsharktank/amdsharktank/layers/rotary\_embedding\_hf.py                           |      121 |        3 |     98% |104, 252-253 |
| amdsharktank/amdsharktank/layers/testing.py                                         |       67 |       21 |     69% |302, 372-384, 390, 434-473 |
| amdsharktank/amdsharktank/layers/token\_embedding.py                                |       12 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/\_\_init\_\_.py                                    |        7 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/clip/\_\_init\_\_.py                               |        2 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/clip/clip.py                                       |      206 |       31 |     85% |80, 123, 131, 143, 159-162, 171, 249, 326, 337, 340, 343, 397, 412, 439, 454, 487, 490, 493, 544-557, 568-570 |
| amdsharktank/amdsharktank/models/clip/export.py                                     |       27 |       10 |     63% |40-43, 51-59 |
| amdsharktank/amdsharktank/models/clip/export\_toy\_text\_model\_iree\_test\_data.py |       11 |        1 |     91% |        29 |
| amdsharktank/amdsharktank/models/clip/testing.py                                    |       67 |        4 |     94% |   175-179 |
| amdsharktank/amdsharktank/models/deepseek/testing.py                                |       22 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/deepseek/toy\_deepseek.py                          |       33 |        9 |     73% | 84-94, 98 |
| amdsharktank/amdsharktank/models/dummy/\_\_init\_\_.py                              |        1 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/dummy/dummy.py                                     |       39 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/flux/\_\_init\_\_.py                               |        1 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/flux/compile.py                                    |        1 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/flux/export.py                                     |       55 |       24 |     56% |35-36, 56, 80, 95-98, 104-127 |
| amdsharktank/amdsharktank/models/flux/export\_flux\_transformer\_mlir.py            |       13 |       13 |      0% |      7-38 |
| amdsharktank/amdsharktank/models/flux/flux.py                                       |      233 |       29 |     88% |82-91, 117-121, 129, 135, 137, 142, 147, 152, 218, 222, 235, 242, 268-279, 288, 407 |
| amdsharktank/amdsharktank/models/flux/testing.py                                    |       54 |       10 |     81% |31, 154, 209-227 |
| amdsharktank/amdsharktank/models/grok/testing.py                                    |       22 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/grok/toy\_grok.py                                  |       31 |        6 |     81% | 67-72, 76 |
| amdsharktank/amdsharktank/models/llama4/testing.py                                  |       41 |        1 |     98% |        18 |
| amdsharktank/amdsharktank/models/llama/testing.py                                   |       58 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/llama/toy\_llama.py                                |       51 |        6 |     88% |160-166, 170 |
| amdsharktank/amdsharktank/models/llm/\_\_init\_\_.py                                |        1 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/llm/config.py                                      |       43 |        4 |     91% |     39-42 |
| amdsharktank/amdsharktank/models/llm/export.py                                      |       78 |       20 |     74% |25-30, 36, 71-73, 81-85, 90-93, 119, 127-130, 157 |
| amdsharktank/amdsharktank/models/llm/llm.py                                         |       99 |        6 |     94% |178, 202, 231, 234, 369-370 |
| amdsharktank/amdsharktank/models/llm/testing.py                                     |       69 |       12 |     83% |   174-191 |
| amdsharktank/amdsharktank/models/punet/config.py                                    |       84 |       34 |     60% |70-82, 87-91, 98-122, 126-130 |
| amdsharktank/amdsharktank/models/punet/layers.py                                    |      324 |      191 |     41% |135-180, 195-226, 258, 280-285, 303-330, 341-355, 366-388, 393-397, 400-412, 420-445, 453-500, 514-520, 525-530, 617-625, 628-632, 655-660, 669-696, 721-726, 729, 739-740, 743-745 |
| amdsharktank/amdsharktank/models/punet/sharding.py                                  |       31 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/punet/testing.py                                   |       65 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/punet/tools/sample\_data.py                        |       26 |       21 |     19% |15-20, 33-46, 50-53 |
| amdsharktank/amdsharktank/models/t5/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/t5/export.py                                       |       58 |       31 |     47% |42-46, 56-72, 97-105, 117-143 |
| amdsharktank/amdsharktank/models/t5/t5.py                                           |      344 |      103 |     70% |126, 160, 189, 236-240, 266-269, 272-284, 313, 326, 334-336, 347, 360, 436-448, 464-480, 517, 557-571, 591-597, 605-642, 649-655, 662, 710, 713, 719-753, 780, 787, 793, 801, 840-842, 850-861, 894-895, 901-905, 911, 926-927, 949-959, 985, 1013, 1018, 1023-1025, 1031, 1034 |
| amdsharktank/amdsharktank/models/t5/testing.py                                      |       22 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/vae/config.py                                      |       39 |       13 |     67% |44-48, 54-62 |
| amdsharktank/amdsharktank/models/vae/layers.py                                      |       97 |        6 |     94% |48, 101, 103, 205, 231, 235 |
| amdsharktank/amdsharktank/models/vae/model.py                                       |       67 |        7 |     90% |24-25, 33, 63, 94, 108, 116 |
| amdsharktank/amdsharktank/models/vae/testing.py                                     |       14 |        0 |    100% |           |
| amdsharktank/amdsharktank/models/vae/tools/diffuser\_ref.py                         |       50 |       13 |     74% |39-60, 87, 104 |
| amdsharktank/amdsharktank/models/vae/tools/run\_vae.py                              |       75 |       47 |     37% |64-158, 162 |
| amdsharktank/amdsharktank/models/vae/tools/sample\_data.py                          |       14 |        5 |     64% |27-29, 39-40 |
| amdsharktank/amdsharktank/ops/\_\_init\_\_.py                                       |       13 |        0 |    100% |           |
| amdsharktank/amdsharktank/ops/\_registry.py                                         |      218 |       15 |     93% |52, 142, 147, 277-280, 292, 329, 332-335, 348, 466, 488, 496, 543 |
| amdsharktank/amdsharktank/ops/attention\_impls.py                                   |      156 |       41 |     74% |49-51, 81, 154-156, 165, 182, 239, 255-302 |
| amdsharktank/amdsharktank/ops/cpu\_impls.py                                         |       20 |        1 |     95% |        43 |
| amdsharktank/amdsharktank/ops/custom\_impls.py                                      |      122 |       47 |     61% |66-70, 88, 104, 125-134, 151-170, 200-220, 229, 233-236, 259, 261, 263 |
| amdsharktank/amdsharktank/ops/default\_impls.py                                     |      615 |      116 |     81% |54, 147, 149, 181, 183, 185, 218, 220, 222, 246-253, 258-266, 271-277, 282-290, 295-302, 308-322, 354, 356, 370-371, 445, 466-487, 501, 511, 695, 706, 737, 748-750, 789, 895, 966, 971, 976, 982, 1050, 1145, 1149, 1216-1228, 1233, 1238, 1270 |
| amdsharktank/amdsharktank/ops/qconv\_impls.py                                       |      123 |       31 |     75% |47, 53, 67-71, 88, 94, 109, 137-142, 168-177, 229, 252, 270-285, 298, 303, 310 |
| amdsharktank/amdsharktank/ops/qlinear\_impls.py                                     |       91 |       16 |     82% |41, 72, 91, 95, 109-112, 123-124, 150-151, 170, 173, 196-198, 217 |
| amdsharktank/amdsharktank/ops/quantized\_impls.py                                   |      233 |       13 |     94% |84, 92, 94-100, 102-109, 120-121, 145, 258-260, 397 |
| amdsharktank/amdsharktank/ops/shape.py                                              |       28 |        1 |     96% |        84 |
| amdsharktank/amdsharktank/ops/sharded\_impls.py                                     |      925 |       85 |     91% |228, 466, 528-530, 537, 545, 560, 570-574, 584-589, 604-605, 693-702, 752-760, 904, 947, 999, 1013, 1015, 1018, 1023, 1026, 1092-1094, 1177-1181, 1194, 1211, 1220, 1228-1230, 1256, 1272, 1282, 1306, 1332, 1334, 1344, 1346, 1411, 1461, 1568, 1598, 1603, 1731, 1733, 1828, 1838-1848, 2029-2030, 2054, 2117, 2126-2131, 2143, 2147, 2187-2188, 2193-2194 |
| amdsharktank/amdsharktank/ops/signatures.py                                         |      360 |       43 |     88% |146, 163, 206, 231, 264, 283, 301, 316, 335, 353, 368, 408, 424, 430, 446, 459, 501-507, 528, 598, 606, 639, 659, 672, 697, 736, 757, 780, 811, 867, 904, 922, 938, 994, 1037, 1095, 1226, 1279, 1336 |
| amdsharktank/amdsharktank/ops/utils.py                                              |       93 |        5 |     95% |38, 43, 229, 234, 239 |
| amdsharktank/amdsharktank/pipelines/flux/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| amdsharktank/amdsharktank/pipelines/flux/flux\_pipeline.py                          |      138 |      109 |     21% |40-93, 121-133, 155-188, 210-228, 238, 244-246, 269-277, 295-317, 320, 339-368, 373-474, 478 |
| amdsharktank/amdsharktank/tools/convert\_dataset.py                                 |       27 |        1 |     96% |        51 |
| amdsharktank/amdsharktank/tools/e2e\_model\_test.py                                 |      273 |      253 |      7% |46-62, 75-541, 547-623, 636 |
| amdsharktank/amdsharktank/tools/import\_hf\_dataset.py                              |       16 |       10 |     38% | 33-54, 60 |
| amdsharktank/amdsharktank/transforms/dataset/\_\_init\_\_.py                        |        2 |        0 |    100% |           |
| amdsharktank/amdsharktank/transforms/dataset/dataset.py                             |       21 |        6 |     71% | 25, 43-47 |
| amdsharktank/amdsharktank/transforms/dataset/sharding.py                            |       38 |       28 |     26% |32-34, 37-49, 54-68, 71 |
| amdsharktank/amdsharktank/types/\_\_init\_\_.py                                     |        6 |        0 |    100% |           |
| amdsharktank/amdsharktank/types/gguf\_interop/\_\_init\_\_.py                       |        2 |        0 |    100% |           |
| amdsharktank/amdsharktank/types/gguf\_interop/base.py                               |       70 |       50 |     29% |42-44, 48-61, 65-81, 99-104, 115-138, 142-163, 167-168 |
| amdsharktank/amdsharktank/types/gguf\_interop/layouts.py                            |      104 |       68 |     35% |47-49, 53-60, 64, 67, 107-110, 118-144, 157, 160, 170-217, 226-227, 230, 234, 237, 246-247, 250, 254, 257, 282-283, 287-300, 304, 307 |
| amdsharktank/amdsharktank/types/layout\_utils.py                                    |      111 |        8 |     93% |90, 125, 127, 204-207, 250, 259 |
| amdsharktank/amdsharktank/types/layouts.py                                          |      281 |       46 |     84% |117, 160-161, 166-167, 192-199, 290-302, 400, 404, 480, 488, 496, 504, 517, 530, 538, 550, 553, 556-564, 567-577, 684, 700, 705 |
| amdsharktank/amdsharktank/types/misc.py                                             |       55 |        1 |     98% |       122 |
| amdsharktank/amdsharktank/types/ocp\_floats.py                                      |       98 |       19 |     81% |93-118, 163, 292 |
| amdsharktank/amdsharktank/types/pipelining.py                                       |       66 |        2 |     97% |  163, 183 |
| amdsharktank/amdsharktank/types/quantizers.py                                       |      293 |       40 |     86% |131, 177-178, 181-182, 214, 253, 308-309, 318, 336, 342, 356, 362, 388, 390, 453, 455, 493-494, 551, 583, 585, 630, 653, 661, 691-692, 696, 708-718, 733, 745-752 |
| amdsharktank/amdsharktank/types/sharding.py                                         |      155 |       40 |     74% |33, 104-106, 109-150, 262-263, 266, 320, 363-366, 369-402, 407-408, 411, 450-451, 454, 466-475 |
| amdsharktank/amdsharktank/types/tensors.py                                          |      884 |      121 |     86% |73-78, 88-90, 161-167, 187-188, 193, 222, 230, 266, 294, 298, 339, 364, 382, 399-401, 409-411, 414, 419, 422, 426, 434-435, 451-453, 467-469, 483-485, 489, 492-494, 502-504, 512-513, 553, 570-572, 641, 687-689, 713, 715, 758, 765, 773, 806-807, 812, 849, 853, 855, 890, 895, 902, 953-954, 977-978, 983-984, 1112, 1143, 1168, 1170, 1172, 1174, 1239, 1286-1287, 1305-1307, 1312, 1436, 1449, 1451-1456, 1458, 1460, 1471, 1474, 1476, 1486, 1489, 1518, 1580-1582, 1587, 1647-1648, 1661, 1740, 1751-1752, 1760-1761, 1769-1770, 1776-1779, 1805-1806, 1933-1934, 1948, 1954, 1958-1959 |
| amdsharktank/amdsharktank/types/theta.py                                            |      377 |       57 |     85% |72, 80, 109, 140-150, 170, 182-183, 212-213, 219, 231, 234, 347-351, 414, 475-476, 480-482, 492, 500-501, 505-507, 516-517, 541, 547-548, 562-563, 580-581, 605-606, 627-628, 651, 665-667, 673, 711, 759-761, 766 |
| amdsharktank/amdsharktank/utils/\_\_init\_\_.py                                     |        1 |        0 |    100% |           |
| amdsharktank/amdsharktank/utils/attention.py                                        |       55 |       11 |     80% |124-131, 166-182 |
| amdsharktank/amdsharktank/utils/azure.py                                            |       58 |       58 |      0% |     7-122 |
| amdsharktank/amdsharktank/utils/cli.py                                              |      118 |       73 |     38% |35-39, 73-190, 201-206, 217-229, 240-272, 287-308, 319, 332-335, 345, 358-370, 379, 395-396, 398, 401-402, 414, 419-422, 431-449 |
| amdsharktank/amdsharktank/utils/create\_cache.py                                    |       13 |        4 |     69% | 20, 31-33 |
| amdsharktank/amdsharktank/utils/debugging.py                                        |       91 |       29 |     68% |46-63, 67-74, 81-83, 127, 138 |
| amdsharktank/amdsharktank/utils/e2e\_test\_utils.py                                 |       89 |       71 |     20% |32-40, 45-49, 52-70, 75-119, 122-126, 129-133 |
| amdsharktank/amdsharktank/utils/evaluate.py                                         |       59 |       34 |     42% |29-50, 57, 71-90, 121-122, 131-151 |
| amdsharktank/amdsharktank/utils/export.py                                           |       70 |        4 |     94% |140, 151, 179, 212 |
| amdsharktank/amdsharktank/utils/export\_artifacts.py                                |      177 |      130 |     27% |38-46, 60, 67, 74, 81, 88, 122-156, 167, 180-184, 210-230, 248-260, 264-270, 282-298, 322-364, 388-425, 449-462, 488-514, 536-542 |
| amdsharktank/amdsharktank/utils/functools.py                                        |       10 |        8 |     20% |     27-35 |
| amdsharktank/amdsharktank/utils/hf.py                                               |      109 |       74 |     32% |36-47, 51-53, 65-100, 115-157, 215, 237-247, 251-252 |
| amdsharktank/amdsharktank/utils/hf\_datasets.py                                     |       92 |       26 |     72% |40-43, 46-65, 68, 88, 103, 112-113, 118, 544-562, 566 |
| amdsharktank/amdsharktank/utils/io.py                                               |       39 |        9 |     77% |65-72, 83-86 |
| amdsharktank/amdsharktank/utils/iree.py                                             |      287 |       56 |     80% |187, 198-201, 299, 303, 307, 313-320, 326, 332, 338, 380, 498-499, 546-547, 555-559, 660-681, 696-703, 715-723, 746 |
| amdsharktank/amdsharktank/utils/llm\_artifacts.py                                   |       31 |        2 |     94% |    37, 42 |
| amdsharktank/amdsharktank/utils/llm\_scheduler.py                                   |       91 |        4 |     96% |43, 47, 51, 62 |
| amdsharktank/amdsharktank/utils/llm\_tasks.py                                       |      139 |        3 |     98% |57, 61, 66 |
| amdsharktank/amdsharktank/utils/llm\_utils.py                                       |      464 |      109 |     77% |114-128, 150-151, 154-157, 161-166, 217, 257-259, 269-273, 286, 556-563, 646-647, 653, 663-664, 685, 688-728, 741, 744-752, 779-782, 784-785, 787, 806, 810, 847-854, 857-870, 898-904, 926 |
| amdsharktank/amdsharktank/utils/load\_llm.py                                        |      173 |      136 |     21% |39-41, 45, 51-65, 70-82, 93-108, 120, 123, 138-162, 166, 173, 176-185, 190-197, 200-210, 220-238, 241-279, 282-352, 355-357 |
| amdsharktank/amdsharktank/utils/logging.py                                          |        6 |        1 |     83% |        17 |
| amdsharktank/amdsharktank/utils/math.py                                             |       12 |        5 |     58% | 17, 25-28 |
| amdsharktank/amdsharktank/utils/misc.py                                             |       58 |        9 |     84% |35, 100, 108-114 |
| amdsharktank/amdsharktank/utils/patching.py                                         |      137 |       46 |     66% |138, 141, 166, 203-206, 215-221, 226, 236-261, 269-285, 288-299, 328, 362, 364 |
| amdsharktank/amdsharktank/utils/random.py                                           |       38 |        0 |    100% |           |
| amdsharktank/amdsharktank/utils/testing.py                                          |      444 |      207 |     53% |104-106, 157-268, 291-297, 308, 321-331, 344-368, 374-395, 411-420, 425-438, 442-446, 485-569, 604, 657-660, 697-703, 734, 758-766, 781, 786, 792-796, 804-807, 813-820, 828-832, 900, 949, 954-968, 1009, 1045-1047, 1058, 1064, 1074, 1077, 1119 |
| amdsharktank/amdsharktank/utils/tokenizer.py                                        |       51 |       35 |     31% |34-38, 42-46, 50, 63-66, 69-72, 76, 80-81, 85-110 |
| amdsharktank/amdsharktank/utils/tree.py                                             |       71 |        2 |     97% |   81, 220 |
| amdsharktank/conftest.py                                                            |      153 |       12 |     92% |297, 304, 311, 343, 354, 378, 383-386, 415, 442-443 |
| amdsharktank/integration/models/llama/llama\_integration\_test.py                   |       32 |       22 |     31% |     42-86 |
| amdsharktank/integration/models/punet/integration\_test.py                          |       94 |       57 |     39% |15-16, 21-35, 56-70, 78-88, 98-113, 122-133, 143-155, 162, 167-179, 186, 197-204, 222-232, 254-267 |
| amdsharktank/setup.py                                                               |       18 |       18 |      0% |      7-34 |
| amdsharktank/tests/evaluate/perplexity\_iree\_test.py                               |       46 |       28 |     39% |30-36, 43-55, 58-68, 78-83, 88-98, 102 |
| amdsharktank/tests/evaluate/perplexity\_torch\_test.py                              |       38 |       22 |     42% |29-34, 37-46, 49-65, 70-75, 79 |
| amdsharktank/tests/examples/main\_test.py                                           |       24 |        1 |     96% |        45 |
| amdsharktank/tests/examples/paged\_llm\_v1\_test.py                                 |       16 |        5 |     69% |     29-33 |
| amdsharktank/tests/export\_ir/export\_test.py                                       |       37 |        0 |    100% |           |
| amdsharktank/tests/kernels/attention\_template\_test.py                             |       76 |        8 |     89% |23, 112-118, 144 |
| amdsharktank/tests/kernels/attention\_wave\_test.py                                 |       26 |        2 |     92% |    31, 65 |
| amdsharktank/tests/kernels/batch\_matmul\_transpose\_b\_test.py                     |       94 |        6 |     94% |127-130, 143, 170 |
| amdsharktank/tests/kernels/conv\_2d\_nchw\_fchw\_test.py                            |       42 |        2 |     95% |    63, 91 |
| amdsharktank/tests/kernels/einsum\_q4\_test.py                                      |       69 |        3 |     96% |94, 120, 141 |
| amdsharktank/tests/kernels/gemm\_fp4\_asm\_test.py                                  |       57 |       38 |     33% |30, 65-124 |
| amdsharktank/tests/kernels/gemm\_fp4\_test.py                                       |      112 |       40 |     64% |63, 68, 139, 142, 165-226 |
| amdsharktank/tests/kernels/mlir\_kernel\_test.py                                    |       21 |        0 |    100% |           |
| amdsharktank/tests/kernels/mmt\_block\_scaled\_offset\_q4\_test.py                  |       46 |        3 |     93% |49, 79, 100 |
| amdsharktank/tests/kernels/mmt\_block\_scaled\_q8\_test.py                          |       43 |        3 |     93% |46, 74, 94 |
| amdsharktank/tests/kernels/mmt\_super\_block\_scaled\_offset\_q4\_test.py           |       71 |       20 |     72% |39-64, 97, 156, 174 |
| amdsharktank/tests/kernels/mmtfp\_test.py                                           |       60 |        4 |     93% |57, 81, 99, 125 |
| amdsharktank/tests/kernels/pooling\_nchw\_sum\_test.py                              |       42 |        2 |     95% |    58, 78 |
| amdsharktank/tests/kernels/rotary\_test.py                                          |       18 |        0 |    100% |           |
| amdsharktank/tests/kernels/topk\_test.py                                            |       31 |        0 |    100% |           |
| amdsharktank/tests/kernels/wave/extend\_attention\_test.py                          |       90 |       56 |     38% |41, 80-209, 235-267 |
| amdsharktank/tests/kernels/wave/mxfp4\_gemm\_test.py                                |       64 |       39 |     39% |33, 66-137 |
| amdsharktank/tests/kernels/wave/wave\_utils\_test.py                                |       30 |        0 |    100% |           |
| amdsharktank/tests/layers/base\_test.py                                             |       22 |        0 |    100% |           |
| amdsharktank/tests/layers/configs\_test.py                                          |       14 |        0 |    100% |           |
| amdsharktank/tests/layers/kv\_cache\_test.py                                        |       85 |        0 |    100% |           |
| amdsharktank/tests/layers/linear\_test.py                                           |       92 |        1 |     99% |       219 |
| amdsharktank/tests/layers/mixture\_of\_experts\_block\_test.py                      |      286 |        1 |     99% |       849 |
| amdsharktank/tests/layers/mmdit\_test.py                                            |       56 |        1 |     98% |        96 |
| amdsharktank/tests/layers/paged\_llama\_attention\_block\_test.py                   |      187 |       40 |     79% |54-71, 82-164, 464 |
| amdsharktank/tests/layers/rotary\_embedding\_hf\_test.py                            |      236 |       10 |     96% |303-304, 398-405 |
| amdsharktank/tests/layers/rotary\_embedding\_test.py                                |      131 |       46 |     65% |129-199, 217-225, 229-242 |
| amdsharktank/tests/layers/sharded\_conv2d\_with\_iree\_test.py                      |       79 |        0 |    100% |           |
| amdsharktank/tests/models/clip/clip\_test.py                                        |      251 |       53 |     79% |91, 96-111, 121, 131, 209-253, 298-325, 350-384, 393, 403 |
| amdsharktank/tests/models/deepseek/test\_deepseek.py                                |       31 |        3 |     90% |     73-84 |
| amdsharktank/tests/models/flux/flux\_test.py                                        |      167 |       76 |     54% |85-87, 91-92, 128, 173-228, 237-255, 264-284, 319, 327, 334-353, 363-392, 402, 411, 421-429, 433 |
| amdsharktank/tests/models/grok/test\_grok.py                                        |       25 |        0 |    100% |           |
| amdsharktank/tests/models/llama4/llama4\_test.py                                    |       41 |        1 |     98% |        98 |
| amdsharktank/tests/models/llama4/moe\_test.py                                       |       92 |        1 |     99% |       203 |
| amdsharktank/tests/models/llama/attention\_test.py                                  |       63 |        1 |     98% |       194 |
| amdsharktank/tests/models/llama/benchmark\_amdgpu\_test.py                          |      110 |       69 |     37% |34, 37-49, 59-93, 96-106, 115-169, 194-209, 213-228, 232-251, 256-275, 282-352, 368-389, 393-411, 415 |
| amdsharktank/tests/models/llama/quantized\_theta\_test.py                           |       20 |        0 |    100% |           |
| amdsharktank/tests/models/llama/quark\_parity\_test.py                              |       55 |       40 |     27% |21-22, 29-101, 105 |
| amdsharktank/tests/models/llama/rot\_emb\_test.py                                   |       37 |        1 |     97% |        81 |
| amdsharktank/tests/models/llama/test\_llama.py                                      |      114 |       16 |     86% |187-198, 203-213, 218-231 |
| amdsharktank/tests/models/llama/toy\_llama\_test.py                                 |       85 |        1 |     99% |        41 |
| amdsharktank/tests/models/punet/resnet\_test.py                                     |       42 |        1 |     98% |        93 |
| amdsharktank/tests/models/punet/sharded\_resnet\_block\_with\_iree\_test.py         |       43 |       12 |     72% |    76-113 |
| amdsharktank/tests/models/punet/up\_down\_block\_test.py                            |       49 |        1 |     98% |       149 |
| amdsharktank/tests/models/t5/t5\_test.py                                            |      269 |       59 |     78% |80-108, 146-174, 187-221, 266, 280, 289, 298, 307, 316, 325, 435-477, 522, 531, 540, 549, 558 |
| amdsharktank/tests/models/vae/vae\_test.py                                          |      223 |      115 |     48% |85-120, 126-135, 140-149, 155-250, 276-290, 295-308, 372-376, 383-474, 575-582, 591-595, 600-605, 611 |
| amdsharktank/tests/ops/ops\_test.py                                                 |      854 |       30 |     96% |268-271, 386-392, 399-405, 412-419, 797-802, 1392 |
| amdsharktank/tests/ops/pipeline\_parallelized\_test.py                              |      153 |        4 |     97% |57, 181, 193, 203 |
| amdsharktank/tests/ops/qconv\_test.py                                               |       97 |       12 |     88% |192-228, 232 |
| amdsharktank/tests/ops/quantized\_test.py                                           |      103 |        0 |    100% |           |
| amdsharktank/tests/ops/shaping/expand\_op\_test.py                                  |       17 |        1 |     94% |        63 |
| amdsharktank/tests/ops/shaping/flatten\_op\_test.py                                 |       17 |        1 |     94% |        61 |
| amdsharktank/tests/ops/shaping/permute\_op\_test.py                                 |       17 |        1 |     94% |        59 |
| amdsharktank/tests/ops/shaping/reshape\_op\_test.py                                 |       17 |        1 |     94% |        63 |
| amdsharktank/tests/ops/shaping/squeeze\_op\_test.py                                 |       17 |        1 |     94% |        63 |
| amdsharktank/tests/ops/shaping/transpose\_op\_test.py                               |       27 |        1 |     96% |        82 |
| amdsharktank/tests/ops/shaping/unflatten\_op\_test.py                               |       17 |        1 |     94% |        61 |
| amdsharktank/tests/ops/shaping/unsqueeze\_op\_test.py                               |       17 |        1 |     94% |        64 |
| amdsharktank/tests/ops/shaping/view\_op\_test.py                                    |       25 |        1 |     96% |        94 |
| amdsharktank/tests/ops/sharded\_test.py                                             |     1403 |       20 |     99% |605-611, 693, 1978, 1981, 1985, 2008, 2012, 2186, 2195-2197, 2205, 2232, 2450 |
| amdsharktank/tests/ops/test\_attention\_ops.py                                      |       28 |        1 |     96% |       106 |
| amdsharktank/tests/pipelines/flux/flux\_pipeline\_test.py                           |       41 |       23 |     44% |25-27, 32-65, 77-121, 128, 135 |
| amdsharktank/tests/pytest\_fixtures\_test.py                                        |       19 |        0 |    100% |           |
| amdsharktank/tests/tools/amdsharktank\_test.py                                      |       19 |        0 |    100% |           |
| amdsharktank/tests/tools/convert\_dataset\_test.py                                  |       22 |        0 |    100% |           |
| amdsharktank/tests/tools/convert\_to\_json\_test.py                                 |       18 |        0 |    100% |           |
| amdsharktank/tests/tools/list\_sfaetensors\_test.py                                 |       13 |        0 |    100% |           |
| amdsharktank/tests/transforms/dataset\_transforms\_test.py                          |       34 |       25 |     26% | 23-84, 88 |
| amdsharktank/tests/types/dataset\_test.py                                           |      182 |       36 |     80% |238-258, 267-293, 302 |
| amdsharktank/tests/types/layout\_utils\_test.py                                     |       75 |        1 |     99% |       231 |
| amdsharktank/tests/types/layouts\_test.py                                           |       68 |        1 |     99% |       148 |
| amdsharktank/tests/types/quantizers\_test.py                                        |      266 |        1 |     99% |       637 |
| amdsharktank/tests/types/slice\_test.py                                             |       14 |        0 |    100% |           |
| amdsharktank/tests/types/tensors\_test.py                                           |      164 |        1 |     99% |       221 |
| amdsharktank/tests/utils/iree\_test.py                                              |       56 |        6 |     89% | 68-72, 92 |
| amdsharktank/tests/utils/llm\_basic\_scheduler\_test.py                             |       60 |        3 |     95% |31, 44, 109 |
| amdsharktank/tests/utils/llm\_chunk\_scheduler\_test.py                             |       63 |        3 |     95% |33, 46, 109 |
| amdsharktank/tests/utils/llm\_decode\_tasks\_test.py                                |       74 |        0 |    100% |           |
| amdsharktank/tests/utils/llm\_prefill\_tasks\_test.py                               |      140 |        0 |    100% |           |
| amdsharktank/tests/utils/llm\_utils\_test.py                                        |       85 |        0 |    100% |           |
| amdsharktank/tests/utils/misc\_test.py                                              |        9 |        0 |    100% |           |
| amdsharktank/tests/utils/patching\_test.py                                          |       81 |        0 |    100% |           |
| amdsharktank/tests/utils/random\_test.py                                            |       23 |        0 |    100% |           |
| amdsharktank/tests/utils/testing\_test.py                                           |      132 |        0 |    100% |           |
| amdsharktank/tests/utils/tree\_test.py                                              |       20 |        0 |    100% |           |
|                                                                           **TOTAL** | **24894** | **5570** | **78%** |           |


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