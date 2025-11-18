!matA_0 = tensor<2048x2048xf16>
!matB_0 = tensor<2048x2048xf16>
!matC_0 = tensor<2048x2048xf32>

!matC_1 = tensor<2048x2048xf32>

func.func @main(%arg0: !matA_0, %arg1: !matB_0) -> !matC_1 {
  %cst = arith.constant 0.000000e+00 : f32
  %5 = tensor.empty() : !matC_0
  %6 = linalg.fill ins(%cst : f32) outs(%5 : !matC_0) -> !matC_0
  %7 = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins(%arg0, %arg1 : !matA_0, !matB_0) outs(%6 : !matC_0) -> !matC_0
  %8 = tensor.empty() : !matC_1
  %9 = linalg.fill ins(%cst : f32) outs(%8 : !matC_1) -> !matC_1
  %10 = linalg.matmul
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d1, d2)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ]
    ins(%7, %7 : !matC_0, !matC_0) outs(%9 : !matC_1) -> !matC_1
  return %10 : !matC_1
}
