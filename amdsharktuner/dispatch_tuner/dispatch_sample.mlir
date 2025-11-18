#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<1024x1280xf16>, %arg1: tensor<1280x1280xf16>) -> tensor<1024x1280xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024x1280xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1024x1280xf32>) -> tensor<1024x1280xf32>
    %2 = linalg.matmul indexing_maps = [#map, #map1, #map2] ins(%arg0, %arg1 : tensor<1024x1280xf16>, tensor<1280x1280xf16>) outs(%1 : tensor<1024x1280xf32>) -> tensor<1024x1280xf32>
    return %2 : tensor<1024x1280xf32>
  }
}
