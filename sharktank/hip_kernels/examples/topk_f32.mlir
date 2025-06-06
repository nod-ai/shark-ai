
module @module {
  util.func public @topk_k4(%arg0: tensor<64x512xf32>) -> (tensor<64x8xf32>, tensor<64x8xi32>) {
    %c0_i32 = arith.constant 0 : i32
    // %cst = arith.constant 0xFF800000 : f32
    %cst = arith.constant 0xFC00 : f32
    %0 = tensor.empty() : tensor<64x512xi32>
    %1 = tensor.empty() : tensor<64x8xf32>
    %2 = tensor.empty() : tensor<64x8xi32>
    %3 = linalg.fill ins(%cst : f32) outs(%1 : tensor<64x8xf32>) -> tensor<64x8xf32>
    %4 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<64x8xi32>) -> tensor<64x8xi32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} outs(%0 : tensor<64x512xi32>) {
    ^bb0(%out: i32):
      %7 = linalg.index 1 : index
      %8 = arith.index_cast %7 : index to i32
      linalg.yield %8 : i32
    } -> tensor<64x512xi32>
    %6:2 = iree_linalg_ext.topk dimension(1) ins(%arg0, %5 : tensor<64x512xf32>, tensor<64x512xi32>) outs(%3, %4 : tensor<64x8xf32>, tensor<64x8xi32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %7 = arith.cmpf ogt, %arg1, %arg2 : f32
      iree_linalg_ext.yield %7 : i1
    } -> tensor<64x8xf32>, tensor<64x8xi32>
    util.return %6#0, %6#1 : tensor<64x8xf32>, tensor<64x8xi32>
  }
}
