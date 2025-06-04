// iree-compile linalg_topk.mlir --iree-hal-target-device=hip --iree-hip-target=gfx942 --iree-hal-executable-object-search-path=~/iree-build --iree-preprocessing-transform-spec-filename=spec_topk.mlir -o linalg_topk_hip.vmfb
// iree-benchmark-module --module=linalg_topk_hip.vmfb --function=topk_k4 --device=hip://4 --input=8x1x131072xf32
// iree-run-module --module=linalg_topk_hip.vmfb --function=topk_k4 --device=hip://4 --input=@input0.npy

module @module {
  util.func public @topk_k4(%arg0: tensor<8x1x131072xf32>) -> (tensor<8x1x4xf32>, tensor<8x1x4xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0xFF800000 : f32
    %0 = tensor.empty() : tensor<8x1x131072xi32>
    %1 = tensor.empty() : tensor<8x1x4xf32>
    %2 = tensor.empty() : tensor<8x1x4xi32>
    %3 = linalg.fill ins(%cst : f32) outs(%1 : tensor<8x1x4xf32>) -> tensor<8x1x4xf32>
    %4 = linalg.fill ins(%c0_i32 : i32) outs(%2 : tensor<8x1x4xi32>) -> tensor<8x1x4xi32>
    %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%0 : tensor<8x1x131072xi32>) {
    ^bb0(%out: i32):
      %7 = linalg.index 2 : index
      %8 = arith.index_cast %7 : index to i32
      linalg.yield %8 : i32
    } -> tensor<8x1x131072xi32>
    %6:2 = iree_linalg_ext.topk dimension(2) ins(%arg0, %5 : tensor<8x1x131072xf32>, tensor<8x1x131072xi32>) outs(%3, %4 : tensor<8x1x4xf32>, tensor<8x1x4xi32>) {
    ^bb0(%arg1: f32, %arg2: f32):
      %7 = arith.cmpf ogt, %arg1, %arg2 : f32
      iree_linalg_ext.yield %7 : i1
    } -> tensor<8x1x4xf32>, tensor<8x1x4xi32>
    util.return %6#0, %6#1 : tensor<8x1x4xf32>, tensor<8x1x4xi32>
  }
}
