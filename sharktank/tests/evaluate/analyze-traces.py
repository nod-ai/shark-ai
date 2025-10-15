import safetensors.torch
import torch

# all_target_logits = [
#     "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f8_e4m3fnuz-logits/logits.safetensors",
#     "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f8_e4m3fnuz-import-preset-logits/logits.safetensors",
# ]
# all_reference_logits = [
#     "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f16-logits/logits.safetensors",
#     "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f16-logits/logits.safetensors",
# ]

# all_target_logits = [
#     "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f8_e4m3fnuz-import-preset-logits/logits.safetensors",
#     "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f16-logits/logits.safetensors",
#     "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f8_e4m3fnuz-import-preset-logits/logits.safetensors",
#     "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f8_e4m3fnuz-logits/logits.safetensors",
# ]
# all_reference_logits = [
#     "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f16-logits-import-preset-logits/logits.safetensors",
#     "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f16-logits-import-preset-logits/logits.safetensors",
#     "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f8_e4m3fnuz-logits/logits.safetensors",
#     "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f16-logits-import-preset-logits/logits.safetensors",
# ]

all_target_logits = [
    "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f8_e4m3fnuz-logits/logits.safetensors",
]
all_reference_logits = [
    "/home/bpetkant/ws/sharktank/experiments/stable-cross-entropy/f16-logits-import-preset-logits/logits.safetensors",
]


for target_logits, reference_logits in zip(all_target_logits, all_reference_logits):
    print(
        "------------------------------------------------------------------------------------------------------"
    )
    print(f"{target_logits} vs {reference_logits}")
    target_logits = safetensors.torch.load_file(target_logits)["logits"]
    reference_logits = safetensors.torch.load_file(reference_logits)["logits"]
    abs_diff = (target_logits - reference_logits).abs()
    std_dev = float(abs_diff.std())
    mean = float(abs_diff.mean())
    maximum = float(abs_diff.max())
    minimum = float(abs_diff.min())

    top10_abs_diff = torch.topk(abs_diff.flatten(), k=10)
    top10_abs_diff_values = top10_abs_diff.values.tolist()
    top10_target_values = target_logits.flatten()[top10_abs_diff.indices].tolist()
    top10_reference_values = reference_logits.flatten()[top10_abs_diff.indices].tolist()

    print(f"std_dev = {std_dev}")
    print(f"mean = {mean}")
    print(f"maximum = {maximum}")
    print(f"minimum = {minimum}")
    print(f"top10_values = {top10_abs_diff_values}")
    print(f"top10_target_values = {top10_target_values}")
    print(f"top10_reference_values = {top10_reference_values}")

    ks = [1, 3, 5]
    for k in ks:
        target_topk = torch.topk(target_logits, k=k)
        reference_topk = torch.topk(reference_logits, k=k)
        target_topk_sorted_indicies = torch.sort(target_topk.indices, dim=-1).values
        reference_topk_sorted_indicies = torch.sort(
            reference_topk.indices, dim=-1
        ).values
        topk_discrepancies = (
            target_topk_sorted_indicies != reference_topk_sorted_indicies
        )
        topk_token_discrepancies = torch.sum(topk_discrepancies, dim=-1) > 0
        total_topk_discrepancies = int(topk_token_discrepancies.sum())
        total_topk_discrepancies_percent = (
            100 * total_topk_discrepancies / target_logits.shape[0]
        )
        print(
            f"total top {k} discrepancies = {total_topk_discrepancies}, ({total_topk_discrepancies_percent}%)"
        )
