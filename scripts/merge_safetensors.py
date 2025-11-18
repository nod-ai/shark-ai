import glob
import sys
import safetensors.torch

if len(sys.argv) != 2:
    print(
        "Path to safetensors file is missing. Usage: python merge_safetensors.py <directory containing safetensors>"
    )
else:
    safetensorDir = sys.argv[1]
    merge_state_dict = {}
    safetensorCount = len(glob.glob(f"{safetensorDir}/*.safetensors"))
    print(
        f"Found {safetensorCount} safetensors file. Merging them to merged.safetensors"
    )
    merged_file = "merged.safetensors"
    zeroInFixedName = (5 - len(str(safetensorCount))) * "0"
    for count in range(1, safetensorCount + 1):
        zeroInModelName = (5 - len(str(count))) * "0"
        merge_state_dict.update(
            safetensors.torch.load_file(
                f"{safetensorDir}/model-{zeroInModelName}{count}-of-{zeroInFixedName}{safetensorCount}.safetensors"
            )
        )
    safetensors.torch.save_file(merge_state_dict, merged_file)
