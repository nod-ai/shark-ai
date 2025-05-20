from safetensors import safe_open
import os

import re
import torch

TRACE_PATH = "./traces"

FRAMES = 1

# Map from string dtype to torch dtype
TORCH_DTYPE_MAP = {
    'f32': torch.float32,
    'f64': torch.float64,
    'f16': torch.float16,
    'bf16': torch.bfloat16,
    'i32': torch.int32,
    'i64': torch.int64,
    'i16': torch.int16,
    'i8': torch.int8,
    'u8': torch.uint8,
    'bool': torch.bool,
}

def extract_tensor_spec_and_contents(line):
    # Match everything before the first '='
    match = re.match(r'^([0-9x]+)([a-zA-Z0-9_]+)=(\[.*)$', line.strip())
    if not match:
        raise ValueError(f"Invalid tensor line: {line}")
    shape_str, dtype_str, rest = match.groups()
    return shape_str, dtype_str, rest

def extract_balanced_brackets(s):
    # Finds the full nested bracket content
    bracket_count = 0
    for i, c in enumerate(s):
        if c == '[':
            bracket_count += 1
        elif c == ']':
            bracket_count -= 1
            if bracket_count == 0:
                return s[:i + 1]
    raise ValueError("Unbalanced brackets in tensor data")

def tokenize(s):
    tokens = []
    num = ''
    for c in s:
        if c in '[]':
            if num:
                tokens.append(num)
                num = ''
            tokens.append(c)
        elif c.isspace():
            if num:
                tokens.append(num)
                num = ''
        else:
            num += c
    if num:
        tokens.append(num)
    return tokens

def parse_tokens(tokens):
    stack = []
    current = []
    for token in tokens:
        if token == '[':
            stack.append(current)
            current = []
        elif token == ']':
            if not stack:
                raise ValueError("Mismatched brackets")
            prev = stack.pop()
            prev.append(current)
            current = prev
        else:
            try:
                if '.' in token or 'e' in token.lower():
                    current.append(float(token))
                else:
                    current.append(int(token))
            except Exception:
                raise ValueError(f"Invalid number token: {token}")
    if stack:
        raise ValueError("Unbalanced brackets at end of input")
    return current[0] if current else []

def parse_tensor_line(line):
    shape_str, dtype_str, raw_contents = extract_tensor_spec_and_contents(line)
    dtype = TORCH_DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    # Parse declared shape string
    shape = list(map(int, shape_str.strip().split("x")[:-1]))
    
    # Extract and parse bracketed content
    bracketed = extract_balanced_brackets(raw_contents)
    tokens = tokenize(bracketed)
    nested_list = parse_tokens(tokens)

    # Convert to tensor and reshape to declared shape
    flat_tensor = torch.tensor(nested_list, dtype=dtype)
    reshaped_tensor = flat_tensor.reshape(shape)
    return reshaped_tensor

def parse_text_to_torch(text):
    with open(text, "r") as f:
        contents = f.read()
    lines = contents.strip().splitlines()
    results = {}
    current_trace = None

    for line in lines:
        line = line.strip()
        if line.startswith("===") and line.endswith("==="):
            current_trace = line.strip("= ").strip()
        elif "hal.buffer_view" in line:
            break
        elif "=" in line and "[" in line and current_trace:
            tensor = parse_tensor_line(line)
            results[current_trace] = tensor

    return results

def get_reference_trace(filename):
    with safe_open(filename, framework="pt") as f:
        for key in f.keys():
            tensors = f.get_tensor(key)
    return tensors


def compare_with_reference(parsed_tensors, reference_dir=".", tolerance=1e-2):
    all_passed = True
    for name, parsed_tensor in parsed_tensors.items():
        ref_path = os.path.join(reference_dir, f"{name}.safetensors")
        if not os.path.exists(ref_path):
            print(f"[MISSING] Reference file not found: {ref_path}")
            all_passed = False
            continue

        ref_tensor = get_reference_trace(ref_path)

        if parsed_tensor.shape != ref_tensor.shape:
            print(f"[FAIL] Shape mismatch in {name}: parsed {parsed_tensor.shape} vs ref {ref_tensor.shape}")
            all_passed = False
        elif parsed_tensor.dtype != ref_tensor.dtype:
            print(f"[FAIL] Dtype mismatch in {name}: parsed {parsed_tensor.dtype} vs ref {ref_tensor.dtype}")
            all_passed = False
        elif not torch.allclose(parsed_tensor, ref_tensor, rtol=tolerance, atol=tolerance):
            max_diff = (parsed_tensor - ref_tensor).abs().max().item()
            print(f"[FAIL] Value mismatch in {name} (max diff: {max_diff:.4e})")
            all_passed = False
        else:
            print(f"[PASS] {name}")
        return all_passed

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, required=True)
    args = p.parse_args()
    iree_tensors = parse_text_to_torch(args.file)
    passed = compare_with_reference(iree_tensors, reference_dir=TRACE_PATH, tolerance=1e-2)
    if passed:
        print("\n✅ All traces match references within tolerance.")
    else:
        print("\n❌ Some traces did not match.")