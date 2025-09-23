"""
Tests Shortfin Online Serving through prompt and response.
"""
import argparse
import os
import subprocess
import sys
import time
import requests
import signal
import re

def wait_for_server(port, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            time.sleep(2)
    return False

def main():

    defaults = {
        "irpa": "/sharedfile/attn/fp8_attn.irpa",
        "tokenizer_json": "/shark-dev/8b/instruct/tokenizer.json",
        "vmfb": os.path.join(os.getcwd(), "../output_artifacts/output.vmfb"),
        "model_config": os.path.join(
            os.getcwd(), "../output_artifacts/config_attn.json"
        ),
        "port": 8959,
        "tensor_parallelism_size": "1",
    }

    parser = argparse.ArgumentParser(description="Start server and run client")
    parser.add_argument("--irpa", default=defaults["irpa"], help="Path to IRPA file")
    parser.add_argument(
        "--tokenizer_json",
        default=defaults["tokenizer_json"],
        help="Tokenizer JSON path",
    )
    parser.add_argument("--vmfb", default=defaults["vmfb"], help="VMFB file path")
    parser.add_argument(
        "--model_config",
        default=defaults["model_config"],
        help="Model config JSON path",
    )
    parser.add_argument(
        "--port", type=int, default=defaults["port"], help="Port number for server"
    )


    output_dir = os.path.join(os.getcwd(), "../output_artifacts")
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "online_serving.log")

    print("Running server ...")

    server_cmd = [
        sys.executable,
        "-m",
        "shortfin_apps.llm.server",
        f"--tokenizer_json={args.tokenizer_json}",
        f"--model_config={args.model_config}",
        f"--vmfb={args.vmfb}",
        f"--parameters={args.irpa}",
        "--device=hip",
        "--device_ids", "0",
        "--port", str(args.port)
    ]
    server_proc = subprocess.Popen(server_cmd)

    if not wait_for_server(args.port):
        print("Failed to start the server")
        server_proc.kill()
        sys.exit(1)

    print(
        f"Server with PID {server_proc.pid} is ready to accept requests on port {args.port}..."
    )

    print("Running Client ...")
    start_time = time.time()

    try:
        response = requests.post(
            f"http://localhost:{args.port}/generate",
            headers={"Content-Type": "application/json"},
            json={
                "text": "<|begin_of_text|>Name the capital of the United States.<|eot_id|>",
                "sampling_params": {"max_completion_tokens": 50},
            },
            timeout=30,
        )
        with open(log_file, "w") as f:
            f.write(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Client request failed: {e}")
        server_proc.kill()
        sys.exit(1)

    end_time = time.time()
    time_taken = int(end_time - start_time)
    with open(log_file, "a") as f:
        f.write(f"\nTime Taken for Getting Response: {time_taken} seconds\n")

    time.sleep(10)
    os.kill(server_proc.pid, signal.SIGKILL)

    if not os.path.exists(log_file):
        print(f"The file '{log_file}' does NOT exist.")
        sys.exit(1)
    else:
        print(f"The file '{log_file}' exists.")

    with open(log_file, "r") as f:
        content = f.read()

    expected = "\"responses\": [{\"text\": \"assistant\\nThe capital of the United States is Washington, D.C.\"}]"
    if expected in content:
        print("[SUCCESS] Online Response Matches Expected Output.")
    elif re.search(
        r'"text": ".*washington(,?\s*d\.?c\.?)?"', content, flags=re.IGNORECASE
    ):
        print("[CHECK REQUIRED] Partially Correct Response Detected.")
        print(content)
        sys.exit(1)
    else:
        print("[FAILURE] Gibberish or Invalid Response Detected.")
        print(content)
        sys.exit(1)


if __name__ == "__main__":
    main()
