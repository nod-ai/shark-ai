# Generate golden I/O for Wan Video

 - Create a new .venv:
```shell
python -m venv .goldens_venv
source .goldens_venv/bin/activate
```
 - Install torch and flash attention:
```shell
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.4
git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention
BUILD_TARGET="rocm" MAX_JOBS=$((`nproc` - 1)) python -m pip install -v . --no-build-isolation
cd ..
```
 - Clone Wan2.1 repo and install with requirements:
```shell
git clone https://github.com/Wan-Video/Wan2.1.git
cd Wan2.1
python -m pip install -r requirements.txt .
```
 - Download model checkpoint:
```shell
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir Wan2.1-T2V-14B
```
 - Test functionality of enviroment by generating a video:
```shell
python generate_goldens.py  --task t2v-14B --size 1280*720 --ckpt_dir ./Wan2.1-T2V-14B
```
