#!/usr/bin/env bash
# 一键创建 GPU 环境(实测:Python 3.10 + CUDA 12.x + A100)
# 用法:  bash setup_env.sh [env_name]
# 默认环境名:subtitle
set -euo pipefail

ENV_NAME="${1:-subtitle}"

echo "==> [1/5] 创建 conda 环境: $ENV_NAME (python 3.10)"
conda create -y -n "$ENV_NAME" python=3.10

# 在脚本里激活 conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "==> [2/5] 安装 GPU 版 paddle (paddle 官方 cu120 源)"
pip install paddlepaddle-gpu==2.6.2.post120 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu120/

echo "==> [3/5] 安装 OCR / 视觉依赖"
pip install paddleocr==2.8.1 opencv-python==4.9.0.80 numpy==1.26.3

echo "==> [4/5] 安装 cuDNN (paddle-gpu 不自带)"
pip install nvidia-cudnn-cu12==8.9.7.29

echo "==> [5/5] 建 libcudnn.so 软链 (paddle dlopen 找无版本号名)"
SP=$(python -c "import site; print(site.getsitepackages()[0])")
ln -sf libcudnn.so.8 "$SP/nvidia/cudnn/lib/libcudnn.so"

echo ""
echo "==> 完成。验证:"
export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
python -c "import paddle; print('paddle', paddle.__version__, '| cudnn', paddle.get_cudnn_version(), '| gpu', paddle.device.cuda.device_count())"
echo ""
echo "下次运行前执行:  conda activate $ENV_NAME && source env.sh"
