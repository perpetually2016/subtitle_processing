# 每次运行前: source env.sh
# 设置 GPU 版 paddle 所需的 cuDNN 库路径 + 指定单卡
# (GPU 版 paddle 不自带 cuDNN,缺这步会报 Cannot load cudnn shared library)

SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"

# 单卡运行;多卡机器/Argo+k8s 调度时由编排层覆盖此变量
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 自检(可选): 应打印 cudnn ver: 8907
# python -c "import paddle; print('cudnn ver:', paddle.get_cudnn_version())"
