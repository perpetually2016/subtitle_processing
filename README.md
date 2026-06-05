# subtitle_processing — 视频硬字幕检测 / 裁剪流水线

基于 **PaddleOCR (PP-OCRv4 中文轻量模型)** 对视频做硬字幕检测,并将视频分类(有字幕 / 干净);
可选地用 **FFmpeg** 把带字幕的底部区域动态裁掉。

> 定位:**只做"检测有没有字幕 + 物理裁掉底部"**。
> 不做字幕内容提取(srt),也不做画面修复(inpaint)。如需这两类能力,见文末"相关方案对比"。

---

## 目录结构

```
subtitle_processing/
├── ocr_video/
│   ├── paddleocr_word.py      # 只检测分类(不剪辑):有字幕→video_word/  干净→video_clean/
│   └── ocr_ffmpeg.py          # 检测 + FFmpeg 动态裁剪:有字幕→裁底后存 video_cropped/,原片备份 video_word/
├── video/                     # 输入视频(*.mp4)
├── subtitle_manifest.json     # 输出:分类清单 {"has_subtitle":[...], "clean":[...]}
└── README.md
```

运行后自动生成:`video_word/`、`video_clean/`、`video_cropped/`(仅 `ocr_ffmpeg.py`)。

---

## 两个脚本的区别

| | `paddleocr_word.py` | `ocr_ffmpeg.py` |
|---|---|---|
| 字幕检测 | ✓ | ✓ |
| 输出 | 仅分类复制 | 分类 + **动态裁剪** |
| 裁剪 | 无 | FFmpeg `crop`,GPU `h264_nvenc` 重编码 |
| 动态边界 | — | 追踪命中字幕的最高 Y 坐标,据此算裁剪高度 |
| 用途 | 先看分类结果是否准 | 确认无误后做实际裁剪产出 |

---

## 检测算法(两脚本一致)

1. **抽帧**:`cv2.VideoCapture`,采样 `总帧数/5` 帧,夹在 `[20, 50]` 之间。
2. **逐帧 OCR**:每帧送 PaddleOCR(中文模型)识别所有文字框。
3. **三道过滤**(逐文字框):
   - 置信度 ≥ `0.5`、字数 ≥ `2`、且**非纯数字**(滤条形码/编号);
   - **横向居中**:文字框中心偏离画面中轴 ≤ `宽×0.25`(滤两侧水印/台标);
   - **纵向靠下**:文字框底部必须落在画面**底部 35%** 区域内(滤顶部标题)。
4. **命中率判定**:命中帧数 / 采样帧数 ≥ `0.12` → 判定"有字幕"。
5. **(仅 ocr_ffmpeg.py)动态裁剪**:记录所有命中字幕里**最高的顶端 Y**,据此算裁剪高度
   并上切 `6px` 余量;用 `SUBTITLE_REGION_RATIO` 兜底防止切过头
   (`ocr_ffmpeg.py` 默认 `0.30`,`paddleocr_word.py` 默认 `0.35`)。

> **关于 `CENTER_BIAS_RATIO = 0.25` 的设计取舍**:该参数假设字幕位于画面中央区域。
> - **好处**:能有效过滤边缘的水印、台标,避免误判。
> - **坏处**:对**确实在左下角 / 右下角等边缘**的字幕会漏检。
>
> 若你的片源字幕常在边缘,可调大此值(放宽居中约束)或针对性关闭该过滤。

---

## 环境部署(实测可复现)

> 实测机器:Ubuntu / Linux 5.15、NVIDIA A100-80GB、CUDA Driver 12.8、FFmpeg 4.4.2(自带 `h264_nvenc`)。
> 脚本写死 `use_gpu=True`,**默认走 GPU**。下面分 GPU / CPU 两套说明。

### 公共前置

- **FFmpeg**:仅 `ocr_ffmpeg.py` 需要;若要 GPU 加速裁剪,FFmpeg 须带 `h264_nvenc`(`ffmpeg -encoders | grep nvenc` 可验证)。
- **OCR 权重**:首次运行**自动联网下载** PP-OCRv4 中文轻量模型(det 4.9MB + rec 11MB + cls 2.2MB ≈ **16MB**)到 `~/.paddleocr/`,无需手动准备。

### 环境文件一览

| 文件 | 用途 |
|---|---|
| `setup_env.sh` | **一键**创建 GPU conda 环境(含 cuDNN + 软链),推荐 |
| `requirements-gpu.txt` | GPU 依赖清单(实测版本) |
| `requirements-cpu.txt` | CPU 依赖清单 |
| `env.sh` | **每次运行前** `source`,设置 `LD_LIBRARY_PATH` + `CUDA_VISIBLE_DEVICES` |

### 方式 A:GPU(推荐,A100)

**一键安装(推荐)**:

```bash
bash setup_env.sh            # 默认环境名 subtitle
# 或自定义环境名: bash setup_env.sh myenv
```

成功会打印类似 `paddle 2.6.2 | cudnn 8907 | gpu 8`。之后每次运行前:`conda activate subtitle && source env.sh`。

**手动安装(等价于上面脚本,逐步排查用)**:

```bash
# 1) 新建专用 conda 环境(py3.10 对 paddle 2.6 兼容最好)
conda create -y -n subtitle python=3.10
conda activate subtitle

# 2) 装 GPU 版 paddle(从 paddle 官方 cu120 源,适配 CUDA 12.x)
pip install paddlepaddle-gpu==2.6.2.post120 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu120/

# 3) 装 OCR / 视觉依赖(numpy 必须 <2,故钉 1.26.3)
pip install paddleocr==2.8.1 opencv-python==4.9.0.80 numpy==1.26.3

# 4) ⚠️ 关键:GPU 版 paddle 不自带 cuDNN,必须单独装
pip install nvidia-cudnn-cu12==8.9.7.29

# 5) ⚠️ 关键:pip 装的 cuDNN 只给 libcudnn.so.8,缺无版本号软链,paddle dlopen 会找不到
SP=$(python -c "import site; print(site.getsitepackages()[0])")
ln -sf libcudnn.so.8 "$SP/nvidia/cudnn/lib/libcudnn.so"
```

**每次运行前**都要把 cuDNN 库挂到 `LD_LIBRARY_PATH`(否则报 `Cannot load cudnn shared library`):

```bash
conda activate subtitle
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SP/nvidia/cudnn/lib:$SP/nvidia/cublas/lib:$SP/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"

# 指定单卡(多卡机器上推荐;配合 Argo/k8s 调度时由编排层注入)
export CUDA_VISIBLE_DEVICES=0

# 自检:应打印 cudnn ver: 8907
python -c "import paddle; print('cudnn ver:', paddle.get_cudnn_version())"
```

> 建议把上面 4 行 export 写进一个 `env.sh`,每次 `source env.sh` 即可。

### 方式 B:CPU(无显卡 / 调试)

CPU 无需 cuDNN,但脚本里写死了 `use_gpu=True`,需改成 `False`:

```bash
conda create -y -n subtitle-cpu python=3.10
conda activate subtitle-cpu
pip install -r requirements-cpu.txt
# 把脚本里 PaddleOCR(... use_gpu=True ...) 改为 use_gpu=False
```

> CPU 能跑,但逐帧推理比 GPU 慢约 5~10×;本任务帧数少,可接受但不推荐生产。

---

## 运行

```bash
conda activate subtitle
source env.sh            # 见上,设置 LD_LIBRARY_PATH + CUDA_VISIBLE_DEVICES

cd ocr_video

# 只检测分类(先验证算法是否准)
python paddleocr_word.py

# 检测 + 动态裁剪(确认无误后做实际产出)
python ocr_ffmpeg.py
```

### 可调参数(环境变量注入,无需改代码)

| 变量 | 默认 | 含义 |
|---|---|---|
| `VIDEO_INPUT_DIR` | `../video` | 输入目录(仅 `ocr_ffmpeg.py` 支持) |
| `VIDEO_OUTPUT_DIR` | 仓库根 | 输出根目录(仅 `ocr_ffmpeg.py` 支持) |
| `OCR_CONFIDENCE_THRESHOLD` | `0.5` | OCR 置信度阈值 |
| `SUBTITLE_REGION_RATIO` | `0.30`(ffmpeg)/ `0.35`(word) | 底部字幕区域占比 / 裁剪兜底比例 |
| `USE_GPU_FFMPEG` | `True`(代码内) | FFmpeg 裁剪是否用 `h264_nvenc`;无 NVENC 改 `False` 走 `libx264` |

---

## 实测 Smoke 结果(4 个样例视频)

```
视频数量: 4
[1/4] 2qX2bjTKTJ4+26997+27094.mp4        🟢 无字幕 (2.29s)
[2/4] 2W0Poln0U8c+172819+172998.mp4      🟢 无字幕 (6.87s)
[3/4] 1PWd6SuN2hk+31781+31986.mp4        🟢 无字幕 (6.83s)
[4/4] 2W0Poln0U8c+53179+53269.mp4        🟢 无字幕 (2.70s)
完成 | 有字幕:0  无字幕:4 | 总耗时 18.73s
```

结果与仓库自带 `subtitle_manifest.json` 一致(4 个均判为 clean),算法行为可复现。

---

## 常见问题

| 报错 / 现象 | 原因 | 解决 |
|---|---|---|
| `Cannot load cudnn shared library` | 没装 cuDNN 或没建 `libcudnn.so` 软链 / 没设 `LD_LIBRARY_PATH` | 见环境部署步骤 4/5 + 运行前 export |
| `Could not find a version ... paddlepaddle-gpu==2.6.2` | 官方源里是 `2.6.2.post120` | 用 `2.6.2.post120` |
| numpy 报 ABI / `numpy.dtype size changed` | numpy 2.x 与 paddleocr 2.x 不兼容 | 钉 `numpy==1.26.3` |
| FFmpeg `Unknown encoder h264_nvenc` | FFmpeg 未编译 NVENC | 改 `USE_GPU_FFMPEG=False` 走 CPU `libx264` |

---

## 单卡吞吐优化 Plan(暂不实施,仅记录)

> 目标:**把单张 A100 跑满**(多卡由 Argo + k8s 调度,不在脚本内做)。
> 现状:跑 smoke 时 GPU 利用率个位数、显存数百 MB(80GB 卡几乎空转)。
> 瓶颈**不在算力**,在于喂数据方式:逐帧 CPU seek 解码 + `batch=1` OCR + 视频串行。

| 优化项 | 当前 | 目标 | 预期收益 | 改动量 |
|---|---|---|---|---|
| **批量推理** 攒帧成 batch 一次喂 OCR | batch=1 | batch 32/64 | **吞吐 ↑5~20×**(A100 跑 1 帧≈跑 64 帧) | 中 |
| **先 crop 再 OCR** 只送底部 35% | 整帧推理 | 裁后小图 | 算力/显存 **↓~65%** | 小 |
| **GPU 解码** decord/NVDEC 替代 cv2 seek | CPU 单线程 seek | 显卡解码 | 消除 IO 瓶颈,管道跟上算力 | 中 |
| **流水线并发** 解码/OCR/写盘三段队列 | 全串行 | 生产者-消费者 | GPU 不停顿,利用率 ↑ | 中 |
| **TensorRT + FP16** paddle inference | FP32 原生 | TRT 半精度 | 用上 Tensor Core,再 ↑ | 中 |

**当前 vs 目标(单卡)**

- 当前:GPU 利用率 < 10%,4 个短视频 18.73s,大部分时间在解码/Python 开销。
- 目标:GPU 利用率 70%+,显存吃到合理水位(大 batch),单卡吞吐数量级提升。

> 注:本任务"轻量 OCR + 重 IO",即便优化到位,瓶颈也可能从 GPU 转到磁盘/解码;
> "跑满单卡"靠 batch化 + GPU解码 + 先crop,**整机产能靠 Argo/k8s 多卡分片**(已规划,不在本仓库实现)。

---

## 模型升级 Plan:PP-OCRv4 → PP-OCRv5(暂不实施,仅记录)

> 现状:本仓库锁定 `paddleocr==2.8.1`,实测下载的是 **PP-OCRv4** 模型(det/rec v4 + cls v2.0)。
> 这是当前主线的**上一代**:PaddleOCR 已到 3.x(最新 3.6),主线模型为 **PP-OCRv5**。

### 为什么现在是旧的

不是有意选旧,而是**被版本号锁定的副作用**:`PP-OCRv5 仅在 paddleocr 3.x 提供`,而仓库钉的 2.8.1 属于 2.x 时代,其能拿到的最高就是 v4。

> 对"判断底部有无字幕"这个**二分类**任务而言,v4 已够用(只需检测到文字框 + 置信度,不依赖 v5 的识别精度),所以未升级也能稳定跑通。

### PP-OCRv5 相比 v4 的收益(官方数据)

| 维度 | 提升 |
|---|---|
| 端到端精度 | 多场景复杂评测集 **+13 个百分点** |
| 难场景检测 | 手写、古籍、竖排、生僻字、日文显著增强 |
| 文字类型 | 单模型覆盖 简中/拼音/繁中/英/日 5 大类 |
| 模型档位 | 同样有 mobile(轻量)/server(高精)两档 |

对本任务的实际意义:**字幕字体花哨 / 半透明 / 描边时,漏检与误检更少**;同时与本机 VSR(已用 paddleocr 3.x)**技术栈统一**,省去维护两套 paddle 环境。

### 升级代价 ⚠️(不是改版本号就行)

`paddleocr 2.x → 3.x` 是**破坏性 API 升级**,以下写法在 3.x 会报错,需改适配代码(约十余行):

| 旧 (2.8.1) | 新 (3.x) |
|---|---|
| `PaddleOCR(use_angle_cls=True, use_gpu=True, show_log=False)` | 参数改名/移除(`use_gpu`→`device`、`show_log` 删除等) |
| `ocr.ocr(img, cls=True)` | 调用方式与返回结构改变,需重写结果解析 |
| `numpy==1.26.3` (必须 <2) | 3.x 配 `numpy 2.x`,同时需升级依赖矩阵 |

### 实施步骤(待批准后执行)

1. 新建并行环境 `subtitle-v5`(不动现有 `subtitle`),装 `paddleocr>=3.4 paddlepaddle-gpu(3.x)`。
2. 改造 `paddleocr_word.py` / `ocr_ffmpeg.py` 的 OCR 初始化与结果解析,适配 3.x API(选 `PP-OCRv5_mobile` 轻量档保持速度)。
3. 在现有 4 个样例视频上做**回归对比**:确认分类结果与 v4 一致或更优,记录耗时差异。
4. 通过后更新 `requirements-*.txt` / `setup_env.sh` / 本 README,切换默认。

> 决策建议:**产能优先、求稳** → 维持 v4;**统一栈 + 提升鲁棒性(尤其与 VSR 同上生产)** → 升 v5。

---

## 相关方案对比(若需求变化)

| 需求 | 推荐方案 |
|---|---|
| **去字幕、保留画面**(而非裁掉底部) | [video-subtitle-remover (VSR)](https://github.com/YaoFANGUK/video-subtitle-remover) — AI inpaint 填充修复,无损分辨率 |
| **提取字幕文本生成 srt** | [video-subtitle-extractor (VSE)](https://github.com/YaoFANGUK/video-subtitle-extractor) — 关键帧检测+识别+去重,支持 87 语言 |
| **本仓库** | 检测有无字幕 + 物理裁掉底部 |
