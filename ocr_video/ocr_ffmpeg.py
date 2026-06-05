#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: 视频字幕检测与智能动态裁剪（生产环境高能版）
1. 100% 保留原生过滤算法，增加最高字幕位置的动态追踪。
2. FFmpeg 放弃一刀切，根据每条视频的字幕实际高度动态裁剪。
3. 支持 Nvidia GPU 硬件加速重编码，大幅提升剪辑速度。
"""

import os
import json
import time
import cv2
import subprocess
from pathlib import Path

# =====================================================================
# 🎛️ 生产环境硬件配置
# =====================================================================
# 如果你的机器带显卡（Nvidia），请保持 True，FFmpeg 剪辑速度会提升数倍
# 如果报错提示找不到 nvenc 编码器，请将其改为 False，会自动切回 CPU 编码
USE_GPU_FFMPEG = True

# =====================================================================
# 📂 路径配置（支持生产环境自定义注入）
# =====================================================================
VIDEO_DIR = Path(os.getenv('VIDEO_INPUT_DIR', Path(__file__).parent.parent / "video"))
OUTPUT_BASE_DIR = Path(os.getenv('VIDEO_OUTPUT_DIR', Path(__file__).parent.parent))

VIDEO_WORD_DIR = OUTPUT_BASE_DIR / "video_word"
VIDEO_CLEAN_DIR = OUTPUT_BASE_DIR / "video_clean"
VIDEO_CROPPED_DIR = OUTPUT_BASE_DIR / "video_cropped"
MANIFEST_PATH = OUTPUT_BASE_DIR / "subtitle_manifest.json"

VIDEO_WORD_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_CROPPED_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================================
# 参数（100% 继承自你的原生算法参数）
# =====================================================================
SAMPLE_FRAMES = int(os.getenv('OCR_SAMPLE_FRAMES', '15'))
CONFIDENCE_THRESHOLD = float(os.getenv('OCR_CONFIDENCE_THRESHOLD', '0.5'))
SUBTITLE_REGION_RATIO = float(os.getenv('SUBTITLE_REGION_RATIO', '0.30'))

CENTER_BIAS_RATIO = 0.25
MIN_HIT_RATIO = 0.12

# 裁剪安全缓冲区（像素）：在动态算出的字幕最高点之上，再往上多切 6 像素，确保把字幕毛边、边框彻底切干净
CROP_MARGIN = 6

# =====================================================================
# OCR 初始化
# =====================================================================
print("初始化 PaddleOCR...")
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ch',
    use_gpu=True,
    show_log=False
)


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, n


# =====================================================================
# 🛠️ FFmpeg 动态裁剪与流复制工具
# =====================================================================

def ffmpeg_dynamic_crop(input_path, output_path, width, crop_height):
    """根据动态计算出的精确保留高度，调用 FFmpeg 进行裁剪"""
    # 确保高度为偶数（H.264 编码器的硬性要求）
    if crop_height % 2 != 0:
        crop_height -= 1

    # 根据配置选择编码器：GPU 硬件加速 or CPU 编码
    video_codec = 'h264_nvenc' if USE_GPU_FFMPEG else 'libx264'

    cmd = [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-vf', f'crop={width}:{crop_height}:0:0',  # 保留从顶部(0,0)到 crop_height 的区域
        '-c:v', video_codec,
    ]

    # 针对不同编码器微调参数，确保速度与画质平衡
    if USE_GPU_FFMPEG:
        cmd.extend(['-preset', 'fast', '-cq', '22'])
    else:
        cmd.extend(['-preset', 'fast', '-crf', '20', '-threads', '4'])

    cmd.extend(['-c:a', 'copy', str(output_path)])

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ FFmpeg 智能裁剪失败: {e.stderr[-300:]}")
        return False


def ffmpeg_copy_video(input_path, output_path):
    """无损极速复制分流（秒级完成）"""
    cmd = ['ffmpeg', '-y', '-i', str(input_path), '-c', 'copy', str(output_path)]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ FFmpeg 复制失败: {e.stderr[-300:]}")
        return False


# =====================================================================
# 是否有字幕（升级版：完美继承检测算法，同时动态捕捉字幕最高位置）
# =====================================================================
def detect_subtitle_and_position(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, SUBTITLE_REGION_RATIO

    w, h, total_frames = get_video_info(video_path)
    if total_frames == 0:
        cap.release()
        cap.release()
        return False, SUBTITLE_REGION_RATIO

    sample_frames = min(max(20, total_frames // 5), 50)
    sample_interval = max(1, total_frames // sample_frames)
    bottom_region_start = int(h * (1 - SUBTITLE_REGION_RATIO))

    hit_frames = 0
    video_center_x = w / 2

    # ✨ 核心智能化：记录所有有效字幕中，出现过的最靠上的 Y 坐标（初始值为画面最底部）
    highest_text_top_y = h

    for i in range(sample_frames):
        frame_idx = min(i * sample_interval, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        result = ocr.ocr(frame, cls=True)
        if not result or not result[0]:
            continue

        frame_hit = False

        for line in result[0]:
            coords = line[0]
            text = (line[1][0] or "").strip()
            conf = line[1][1]

            if conf < 0.5 or len(text) < 2 or text.isdigit():
                continue

            # 横向居中校验
            x_coords = [p[0] for p in coords]
            box_left, box_right = min(x_coords), max(x_coords)
            box_center_x = (box_left + box_right) / 2
            if abs(box_center_x - video_center_x) > (w * CENTER_BIAS_RATIO):
                continue

            y_coords = [p[1] for p in coords]
            box_top = min(y_coords)  # 文字块的最顶端 Y 坐标
            box_bottom = max(y_coords)  # 文字块的最底端 Y 坐标

            if box_bottom < bottom_region_start:
                continue

            frame_hit = True

            # ✨ 核心智能化：如果该有效字幕的顶端比历史记录更靠上（Y值更小），则刷新最高点
            if box_top < highest_text_top_y:
                highest_text_top_y = box_top

        if frame_hit:
            hit_frames += 1

    cap.release()

    actual_ratio = hit_frames / sample_frames
    if actual_ratio >= MIN_HIT_RATIO:
        # 计算需要切掉的实际高度比例
        # 如果动态抓到的字幕最高点太高，为了防止切过头破坏主画面，用预设的 SUBTITLE_REGION_RATIO 兜底
        max_allowed_crop_h = int(h * SUBTITLE_REGION_RATIO)
        actual_crop_h = h - highest_text_top_y + CROP_MARGIN

        final_crop_h = min(actual_crop_h, max_allowed_crop_h)
        dynamic_remove_ratio = final_crop_h / h

        return True, dynamic_remove_ratio

    return False, SUBTITLE_REGION_RATIO


# =====================================================================
# 主控制流
# =====================================================================
def main():
    start = time.time()

    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
    except FileNotFoundError:
        print("❌ 错误: 系统未检测到 FFmpeg 环境！")
        return

    videos = list(VIDEO_DIR.glob("*.mp4"))
    manifest = {"has_subtitle": [], "clean": []}

    print(f"视频数量: {len(videos)} | FFmpeg GPU 加速: {USE_GPU_FFMPEG}")

    for i, v in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] 检测: {v.name}")

        w, h, _ = get_video_info(v)
        t0 = time.time()

        # ✨ 调用升级后的检测函数，获取是否命中以及量身定制的动态裁剪比例
        flag, dynamic_ratio = detect_subtitle_and_position(v)
        cost = time.time() - t0

        if flag:
            manifest["has_subtitle"].append(v.name)

            # 计算动态保留高度
            precise_crop_height = int(h * (1 - dynamic_ratio))

            print(
                f"  🟠 有字幕 -> 智能识别字幕边界：建议切掉底部 {int(dynamic_ratio * 100)}% (保留上方 {precise_crop_height}px)")

            dst_cropped = VIDEO_CROPPED_DIR / f"{v.stem}_cropped{v.suffix}"
            crop_success = ffmpeg_dynamic_crop(v, dst_cropped, w, precise_crop_height)

            if crop_success:
                print(f"    ✅ [智能裁剪成功] -> video_cropped/{dst_cropped.name}")
            else:
                print(f"    ❌ [智能裁剪错误]")

            # 无损备份原视频
            dst_word = VIDEO_WORD_DIR / v.name
            ffmpeg_copy_video(v, dst_word)
        else:
            manifest["clean"].append(v.name)
            print(f"  🟢 无字幕 -> FFmpeg 无损极速分流 ({cost:.2f}s)")
            dst_clean = VIDEO_CLEAN_DIR / v.name
            ffmpeg_copy_video(v, dst_clean)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n====================")
    print("完成")
    print(f"有字幕（已动态裁剪）: {len(manifest['has_subtitle'])}")
    print(f"无字幕（已极速分流）: {len(manifest['clean'])}")
    print(f"总耗时: {time.time() - start:.2f}s")
    print(f"manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()