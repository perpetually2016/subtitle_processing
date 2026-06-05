#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: 视频字幕检测（不剪辑）
只做筛选 + 分类
"""

import os
import json
import time
import cv2
from pathlib import Path

# =========================
# 路径
# =========================

VIDEO_DIR = Path(__file__).parent.parent / "video"
VIDEO_WORD_DIR = Path(__file__).parent.parent / "video_word"
VIDEO_CLEAN_DIR = Path(__file__).parent.parent / "video_clean"
MANIFEST_PATH = Path(__file__).parent.parent / "subtitle_manifest.json"

VIDEO_WORD_DIR.mkdir(exist_ok=True)
VIDEO_CLEAN_DIR.mkdir(exist_ok=True)

# =========================
# 参数
# =========================

SAMPLE_FRAMES = int(os.getenv('OCR_SAMPLE_FRAMES', '15'))
CONFIDENCE_THRESHOLD = float(os.getenv('OCR_CONFIDENCE_THRESHOLD', '0.5'))
SUBTITLE_REGION_RATIO = float(os.getenv('SUBTITLE_REGION_RATIO', '0.35'))

# ✨ 微调优化：新增两个辅助过滤参数
# CENTER_BIAS_RATIO: 允许文本中心点偏离视频中轴线的最大比例（0.25表示限制在中间50%的区域内）。过滤两边角落的水印。
CENTER_BIAS_RATIO = 0.25
# MIN_HIT_RATIO: 命中字幕的帧数占总采样帧数的最小比例。
# 如果采样了30帧，只有1~2帧有字，大概率是背景杂质或偶尔闪过的值；真正有字幕的解说视频通常至少有 10%~15% 以上的帧有字。
MIN_HIT_RATIO = 0.12

# =========================
# OCR
# =========================

print("初始化 PaddleOCR...")
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ch',
    use_gpu=True,
    show_log=False
)


# =========================
# 视频信息
# =========================

def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, n


# =========================
# 是否有字幕
# =========================

def has_subtitle(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return False

    w, h, total_frames = get_video_info(video_path)

    if total_frames == 0:
        cap.release()
        return False

    sample_frames = min(
        max(20, total_frames // 5),
        50
    )

    sample_interval = max(
        1,
        total_frames // sample_frames
    )

    bottom_region_start = int(
        h * (1 - SUBTITLE_REGION_RATIO)
    )

    hit_frames = 0
    video_center_x = w / 2  # ✨ 微调优化：获取视频横向中心线

    for i in range(sample_frames):

        frame_idx = min(
            i * sample_interval,
            total_frames - 1
        )

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx
        )

        ret, frame = cap.read()

        if not ret:
            continue

        result = ocr.ocr(
            frame,
            cls=True
        )

        if not result or not result[0]:
            continue

        frame_hit = False

        for line in result[0]:

            coords = line[0]

            text = (
                line[1][0] or ""
            ).strip()

            conf = line[1][1]

            if conf < 0.5:
                continue

            if len(text) < 2:
                continue

            # ✨ 微调优化：过滤纯数字（解决 116.mp4 笔杆条形码问题）
            if text.isdigit():
                continue

            # ✨ 微调优化：横向居中校验（解决 112、113 等两旁角落常驻水印问题）
            x_coords = [p[0] for p in coords]
            box_left, box_right = min(x_coords), max(x_coords)
            box_center_x = (box_left + box_right) / 2
            if abs(box_center_x - video_center_x) > (w * CENTER_BIAS_RATIO):
                continue

            y_coords = [
                p[1]
                for p in coords
            ]

            box_bottom = max(y_coords)

            if box_bottom < bottom_region_start:
                continue

            frame_hit = True
            break

        if frame_hit:
            hit_frames += 1

        # ✨ 微调优化：删除原先的“仅命中2帧就提前退出”的暴利截断逻辑。
        # 必须走完大体循环，用“命中率”来做最终判定，才能对抗偶尔出现的背景杂质。

    cap.release()

    # ✨ 微调优化：通过最终的命中帧数比例来决定是否有字幕
    actual_ratio = hit_frames / sample_frames
    if actual_ratio >= MIN_HIT_RATIO:
        return True

    return False

def main():
    start = time.time()

    videos = list(VIDEO_DIR.glob("*.mp4"))

    manifest = {
        "has_subtitle": [],
        "clean": []
    }

    print(f"视频数量: {len(videos)}")

    for i, v in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] 检测: {v.name}")

        t0 = time.time()
        flag = has_subtitle(v)
        cost = time.time() - t0

        if flag:
            dst = VIDEO_WORD_DIR / v.name
            manifest["has_subtitle"].append(v.name)
            print(f"  🟠 有字幕 -> video_word ({cost:.2f}s)")
        else:
            dst = VIDEO_CLEAN_DIR / v.name
            manifest["clean"].append(v.name)
            print(f"  🟢 无字幕 -> video_clean ({cost:.2f}s)")

        # 只复制分类，不剪辑
        try:
            dst.write_bytes(v.read_bytes())
        except Exception:
            import shutil
            shutil.copy2(v, dst)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n====================")
    print("完成")
    print(f"有字幕: {len(manifest['has_subtitle'])}")
    print(f"无字幕: {len(manifest['clean'])}")
    print(f"总耗时: {time.time() - start:.2f}s")
    print(f"manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()