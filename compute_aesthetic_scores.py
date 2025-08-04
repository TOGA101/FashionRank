#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对 DeepFashion-MultiModal 数据集中 *全身图像* 计算 HumanAesExpert-1B 模型的审美分数
（slow inference / MetaVoter），结果写入 <root>/scores/scores.txt
格式：<image_name> <score> 例：MEN-Tees_Tanks-id_00007466-05_7_additional.jpg 0.691

全身图像判定：
  1) images/ 下存在原始 JPG 图像 <name>.jpg
  2) keypoints/keypoints_loc.txt 与 keypoints/keypoints_vis.txt 中都含有一行首 token == <name>.jpg
  3) segm/ 下存在对应分割文件 <name>_segm.png

依赖:
  - torch, torchvision
  - transformers (HuggingFace)
  - pillow
  - tqdm

执行示例:
  python compute_aesthetic_scores.py --root /path/to/DeepFashion-MultiModal
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Set, Iterable
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from tqdm import tqdm

# ------------------- 以下函数为官方示例代码（保持核心一致） -------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(im) for im in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
# ------------------- 示例代码引用结束 -------------------


def parse_keypoints_file(fp: Path) -> Set[str]:
    """读取 keypoints_loc.txt / keypoints_vis.txt，返回出现过的图像文件名集合（行首 token）。"""
    names: Set[str] = set()
    with fp.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            names.add(line.split()[0])
    return names


def iter_full_body_images(
    images_dir: Path,
    segm_dir: Path,
    kp_loc_set: Set[str],
    kp_vis_set: Set[str],
    segm_suffix: str = "_segm",
    segm_ext: str = ".png"
) -> Iterable[Path]:
    """仅生成满足全身条件的 JPG 图像路径。"""
    valid_kp = kp_loc_set & kp_vis_set
    for img_path in images_dir.rglob("*.jpg"):
        name = img_path.name  # e.g. XXX.jpg
        if name not in valid_kp:
            continue
        segm_path = segm_dir / f"{img_path.stem}{segm_suffix}{segm_ext}"
        if segm_path.is_file():
            yield img_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute HumanAesExpert-1B MetaVoter aesthetic scores for full-body images."
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="数据集主目录（包含 images/, segm/, keypoints/）")
    parser.add_argument("--model_path", type=str, default="KwaiVGI/HumanAesExpert-1B",
                        help="HuggingFace 模型名或本地路径")
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--max_num", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None,
                        help="输出文件（默认 <root>/scores/scores.txt）")
    args = parser.parse_args()

    root = args.root.resolve()
    images_dir = root / "images"
    segm_dir = root / "segm"
    keypoints_dir = root / "keypoints"
    kp_loc_file = keypoints_dir / "keypoints_loc.txt"
    kp_vis_file = keypoints_dir / "keypoints_vis.txt"

    if args.output is None:
        out_file = root / "scores" / "scores.txt"
    else:
        out_file = args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 基础存在性检查
    for p in [images_dir, segm_dir, keypoints_dir, kp_loc_file, kp_vis_file]:
        if not p.exists():
            raise FileNotFoundError(f"缺少必要路径/文件: {p}")

    # 读取 keypoints
    kp_loc_set = parse_keypoints_file(kp_loc_file)
    kp_vis_set = parse_keypoints_file(kp_vis_file)

    # 过滤全身图像
    full_body_list = list(iter_full_body_images(images_dir, segm_dir, kp_loc_set, kp_vis_set))
    if not full_body_list:
        raise RuntimeError("未找到符合条件的全身图像。请确认 keypoints 与 segm 命名规则。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型（保持官方示例参数）
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True
    ).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    # 推理 & 写出
    with torch.inference_mode(), out_file.open("w", encoding="utf-8") as fout:
        pbar = tqdm(full_body_list, desc="Scoring (MetaVoter slow mode)", unit="img")
        for img_path in pbar:
            try:
                pixel_values = load_image(
                    str(img_path),
                    input_size=args.input_size,
                    max_num=args.max_num
                ).to(torch.float16).to(device)
                score = model.run_metavoter(tokenizer, pixel_values)
                score_val = float(score.item() if isinstance(score, torch.Tensor) else score)
                fout.write(f"{img_path.name} {score_val:.3f}\n")
            except Exception as e:
                pbar.write(f"[WARN] 跳过 {img_path.name}: {e}")

    print(f"完成，共写入 {len(full_body_list)} 条结果 -> {out_file}")

if __name__ == "__main__":
    main()
