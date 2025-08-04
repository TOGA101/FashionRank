#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepFashion-MultiModal 图像裁剪 + Marqo-FashionSigLIP 特征提取脚本

功能：
1) images/*.jpg -> (segm/*.png 或 segformer 推理) -> 裁剪四类部位
2) 使用 Hugging Face & PyTorch（GPU 可用则自动启用）提取部位特征
3) 保存到 feat/<原文件名>_feat.pt，字典键：["upper body clothes", "lower body clothes", "whole body clothes", "shoes"]

注意：
- 完全遵循官方 Quick Start：Marqo-FashionSigLIP 与 SegFormer 的加载/推理写法均直接采用官方示例的范式。
- SegFormer 与 DeepFashion-MultiModal 的标签定义不同，已使用各自的官方 id->label 映射组合成四类目标部位。
- 若检测到 whole body clothes（例如连衣裙/连体衣），按照需求不再输出 upper/lower。

TODO（潜在扩展，当前实现不做）：
- 如需“抠图式裁剪”（背景透明化），可用分割 mask 作为 alpha；本文按需求采用紧致外接矩形 crop，保持流程简洁。
- DeepFashion-MultiModal 的 segm 文件夹名称在官方为 `parsing/`，你的数据已放在 `segm/`，本脚本按你的目录命名使用。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from tqdm import tqdm

# ---- Hugging Face: 官方示例范式（遵循示例写法） ---------------------------------
from transformers import AutoModel, AutoProcessor  # Marqo-FashionSigLIP
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation  # SegFormer

# ------------------------- 路径与设备 ------------------------------------------
ROOT = Path(".").resolve()  # 主目录（含 images/, segm/, keypoints/ 等）
IMAGES_DIR = ROOT / "images"
SEGM_DIR = ROOT / "segm"      # 你当前数据的解析标签目录（按你提供的命名）
FEAT_DIR = ROOT / "feat"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------- 标签到四类部位的映射（两种来源分开） ----------------------
# DeepFashion-MultiModal（24 类）：来源于官方 README「Human Parsing Label」
#   0:bg, 1:top, 2:outer, 3:skirt, 4:dress, 5:pants, 6:leggings, 7:headwear,
#   8:eyeglass, 9:neckwear, 10:belt, 11:footwear, 12:bag, 13:hair, 14:face,
#   15:skin, 16:ring, 17:wrist, 18:socks, 19:gloves, 20:necklace, 21:rompers,
#   22:earrings, 23:tie
DFM_MAP = {
    "whole body clothes": {4, 21},            # dress, rompers
    "upper body clothes": {1, 2},             # top, outer
    "lower body clothes": {3, 5, 6},          # skirt, pants, leggings
    "shoes": {11},                            # footwear
}

# SegFormer(b2 clothes)：来自 config.json 的 id2label
#   4:Upper-clothes, 5:Skirt, 6:Pants, 7:Dress, 9:Left-shoe, 10:Right-shoe
SEGF_MAP = {
    "whole body clothes": {7},                # Dress
    "upper body clothes": {4},                # Upper-clothes
    "lower body clothes": {5, 6},             # Skirt, Pants
    "shoes": {9, 10},                         # Left-shoe, Right-shoe
}

PART_KEYS = ["upper body clothes", "lower body clothes", "whole body clothes", "shoes"]


# --------------------------- 模型加载（遵循示例） ------------------------------
def load_marqo_fashionsiglip():
    # 官方示例：AutoModel/AutoProcessor + trust_remote_code=True
    # 参考模型卡用法：processor(...); model.get_image_features(processed['pixel_values'], normalize=True)
    model = AutoModel.from_pretrained("Marqo/marqo-fashionSigLIP", trust_remote_code=True).to(DEVICE)
    processor = AutoProcessor.from_pretrained("Marqo/marqo-fashionSigLIP", trust_remote_code=True)
    model.eval()
    return model, processor


def load_segformer():
    # 官方示例：SegformerImageProcessor / AutoModelForSemanticSegmentation
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to(DEVICE)
    model.eval()
    return model, processor


# --------------------------- 工具函数 ------------------------------------------
def read_dfm_mask(segm_path: Path) -> np.ndarray:
    """读取 DeepFashion-MultiModal 的解析 PNG，转为 numpy int32"""
    segm = Image.open(segm_path)
    return np.array(segm, dtype=np.int32)


def segformer_infer_mask(img_pil: Image.Image,
                         model: AutoModelForSemanticSegmentation,
                         processor: SegformerImageProcessor) -> np.ndarray:
    """按官方示例推理 + 双线性上采样到原图尺寸，返回 HxW 的 int mask"""
    inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits  # [B, C, h, w]
        upsampled = nn.functional.interpolate(
            logits,
            size=img_pil.size[::-1],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        pred = upsampled.argmax(dim=1)[0].detach().to("cpu").numpy().astype(np.int32)
    return pred


def mask_union_bbox(mask: np.ndarray, classes: set[int]) -> Optional[Tuple[int, int, int, int]]:
    """给定若干类别 ID，返回 union 区域的外接矩形 (left, upper, right, lower)。若为空返回 None。"""
    if not classes:
        return None
    m = np.isin(mask, list(classes))
    if not m.any():
        return None
    ys, xs = np.where(m)
    left, right = xs.min(), xs.max() + 1
    top, bottom = ys.min(), ys.max() + 1
    return int(left), int(top), int(right), int(bottom)


def crop_part(img: Image.Image, bbox: Optional[Tuple[int, int, int, int]]) -> Optional[Image.Image]:
    """按外接矩形裁剪；若 bbox 为空返回 None。"""
    if bbox is None:
        return None
    # 避免零面积
    l, t, r, b = bbox
    if r - l <= 1 or b - t <= 1:
        return None
    return img.crop((l, t, r, b))


def extract_image_feature(
    img_pil: Image.Image,
    model: AutoModel,
    processor: AutoProcessor,
) -> torch.Tensor:
    """遵循官方示例：processor -> model.get_image_features(..., normalize=True)"""
    # 示例里使用了 text+images，这里仅需图像特征；为与示例接口稳定性一致，保留 text 为空串占位。
    processed = processor(text=[""], images=[img_pil], padding="max_length", return_tensors="pt")
    pixel_values = processed["pixel_values"].to(DEVICE)
    with torch.inference_mode():
        feats = model.get_image_features(pixel_values, normalize=True)
    return feats[0].detach().to("cpu")  # [D] 向量


def parts_from_mask(
    mask: np.ndarray, source: str
) -> Dict[str, Optional[Tuple[int, int, int, int]]]:
    """根据 mask + 源（'dfm' 或 'segf'）返回四类部位的 bbox；若 whole 存在则 upper/lower 置 None。"""
    if source == "dfm":
        mapping = DFM_MAP
    else:
        mapping = SEGF_MAP

    boxes = {k: mask_union_bbox(mask, classes) for k, classes in mapping.items()}

    # 规则：若 whole 存在，则 upper/lower 不再输出
    if boxes["whole body clothes"] is not None:
        boxes["upper body clothes"] = None
        boxes["lower body clothes"] = None
    return boxes


def process_one_image(
    img_path: Path,
    segformer_model: AutoModelForSemanticSegmentation,
    segformer_processor: SegformerImageProcessor,
    marqo_model: AutoModel,
    marqo_processor: AutoProcessor,
) -> Dict[str, Optional[torch.Tensor]]:
    """处理单张图片：得到四类裁剪并抽取特征；不存在返回 None。"""
    img = Image.open(img_path).convert("RGB")
    segm_path = SEGM_DIR / f"{img_path.stem}_segm.png"

    if segm_path.exists():
        mask = read_dfm_mask(segm_path)
        boxes = parts_from_mask(mask, source="dfm")
    else:
        mask = segformer_infer_mask(img, segformer_model, segformer_processor)
        boxes = parts_from_mask(mask, source="segf")

    feats_dict: Dict[str, Optional[torch.Tensor]] = {k: None for k in PART_KEYS}
    for k in PART_KEYS:
        crop_img = crop_part(img, boxes[k])
        if crop_img is None:
            feats_dict[k] = None
            continue
        feats = extract_image_feature(crop_img, marqo_model, marqo_processor)
        feats_dict[k] = feats
    return feats_dict


def main():
    # 加载模型（遵循官方示例接口）
    segformer_model, segformer_processor = load_segformer()
    marqo_model, marqo_processor = load_marqo_fashionsiglip()

    img_paths = sorted(IMAGES_DIR.glob("*.jpg"))

    for img_path in tqdm(img_paths, desc="Processing images", unit="img"):
        feats = process_one_image(
            img_path,
            segformer_model,
            segformer_processor,
            marqo_model,
            marqo_processor,
        )
        out_path = FEAT_DIR / f"{img_path.stem}_feat.pt"
        torch.save(feats, out_path)


if __name__ == "__main__":
    main()
