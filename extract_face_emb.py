#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量提取 DeepFashion-MultiModal /images 下可检测到 **完整人脸** 的人脸 embedding。
输出: emb/<原文件名>_emb.pt
依赖: torch, torchvision, transformers, huggingface_hub, tqdm, Pillow
GPU: 自动使用可用 GPU
"""

from __future__ import annotations
import os
import sys
import shutil
from pathlib import Path
from typing import Optional

import torch
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel
from huggingface_hub import hf_hub_download

# =========================
# 官方 Quick Start 中的辅助函数 (原样保留 + 轻微封装)
# =========================
def download(repo_id: str, path: str, HF_TOKEN: Optional[str] = None) -> None:
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN,
                        local_dir=path, local_dir_use_symlinks=False)
    with open(files_path, 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN,
                            local_dir=path, local_dir_use_symlinks=False)


def load_model_from_local_path(path: str, HF_TOKEN: Optional[str] = None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model


def load_model_by_repo_id(repo_id: str, save_path: str,
                          HF_TOKEN: Optional[str] = None,
                          force_download: bool = False):
    if force_download and os.path.exists(save_path):
        shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)


# =========================
# 主要处理逻辑
# =========================
def is_face_qualified(score: torch.Tensor,
                      bbox: torch.Tensor,
                      img_w: int,
                      img_h: int,
                      score_thr: float = 0.5,
                      min_area_ratio: float = 0.002) -> bool:
    """
    简单过滤规则：
    - score 大于阈值
    - bbox 面积占比不太小
    TODO:
      1. 确认官方推荐 score 阈值（当前假设 0.5）。
      2. 如果官方提供人脸质量或关键点可见性指标，可替换/补充。
    """
    try:
        s = float(score.squeeze().item())
    except Exception:
        return False
    if torch.isnan(score).any():
        return False
    if s < score_thr:
        return False
    # bbox 形状 (1,4): [x1, y1, x2, y2] 假设输出为此格式（若与官方实现不符需调整）
    # TODO: 验证 bbox 顺序；若为 (cx, cy, w, h) 等格式需相应改写。
    try:
        x1, y1, x2, y2 = bbox.squeeze().tolist()
    except Exception:
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    area = (x2 - x1) * (y2 - y1)
    if area / (img_w * img_h) < min_area_ratio:
        return False
    # 额外：要求 bbox 完全在图像内
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        return False
    return True


def extract_embeddings(
    dataset_root: Path,
    images_subdir: str = "images",
    output_subdir: str = "emb",
    hf_token_env: str = "HF_TOKEN",
    force_download: bool = False,
    device: Optional[str] = None,
):
    """
    遍历 images_subdir 下所有 .jpg，检测 + 对齐 + 特征提取。
    不符合完整人脸条件的直接跳过。
    """
    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_str)

    HF_TOKEN = os.getenv(hf_token_env, None)

    # --- 加载对齐(检测)模型 ---
    aligner_repo = "minchul/cvlface_DFA_mobilenet"
    aligner_cache = os.path.expanduser("~/.cvlface_cache/minchul/cvlface_DFA_mobilenet")
    aligner = load_model_by_repo_id(aligner_repo, aligner_cache, HF_TOKEN, force_download).to(dev).eval()

    # --- 加载人脸嵌入模型 ---
    embed_repo = "minchul/cvlface_adaface_ir101_webface12m"
    embed_cache = os.path.expanduser("~/.cvlface_cache/minchul/cvlface_adaface_ir101_webface12m")
    embed_model = load_model_by_repo_id(embed_repo, embed_cache, HF_TOKEN, force_download).to(dev).eval()

    # 变换：与官方示例一致
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5],
                  std=[0.5, 0.5, 0.5]),
    ])

    images_dir = dataset_root / images_subdir
    out_dir = dataset_root / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(images_dir.rglob("*.jpg"))
    if not img_paths:
        print(f"[WARN] 未找到任何 JPG：{images_dir}")
        return

    processed = 0
    skipped = 0

    pbar = tqdm(img_paths, desc="Extracting face embeddings", unit="img")

    with torch.no_grad():
        for img_path in pbar:
            try:
                save_path = out_dir / f"{img_path.stem}_emb.pt"
                if save_path.exists():
                    # 已存在则跳过（避免重复）
                    skipped += 1
                    continue

                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                inp = transform(img).unsqueeze(0).to(dev)

                # 检测 + 对齐
                aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = aligner(inp)

                if not is_face_qualified(score, bbox, w, h):
                    skipped += 1
                    continue

                # aligned_x 已为 (1,3,112,112) 且归一化（依据官方示例说明）
                # 送入嵌入模型
                # TODO: 如返回类型不是张量（例如 dict），需根据实际 wrapper 调整取值方式
                emb = embed_model(aligned_x.to(dev))

                # 若 emb 是 (1,D)，去掉 batch 维
                if isinstance(emb, torch.Tensor):
                    feat = emb.squeeze(0).detach().cpu()
                else:
                    # TODO: 处理非张量返回（例如 emb["features"]）
                    # 暂不保存，避免误写错误结构
                    skipped += 1
                    continue

                torch.save(
                    {
                        "embedding": feat,
                        "model_repo": embed_repo,
                        "aligner_repo": aligner_repo,
                        "source_image": str(img_path.relative_to(dataset_root)),
                    },
                    save_path
                )
                processed += 1

            except Exception as e:
                skipped += 1
                # 保持简洁：可以添加日志文件；此处仅在需要时打印调试
                # print(f"[ERROR] {img_path}: {e}")
                continue

            pbar.set_postfix({"ok": processed, "skip": skipped})

    print(f"完成: 生成 {processed} 个 embedding, 跳过 {skipped} 张。输出目录: {out_dir}")


def main():
    # 可根据需要改为 argparse；保持简洁
    # TODO: 若需命令行参数，可添加 argparse
    data_root = Path("PATH_TO_DeepFashion_MultiModal_ROOT")  # TODO: 设置为你的数据集根目录
    extract_embeddings(data_root)


if __name__ == "__main__":
    main()
