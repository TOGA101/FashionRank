#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# —— 来自 CVLFace 官方 Quick Start 的核心：下载/加载模型的辅助函数 ——
# 参考：RETINAFACE RESNET50 / ADAFACE IR101 WebFace12M 的 Quick Start
# （保持与官方一致的函数名与调用方式）
from transformers import AutoModel
from huggingface_hub import hf_hub_download
import shutil
import os
import sys

def download(repo_id, path, HF_TOKEN=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN, local_dir=path, local_dir_use_symlinks=False)

def load_model_from_local_path(path, HF_TOKEN=None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model

def load_model_by_repo_id(repo_id, save_path, HF_TOKEN=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)

# —— 本任务逻辑：批处理 DeepFashion-MultiModal 的 images/ 目录 ——
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import torch
from tqdm import tqdm
import argparse
import inspect

def pil_to_cvlface_input(pil_image, device):
    # 与官方示例完全一致的归一化方式（[0,1]→[-1,1]）
    trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return trans(pil_image).unsqueeze(0).to(device)

def main():
    parser = argparse.ArgumentParser(description="Extract face embeddings with CVLFace (DeepFashion-MultiModal).")
    parser.add_argument("--root", type=str, default=".", help="DeepFashion-MultiModal 主目录（包含 images/）")
    parser.add_argument("--images-dir", type=str, default="images", help="相对主目录的 images 子目录名")
    parser.add_argument("--emb-dir", type=str, default="emb", help="相对主目录的输出子目录名")
    parser.add_argument("--det-thres", type=float, default=0.9, help="人脸检测置信度阈值（>该值才提取）")
    parser.add_argument("--force-download", action="store_true", help="重新下载模型文件")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    images_dir = (root / args.images-dir) if hasattr(args, "images-dir") else (root / "images")  # 防止某些终端对连字符处理异常
    if not images_dir.exists():
        images_dir = root / "images"
    emb_dir = root / args.emb_dir
    emb_dir.mkdir(parents=True, exist_ok=True)

    # 环境：GPU 可用则用 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读取 HF_TOKEN（RETINAFACE RESNET50 为私有仓库）
    HF_TOKEN = os.environ.get("HF_TOKEN", None)
    # if HF_TOKEN is None:
    #     raise RuntimeError(
    #         "未检测到环境变量 HF_TOKEN。RETINAFACE RESNET50 为私有模型，需设置你的 Hugging Face 访问令牌：\n"
    #         "  Linux/macOS: export HF_TOKEN=xxxxxxxx\n"
    #         "  Windows(PowerShell): $env:HF_TOKEN='xxxxxxxx'"
    #     )

    # 加载 CVLFace 模型（官方示例写法）
    # 人脸检测/对齐：RETINAFACE RESNET50（CVLFace 提供）
    aligner_repo = "minchul/private_retinaface_resnet50"
    aligner_cache = os.path.expanduser("~/.cvlface_cache/minchul/private_retinaface_resnet50")
    aligner = load_model_by_repo_id(aligner_repo, aligner_cache, HF_TOKEN, force_download=args.force_download).to(device).eval()

    # 人脸嵌入模型：AdaFace IR101（CVLFace 提供）
    recog_repo = "minchul/cvlface_adaface_ir101_webface12m"
    recog_cache = os.path.expanduser("~/.cvlface_cache/minchul/cvlface_adaface_ir101_webface12m")
    fr_model = load_model_by_repo_id(recog_repo, recog_cache, HF_TOKEN, force_download=args.force_download).to(device).eval()

    # 判断是否需要传 keypoints（参考官方 Face Verification 示例）
    input_signature = inspect.signature(fr_model.model.net.forward)  # 官方示例取法
    accepts_kpts = input_signature.parameters.get("keypoints") is not None

    # 遍历 JPG（要求为 .jpg）
    image_paths = sorted([p for p in images_dir.glob("*.jpg")])
    if not image_paths:
        print(f"[警告] 未在 {images_dir} 下找到 .jpg 文件。")
        return

    # 进度条
    pbar = tqdm(image_paths, desc="Extracting embeddings", unit="img")

    # 推理（与官方一致的归一化 & 调用方式；仅添加了阈值判断/保存逻辑）
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        for img_path in pbar:
            try:
                # 输出文件名：原图名 + "_emb".pt
                out_path = emb_dir / f"{img_path.stem}_emb.pt"
                if out_path.exists():
                    continue  # 已存在则跳过（避免重复计算）

                # 读取与预处理
                img = Image.open(img_path).convert("RGB")
                inp = pil_to_cvlface_input(img, device)

                # 检测 + 对齐（官方 Quick Start 返回：aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox）
                aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = aligner(inp)

                # 置信度过滤
                conf = float(score.squeeze().item())
                if not (conf > args.det_thres):
                    continue  # 低置信度或无脸 → 跳过

                # 提取特征（是否需要 keypoints 由签名判断——官方示例）
                if accepts_kpts:
                    feat = fr_model(aligned_x, aligned_ldmks)
                else:
                    feat = fr_model(aligned_x)

                # 存为 1D 向量（.pt）
                feat_vec = feat.squeeze(0).detach().cpu()  # (C,) 向量
                torch.save(feat_vec, out_path)

            except Exception as e:
                # 本任务要求“无关逻辑尽量不加”，这里仅最小化容错：单张失败不影响整体
                # 如需深入定位，可手动打印 img_path 与异常信息
                continue

if __name__ == "__main__":
    main()
