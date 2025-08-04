#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Extract SHAPY shape features (SMPL betas) for full-body images in DeepFashion-MultiModal.
#
# Requirements
# - PyTorch, torchvision, Pillow, omegaconf, tqdm
# - SHAPY + dependencies available as used by the provided regressor `inference.py`.
# - GPU optional but recommended (uses CUDA if available).
#
# Definition of 'full body image'
# We only process images that:
#   (1) Have a corresponding human parsing label file under the dataset's `segm/` folder,
#       named as: <original_image_name_without_ext> + '_segm.png'
#   (2) Appear in BOTH `keypoints/keypoints_loc.txt` and `keypoints/keypoints_vis.txt`.
#
# Notes
# - This script imports the SHAPY regressor utilities from the working directory path
#   './shapy/regressor/inference.py' (relative to where this script lives). Ensure that file exists.
# - The SHAPY checkpoints/config referenced by that `inference.py` must be present; otherwise,
#   model loading will fail.
#
# TODOs
# - If your local DeepFashion-MultiModal uses the folder name `parsing/` instead of `segm/`,
#   change SEGMENTATION_DIR_NAME below accordingly.
# - If your SHAPY `inference.py` expects different config/checkpoint locations, edit
#   DEFAULT_EXP_CFG / DEFAULT_MODEL_FOLDER imports accordingly.
#
# Usage:
#   python extract_shapy_betas.py /path/to/DeepFashion-MultiModal
# Output:
#   <dataset_root>/shape/<image_name>_shape.pt
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Set

import torch
from tqdm import tqdm

# -------------------------------------------------------------------------
# Import SHAPY regressor utilities from the provided file:
#     ./shapy/regressor/inference.py
# We rely ONLY on its public functions / constants used below.
# -------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
REGRESSOR_DIR = THIS_DIR / "shapy" / "regressor"
if REGRESSOR_DIR.exists():
    sys.path.append(str(REGRESSOR_DIR))

try:
    # Expected to exist as per the provided reference file.
    from inference import (  # type: ignore
        load_model,
        preprocess_image,
        infer_single_image,
        DEFAULT_EXP_CFG,
        DEFAULT_MODEL_FOLDER,
        default_conf,
        OmegaConf,  # re-exported in the reference file
    )
except Exception as e:
    raise ImportError(
        "Could not import from ./shapy/regressor/inference.py. "
        "Please ensure the file exists at that path relative to this script."
    ) from e

# -------------------------------------------------------------------------
# Config: dataset folder names
# -------------------------------------------------------------------------
IMAGES_DIR_NAME = "images"
KEYPOINTS_DIR_NAME = "keypoints"
SEGMENTATION_DIR_NAME = "segm"  # Change to 'parsing' if your dataset uses that name.

# Filenames inside keypoints/
KP_LOC_FILE = "keypoints_loc.txt"
KP_VIS_FILE = "keypoints_vis.txt"


def read_keypoint_filenames(kp_file: Path) -> Set[str]:
    """Read first token (image filename) per line from a keypoints .txt file."""
    names: Set[str] = set()
    with kp_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts:
                names.add(parts[0])
    return names


def is_full_body_image(img_path: Path, dataset_root: Path, kp_loc_names: Set[str], kp_vis_names: Set[str]) -> bool:
    """Full body = has corresponding segm PNG and appears in BOTH keypoints files."""
    img_name = img_path.name
    if img_name not in kp_loc_names or img_name not in kp_vis_names:
        return False
    segm_dir = dataset_root / SEGMENTATION_DIR_NAME
    segm_file = segm_dir / f"{img_path.stem}_segm.png"
    return segm_file.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SHAPY SMPL betas for full-body images.")
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the DeepFashion-MultiModal dataset root (contains images/, keypoints/, segm/).",
    )
    args = parser.parse_args()
    dataset_root: Path = args.dataset_root.resolve()

    images_dir = dataset_root / IMAGES_DIR_NAME
    keypoints_dir = dataset_root / KEYPOINTS_DIR_NAME
    segm_dir = dataset_root / SEGMENTATION_DIR_NAME
    shape_dir = dataset_root / "shape"
    shape_dir.mkdir(parents=True, exist_ok=True)

    # Validate required files/folders
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not keypoints_dir.is_dir():
        raise FileNotFoundError(f"Keypoints folder not found: {keypoints_dir}")
    if not segm_dir.is_dir():
        raise FileNotFoundError(f"Segmentation folder not found: {segm_dir}")

    kp_loc_file = keypoints_dir / KP_LOC_FILE
    kp_vis_file = keypoints_dir / KP_VIS_FILE
    if not kp_loc_file.is_file():
        raise FileNotFoundError(f"Missing file: {kp_loc_file}")
    if not kp_vis_file.is_file():
        raise FileNotFoundError(f"Missing file: {kp_vis_file}")

    # Build the sets of image names that have keypoints annotations
    kp_loc_names = read_keypoint_filenames(kp_loc_file)
    kp_vis_names = read_keypoint_filenames(kp_vis_file)

    # Prepare SHAPY model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and merge config following the reference inference.py
    cfg = default_conf.copy()
    cfg.merge_with(OmegaConf.load(str(DEFAULT_EXP_CFG)))
    cfg.output_folder = str(DEFAULT_MODEL_FOLDER.resolve())
    cfg.is_training = False

    model = load_model(cfg, device)

    crop_size = cfg.datasets.pose.transforms.crop_size
    mean = cfg.datasets.pose.transforms.mean
    std = cfg.datasets.pose.transforms.std

    # Iterate images with a progress bar
    img_paths = sorted(images_dir.glob("*.jpg"))
    pbar = tqdm(img_paths, desc="Extracting SHAPY betas", unit="img")

    for img_path in pbar:
        if not is_full_body_image(img_path, dataset_root, kp_loc_names, kp_vis_names):
            continue

        try:
            image_tensor = preprocess_image(str(img_path), crop_size, mean, std)
            betas_np = infer_single_image(model, image_tensor, device)  # numpy array (10,)
            betas = torch.from_numpy(betas_np).float()  # save as float32 tensor
            out_path = shape_dir / f"{img_path.name}_shape.pt"
            torch.save(betas, out_path)
        except Exception:
            # Keep going; show a brief error in the progress bar postfix.
            pbar.set_postfix_str(f"error: {img_path.name}")
            continue


if __name__ == "__main__":
    main()
