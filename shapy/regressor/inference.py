import argparse
from pathlib import Path

import torch
from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

from human_shape.config.defaults import conf as default_conf
from human_shape.utils import Checkpointer

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_EXP_CFG = (BASE_DIR / 'configs' / 'b2a_expose_hrnet_demo.yaml').resolve()
DEFAULT_MODEL_FOLDER = (
    REPO_ROOT / 'data' / 'trained_models' / 'shapy' / 'SHAPY_A'
).resolve()


def load_model(exp_cfg, device):
    from human_shape.models.build import build_model

    model_dict = build_model(exp_cfg)
    model = model_dict['network'].to(device)
    checkpoint_folder = Path(exp_cfg.output_folder) / exp_cfg.checkpoint_folder
    checkpointer = Checkpointer(model, save_dir=str(checkpoint_folder),
                                pretrained=exp_cfg.pretrained)
    checkpointer.load_checkpoint()
    model.eval()
    return model


def preprocess_image(path, crop_size, mean, std):
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


@torch.no_grad()
def infer_single_image(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)
    stage_key = output['stage_keys'][-1]
    betas = output[stage_key]['betas'][0].detach().cpu().numpy()
    return betas


def main():
    parser = argparse.ArgumentParser(description='Estimate SMPL betas from an image')
    parser.add_argument('image', type=str, help='Path to aligned full-body image')
    parser.add_argument('--exp-cfg', type=Path, default=DEFAULT_EXP_CFG,
                        help='Configuration file path')
    parser.add_argument('--model-folder', type=Path, default=DEFAULT_MODEL_FOLDER,
                        help='Folder with trained model checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = default_conf.copy()
    cfg.merge_with(OmegaConf.load(str(args.exp_cfg)))
    cfg.output_folder = str(args.model_folder.resolve())
    cfg.is_training = False

    model = load_model(cfg, device)

    crop_size = cfg.datasets.pose.transforms.crop_size
    mean = cfg.datasets.pose.transforms.mean
    std = cfg.datasets.pose.transforms.std
    image_tensor = preprocess_image(str(args.image), crop_size, mean, std)

    betas = infer_single_image(model, image_tensor, device)
    print(betas)


if __name__ == '__main__':
    main()
