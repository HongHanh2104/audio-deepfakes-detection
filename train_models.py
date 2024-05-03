import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import os
from datetime import datetime

import torch
import torchaudio
import yaml
from torch.utils.data import DataLoader

from src.datasets.detection_dataset import DetectionDataset
from src.models import models
from src.trainer import GDTrainer
from src.utils import set_seed

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET_NAME = "wavefake"
MODEL_NAME = "lcnn"


# Get current time
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y-%H-%M-%S")

PREFIX = f'{MODEL_NAME}_{DATASET_NAME}_{dt_string}'

SAVE_PATH = f'./trained_models/{PREFIX}'

BATCH_SIZE = 64
EPOCHS = 5
SEED = 2104

def save_model(
    model: torch.nn.Module,
    model_dir: Union[Path, str],
    name: str,
) -> None:
    full_model_dir = Path(f"{model_dir}/{name}")
    full_model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{full_model_dir}/ckpt.pth")


def get_datasets(
    datasets_paths: List[Union[Path, str]]
) -> Tuple[DetectionDataset, DetectionDataset]:
    data_train = DetectionDataset(
        asvspoof_path=datasets_paths[0],
        wavefake_path=datasets_paths[1],
        subset="train",
        undersample=False,
        return_label=True,
        return_raw=True
        
    )
    data_test = DetectionDataset(
        asvspoof_path=datasets_paths[0],
        wavefake_path=datasets_paths[1],
        subset="val",
        undersample=False,
        return_label=True,
        return_raw=True
    )

    return data_train, data_test


def train_nn(
    datasets_paths: List[Union[Path, str]],
    batch_size: int,
    epochs: int,
    device: str,
    config: Dict,
    model_dir: Optional[Path] = None,
    config_save_path: str = "configs",
) -> None:

    
    model_config = config["model"]
    model_name, model_parameters = MODEL_NAME, model_config["parameters"]
    optimizer_config = model_config["optimizer"]
    

    checkpoint_path = ""

    data_train, data_test = get_datasets(
        datasets_paths=datasets_paths,
    )
    
    print(f'Number of samples in train set: {len(data_train.samples)}')
    print(f'Number of samples in test set: {len(data_test.samples)}')

    current_model = models.get_model(
        model_name=model_name,
        config=model_parameters,
        device=device,
    ).to(device)

    use_scheduler = "rawnet2" in model_name.lower()
    
    print(f"Starting training {model_name} on {DATASET_NAME} dataset")
    
    current_model = GDTrainer(
        device=device,
        batch_size=batch_size,
        epochs=epochs,
        optimizer_kwargs=optimizer_config,
        use_scheduler=use_scheduler,
    ).train(
        dataset=data_train,
        model=current_model,
        test_dataset=data_test,
    )
    

    torch.save(current_model.state_dict(), f"{model_dir}/ckpt.pth")
    checkpoint_path= f"{model_dir}/ckpt.pth"

    print("Training finished!")

    # Save config for testing
    if model_dir is not None:
        config["checkpoint"] = {"path": checkpoint_path}
        config_save_path = SAVE_PATH + "/config.yaml"
        with open(config_save_path, "w") as f:
            yaml.dump(config, f)
        print(f"Test config saved at location '{config_save_path}'!")


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    
    # fix all seeds
    set_seed(SEED)


    model_dir = SAVE_PATH
    os.makedirs(model_dir, exist_ok=True)

    train_nn(
        datasets_paths=[args.asv_path, args.wavefake_path],
        device=DEVICE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        model_dir=model_dir,
        config=config,
    )
    

def parse_args():
    parser = argparse.ArgumentParser()

    ASVSPOOF_DATASET_PATH = None 
    WAVEFAKE_DATASET_PATH = "/mnt/storage/hanhnlh/ccs/dataset/WaveFake"
    CONFIG_PATH = "/home/hanhnlh/projects/ccs/SPEECH_PROJECT/deepfakes_detectors/audio-deepfake-adversarial-attacks/configs/training/lcnn.yaml"

    parser.add_argument(
        "--asv_path",
        type=str,
        default=ASVSPOOF_DATASET_PATH,
        help="Path to ASVspoof2021 dataset directory",
    )
    parser.add_argument(
        "--wavefake_path",
        type=str,
        default=WAVEFAKE_DATASET_PATH,
        help="Path to WaveFake dataset directory",
    )
    parser.add_argument(
        "--config",
        help="Model config file path (default: config.yaml)",
        type=str,
        default=CONFIG_PATH,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
