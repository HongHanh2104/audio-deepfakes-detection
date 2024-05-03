import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import os

import torch
import yaml
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
)
from torch.utils.data.dataloader import DataLoader

# from torchmetrics.classification import BinaryAccuracy
from src.datasets.deepfake_asvspoof_dataset import SpoofDataset

from src import metrics, utils
from src.datasets.detection_dataset import DetectionDataset
from src.models import models


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DATASET_NAME = "wavefake"
MODEL_NAME = "specrnet"

BATCH_SIZE = 1
EPOCHS = 5
SEED = 2104

def get_dataset(
    datasets_paths: List[Union[Path, str]],
) -> DetectionDataset:
    data_val = DetectionDataset(
        asvspoof_path=datasets_paths[0],
        wavefake_path=datasets_paths[1],
        subset="test",
        return_label=True,
        return_raw=True,
        undersample=False
    )

    return data_val


def evaluate_nn(
    model_path: Optional[Path],
    datasets_paths: List[Union[Path, str]],
    model_config: Dict,
    device: str,
    batch_size: int = 128,
):
    model_name, model_parameters = model_config["name"], model_config["parameters"]


    # Load model architecture
    model = models.get_model(
        model_name=model_name,
        config=model_parameters,
        device=device,
    )
    
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    data_val = get_dataset(
        datasets_paths=datasets_paths,
    )

    print("Number of samples in val set: ", len(data_val))

    print(f"Testing '{model_name}' model, model path: '{model_path}', on {len(data_val)} audio files.")
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)
        
    test_loader = DataLoader(
        data_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=3,
        worker_init_fn=seed_worker,
        generator=g
    )

    batches_number = len(data_val) // batch_size
    num_correct = 0.0
    num_total = 0.0
    y_pred = torch.Tensor([]).to(device)
    y = torch.Tensor([]).to(device)
    y_pred_label = torch.Tensor([]).to(device)

    for i, (batch_x, _, _, batch_y) in enumerate(test_loader):
        # import pdb; pdb.set_trace()
        model.eval()
        if i % 10 == 0:
            print(f"Batch [{i}/{batches_number}]")
        # print(batch_y)
        with torch.no_grad():
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            num_total += batch_x.size(0)
            batch_pred = model(batch_x).squeeze(1)
            batch_pred = torch.sigmoid(batch_pred)
            batch_pred_label = (batch_pred + 0.5).int()

            num_correct += (batch_pred_label == batch_y.int()).sum(dim=0).item()

            y_pred = torch.concat([y_pred, batch_pred], dim=0)
            y_pred_label = torch.concat([y_pred_label, batch_pred_label], dim=0)
            y = torch.concat([y, batch_y], dim=0)

    eval_accuracy = num_correct / num_total

    precision, recall, f1_score, support = precision_recall_fscore_support(
        y.cpu().numpy(), y_pred_label.cpu().numpy(), average="binary", beta=1.0
    )
    auc_score = roc_auc_score(y_true=y.cpu().numpy(), y_score=y_pred.cpu().numpy())

    # For EER flip values, following original evaluation implementation
    y_for_eer = 1 - y

    thresh, eer, fpr, tpr = metrics.calculate_eer(
        y=y_for_eer.cpu().numpy(),
        y_score=y_pred.cpu().numpy(),
    )

    eer_label = f"eval/eer"
    accuracy_label = f"eval/accuracy"
    precision_label = f"eval/precision"
    recall_label = f"eval/recall"
    f1_label = f"eval/f1_score"
    auc_label = f"eval/auc"

    print(
        f"{eer_label}: {eer:.4f}, {accuracy_label}: {eval_accuracy:.4f}, {precision_label}: {precision:.4f}, {recall_label}: {recall:.4f}, {f1_label}: {f1_score:.4f}, {auc_label}: {auc_score:.4f}"
    )

  

def main(args):

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    utils.set_seed(SEED)

    evaluate_nn(
        model_path=config["checkpoint"].get("path", ""),
        datasets_paths=[args.asv_path, args.wavefake_path],
        model_config=config["model"],
        device=DEVICE,
        batch_size=BATCH_SIZE,
    )



def parse_args():
    parser = argparse.ArgumentParser()
    ASVSPOOF_DATASET_PATH = None #'/home/hanhnlh/projects/ccs/dataset/ASVspoof21_DF' 
    WAVEFAKE_DATASET_PATH = "/mnt/storage/hanhnlh/ccs/dataset/WaveFake"
    CONFIG_PATH = "/home/hanhnlh/projects/ccs/SPEECH_PROJECT/deepfakes_detectors/audio-deepfake-adversarial-attacks/trained_models/specrnet_wavefake_22-04-2024-19-13-12/config.yaml"

    parser.add_argument("--asv_path", type=str, default=ASVSPOOF_DATASET_PATH)
    parser.add_argument("--wavefake_path", type=str, default=WAVEFAKE_DATASET_PATH)

    parser.add_argument(
        "--config",
        help="Model config file path (default: config.yaml)",
        type=str,
        default=CONFIG_PATH,
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
