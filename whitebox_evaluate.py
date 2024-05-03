import glob
import os
import pandas as pd

import IPython.display as ipd
import numpy as np
import yaml
from typing import Dict, List, Optional, Union
import soundfile as sf

import torch
import torch.nn as nn
import torchaudio
from torchmetrics import Accuracy
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data.dataloader import DataLoader
from art.utils import get_file

from src import metrics, utils
from src.datasets.detection_dataset import DetectionDataset
from src.models import models

# OUTPUT_SIZE = 8000
# ORIGINAL_SAMPLING_RATE = 16000
# DOWNSAMPLED_SAMPLING_RATE = 16000



ASVSPOOF_DATASET_PATH = "/mnt/storage/hanhnlh/ccs/dataset/WaveFake"
CONFIG_PATH = "/home/hanhnlh/projects/working/audio_spoof/deepfakes_detectors/audio-deepfake-adversarial-attacks/trained_models/lcnn_task_combined_21-12-2023-23-34-40/config.yaml"
MODEL_PATH = "/home/hanhnlh/projects/working/audio_spoof/deepfakes_detectors/audio-deepfake-adversarial-attacks/trained_models/lcnn_task_combined_21-12-2023-23-34-40/ckpt.pth"
ADV_SAMPLES_PATH = '/home/hanhnlh/projects/ccs/deepfakes_detectors/audio-deepfake-adversarial-attacks/outputs_adv/LCNN_deepfake_task_default_params'

WAVE_FAKE_SR = 16000
SEED = 2104

utils.set_seed(SEED)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


# Load model
model_config = config["model"]
model_name, model_parameters = model_config["name"], model_config["parameters"]
print("Loading model: {}".format(model_name))
model = models.get_model(
        model_name=model_name,
        config=model_parameters,
        device=device,
    )

# If provided weights, apply corresponding ones (from an appropriate fold)
if MODEL_PATH:
    model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)

SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]
WAVE_FAKE_CUT = 64600
WAVE_FAKE_SR = 16000


def apply_trim(waveform, sample_rate):
    (
        waveform_trimmed,
        sample_rate_trimmed,
    ) = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, SOX_SILENCE
    )

    if waveform_trimmed.size()[1] > 0:
        waveform = waveform_trimmed
        sample_rate = sample_rate_trimmed

    return waveform, sample_rate

def resample(path, target_sample_rate, normalize=True):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
        path, [["rate", f"{target_sample_rate}"]], normalize=normalize
    )

    return waveform, sample_rate

def resample_wave(waveform, sample_rate, target_sample_rate):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, [["rate", f"{target_sample_rate}"]]
    )
    return waveform, sample_rate


def apply_pad(waveform, cut):
    waveform = waveform.squeeze(0)
    waveform_len = waveform.shape[0]

    if waveform_len >= cut:
        return waveform[:cut]

    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

    return padded_waveform

if __name__ == "__main__":
    
    
    df = pd.read_csv(f"{ADV_SAMPLES_PATH}/adv_samples.csv")

    y = df[df['y'] == 0]
    y_adv = df[df['y_adv'] == 1]
    print(len(y_adv)/len(df)*100)
    
