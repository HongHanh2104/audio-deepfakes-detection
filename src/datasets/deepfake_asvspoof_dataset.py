import logging
from pathlib import Path

import torchaudio
from torch.utils.data import Dataset
import pandas as pd
import glob
import os

from src.datasets.base_dataset import SimpleAudioFakeDataset, AudioDataset, PadDataset

DF_ASVSPOOF_SPLIT = {
    "partition_ratio": [0.7, 0.15],
    "seed": 45
}

LOGGER = logging.getLogger()

class SpoofDataset(Dataset):
    def __init__(self, data_path, k):
        self.data_path = data_path
        self.audio_paths = glob.glob(f'{self.data_path}/*/*/*/*.wav')
        self.k = k 
        # k_string = f'K_{self.k}'
        # self.k_audios = []
        
        # for audio_path in self.audio_paths:
        #     filename = audio_path.split('/')[-1]
        #     if k_string in filename:
        #         self.k_audios.append(audio_path)
        #     else:
        #         continue
        

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        sample_path = self.audio_paths[index]
        
        waveform, _ = sf.read(sample_path)
        waveform = torch.tensor(waveform, dtype=torch.float32)
        # waveform, sample_rate = torchaudio.load(sample_path, normalize=True)
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform[:1, ...]
        waveform, sample_rate = AudioDataset.apply_trim(waveform, sample_rate)
        waveform = PadDataset.apply_pad(waveform, 64_600)
        # sample = torch.unsqueeze(sample, 0)
        label = 0 # spoof
        return waveform, label, sample_path



class DeepFakeASVSpoofDataset(SimpleAudioFakeDataset):

    protocol_file_name = "keys/DF/CM/trial_metadata.txt"
    subset_dir_prefix = "ASVspoof2021_DF_eval"
    subset_parts = ("part00", "part01", "part02", "part03")

    def __init__(self, path, subset="train", transform=None):
        super().__init__(subset, transform)
        print("ASVSpoof2021 dataset .....")
        
        self.path = path
        self.subset = subset
        # self.partition_ratio = DF_ASVSPOOF_SPLIT["partition_ratio"]
        # self.seed = DF_ASVSPOOF_SPLIT["seed"]

        self.transform = transform
        
        self.samples = pd.concat([self.get_fake_samples(), self.get_real_samples()], ignore_index=True)

        

    def get_real_samples(self):
        filename = f'{self.subset}_real.csv'
        
        df = pd.read_csv(os.path.join(self.path, filename))
        
        samples = {
            "sample_name": [],
            "label": [],
            "path": []
        }

        for i, row in df.iterrows():
            samples["sample_name"].append(row['sample_name'])
            samples["label"].append(row["label"])
            samples["path"].append(row['path'])

        return pd.DataFrame(samples)  
    
    def get_fake_samples(self):
        filename = f'{self.subset}_fake.csv'
        
        df = pd.read_csv(os.path.join(self.path, filename))
        
        samples = {
            "sample_name": [],
            "label": [],
            "path": []
        }

        for i, row in df.iterrows():
            samples["sample_name"].append(row['sample_name'])
            samples["label"].append(row["label"])
            samples["path"].append(row['path'])

        return pd.DataFrame(samples)  
    
    
    def get_file_references(self):
        flac_paths = {}
        for part in self.subset_parts:
            path = Path(self.path) / f"{self.subset_dir_prefix}_{part}" / "flac"
            flac_list = list(path.glob("*.flac"))

            for path in flac_list:
                flac_paths[path.stem] = path

        return flac_paths

    def read_protocol(self):
        samples = {
            "sample_name": [],
            "label": [],
            "path": [],
            "attack_type": []
        }

        real_samples = []
        fake_samples = []
        with open(Path(self.path) / self.protocol_file_name, "r") as file:
            for line in file:
                label = line.strip().split(" ")[5]

                if label == "bonafide":
                    real_samples.append(line)
                elif label == "spoof":
                    fake_samples.append(line)

        fake_samples = self.split_samples(fake_samples)
        for line in fake_samples:
            samples = self.add_line_to_samples(samples, line)
            

        real_samples = self.split_samples(real_samples)
        for line in real_samples:
            samples = self.add_line_to_samples(samples, line)

        return pd.DataFrame(samples)

    def add_line_to_samples(self, samples, line):
        sample_name = line.strip().split(" ")[1]
        label = line.strip().split(" ")[5]
        # _, sample_name, _, _, _, label, _, _ = line.strip().split(" ")
        samples["sample_name"].append(sample_name)
        samples["label"].append(label)

        sample_path = self.flac_paths[sample_name]
        assert sample_path.exists()
        samples["path"].append(sample_path)
        samples["attack_type"].append("-")
        return samples
    
    def __get_item__(self, index):
        if isinstance(self.samples, pd.DataFrame):
            sample = self.samples.iloc[index]

            path = str(sample["path"])
            label = sample["label"]
            attack_type = sample["attack_type"]
            if type(attack_type) != str and math.isnan(attack_type):
                attack_type = "N/A"
        else:
            path, label, attack_type = self.samples[index]

        data, sr = sf.read(path)

        if self.transform:
            data = self.transform(data)
        if label == 'spoof':
            label = torch.tensor(0)
        else:
            label = torch.tensor(1)
        return data, attack_type, label 

    
    


if __name__ == "__main__":
    ASVSPOOF_DATASET_PATH = '/mnt/storage/hanhnlh/ccs/dataset/ASVspoof21_DF'

    real = 0
    fake = 0
    datasets = []

    for subset in ['train', 'test', 'val']:
        dataset = DeepFakeASVSpoofDataset(ASVSPOOF_DATASET_PATH, subset=subset)
        print(f'Number of samples in {subset} set: {len(dataset.samples)}')
        real_samples = dataset.samples[dataset.samples['label'] == 'bonafide']
        real += len(real_samples)

        print('real', len(real_samples))

        spoofed_samples = dataset.samples[dataset.samples['label'] == 'spoof']
        fake += len(spoofed_samples)

        print('fake', len(spoofed_samples))

        datasets.append(dataset)

    paths_0 = [str(p) for p in datasets[0].samples.path.values]  # pathlib -> str
    paths_1 = [str(p) for p in datasets[1].samples.path.values]
    paths_2 = [str(p) for p in datasets[2].samples.path.values]

    assert len(paths_0) == len(set(paths_0)), "duplicated paths in subset"
    assert len(paths_1) == len(set(paths_1)), "duplicated paths in subset"
    assert len(paths_2) == len(set(paths_2)), "duplicated paths in subset"

    assert len(set(paths_0).intersection(set(paths_1))) == 0, "duplicated paths"
    assert len(set(paths_1).intersection(set(paths_2))) == 0, "duplicated paths"
    assert len(set(paths_0).intersection(set(paths_2))) == 0, "duplicated paths"

    print("All correct!")

