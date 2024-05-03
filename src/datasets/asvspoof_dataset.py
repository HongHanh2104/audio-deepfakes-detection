from pathlib import Path

import pandas as pd
import os

from src.datasets.base_dataset import SimpleAudioFakeDataset

# ASVSPOOF_SPLIT = {
#     "train": ['A01', 'A07', 'A08', 'A02', 'A09', 'A10', 'A03', 'A04', 'A05', 'A06', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19'],
#     "test":  ['A01', 'A07', 'A08', 'A02', 'A09', 'A10', 'A03', 'A04', 'A05', 'A06', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19'],
#     "val":   ['A01', 'A07', 'A08', 'A02', 'A09', 'A10', 'A03', 'A04', 'A05', 'A06', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19'],
#     "partition_ratio": [0.7, 0.15],
#     "seed": 2104,
# }

ATTACKS = ['A01', 'A07', 'A08', 'A02', 'A09', 'A10', 'A03', 'A04', 'A05', 'A06', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19']

class ASVSpoofDataset(SimpleAudioFakeDataset):

    protocol_folder_name = "ASVspoof2019_LA_cm_protocols"
    subset_dir_prefix = "ASVspoof2019_LA_"
    subsets = ("train", "dev", "eval")

    def __init__(self, path, subset="train", transform=None):
        super().__init__(subset, transform)
        
        print("ASVSpoof2019 dataset .....")
        self.path = path
        self.subset = subset
        self.allowed_attacks = ATTACKS
        # self.partition_ratio = ASVSPOOF_SPLIT["partition_ratio"]
        # self.seed = ASVSPOOF_SPLIT["seed"]
        self.transform = transform

        self.samples = pd.concat([self.get_fake_samples(), self.get_real_samples()], ignore_index=True)
        
        # for subset in self.subsets:
        #     subset_dir = Path(self.path) / f"{self.subset_dir_prefix}{subset}"
        #     subset_protocol_path = self.get_protocol_path(subset)
        #     print(subset_protocol_path)
        #     subset_samples = self.read_protocol(subset_dir, subset_protocol_path)

        #     self.samples = pd.concat([self.samples, subset_samples])

        # self.samples, self.attack_signatures = self.group_by_attack()
    
    def get_real_samples(self):
        filename = f'{self.subset}_real_samples.csv'
        
        df = pd.read_csv(os.path.join(self.path, filename))
        
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        for i, row in df.iterrows():
            samples["user_id"].append(row['user_id'])
            samples["sample_name"].append(row['sample_name'])
            samples["attack_type"].append(row['attack_type'])
            samples["label"].append(row["label"])
            samples["path"].append(row['path'])

        return pd.DataFrame(samples)  
    
    def get_fake_samples(self):
        filename = f'{self.subset}_fake_samples.csv'
        
        df = pd.read_csv(os.path.join(self.path, filename))
        
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        for i, row in df.iterrows():
            samples["user_id"].append(row['user_id'])
            samples["sample_name"].append(row['sample_name'])
            samples["attack_type"].append(row['attack_type'])
            samples["label"].append(row["label"])
            samples["path"].append(row['path'])

        return pd.DataFrame(samples)  

    def get_protocol_path(self, subset):
        
        paths = list((Path(self.path) / self.protocol_folder_name).glob("*.txt"))
        for path in paths:
            if subset in str(path):
                return path

    def read_protocol(self, subset_dir, protocol_path):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        real_samples = []
        fake_samples = []
        with open(protocol_path, "r") as file:
            for line in file:
                attack_type = line.strip().split(" ")[3]

                if attack_type == "-":
                    real_samples.append(line)
                elif attack_type in self.allowed_attacks:
                    fake_samples.append(line)

                if attack_type not in self.allowed_attacks:
                    continue

        fake_samples = self.split_samples(fake_samples)
        for line in fake_samples:
            samples = self.add_line_to_samples(samples, line, subset_dir)

        real_samples = self.split_samples(real_samples)
        for line in real_samples:
            samples = self.add_line_to_samples(samples, line, subset_dir)

        return pd.DataFrame(samples)

    @staticmethod
    def add_line_to_samples(samples, line, subset_dir):
        user_id, sample_name, _, attack_type, label = line.strip().split(" ")
        samples["user_id"].append(user_id)
        samples["sample_name"].append(sample_name)
        samples["attack_type"].append(attack_type)
        samples["label"].append(label)

        assert (subset_dir / "flac" / f"{sample_name}.flac").exists()
        samples["path"].append(subset_dir / "flac" / f"{sample_name}.flac")

        return samples

if __name__ == "__main__":
    ASVSPOOF_DATASET_PATH = '/mnt/storage/hanhnlh/ccs/dataset/ASVspoof19_LA'

    real = 0
    fake = 0
    datasets = []

    for subset in ['train', 'val', 'test']:
        dataset = ASVSpoofDataset(ASVSPOOF_DATASET_PATH, subset=subset)
        
        print(f'Number of samples in {subset} set: {len(dataset.samples)}')

        real_samples = dataset.samples[dataset.samples['label'] == 'bonafide']
        real += len(real_samples)

        print('real', len(real_samples))

        spoofed_samples = dataset.samples[dataset.samples['label'] == 'spoof']
        fake += len(spoofed_samples)

        print('fake', len(spoofed_samples))
        datasets.append(dataset)

    print(real, fake)

    paths_0 = [str(p) for p in datasets[0].samples.path.values]  # pathlib -> str
    paths_1 = [str(p) for p in datasets[1].samples.path.values]
    paths_2 = [str(p) for p in datasets[2].samples.path.values]

    assert len(set(paths_0).intersection(set(paths_1))) == 0, "duplicated paths"
    assert len(set(paths_1).intersection(set(paths_2))) == 0, "duplicated paths"
    assert len(set(paths_0).intersection(set(paths_2))) == 0, "duplicated paths"

    print("All correct!")

    # TODO(PK): points to fulfill
    # [ ] each attack type should be present in each subset
    # [x] no duplicates
