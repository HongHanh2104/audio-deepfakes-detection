from pathlib import Path
import os
import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset

WAVEFAKE_SPLIT = {
    "train": ['multi_band_melgan', 'melgan_large', 'parallel_wavegan', 'waveglow', 'full_band_melgan', 'melgan', 'hifiGAN'],
    "test":  ['multi_band_melgan', 'melgan_large', 'parallel_wavegan', 'waveglow', 'full_band_melgan', 'melgan', 'hifiGAN'],
    "val":   ['multi_band_melgan', 'melgan_large', 'parallel_wavegan', 'waveglow', 'full_band_melgan', 'melgan', 'hifiGAN'],
    "partition_ratio": [0.7, 0.15],
    "seed": 2104
}


class WaveFakeDataset(SimpleAudioFakeDataset):
    

    def __init__(self, path, subset="train", transform=None):
        super().__init__(subset, transform)
        self.path = Path(path)

        self.fold_subset = subset
        # self.allowed_attacks = WAVEFAKE_SPLIT[subset]
        # self.partition_ratio = WAVEFAKE_SPLIT["partition_ratio"]
        self.seed = 2104

        self.samples = pd.concat([self.get_fake_samples(), self.get_real_samples()], ignore_index=True)
        self.check_for_error_file()
        
    def get_fake_samples(self):
        filename = f'{self.fold_subset}_fake.csv'
        
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


    def get_real_samples(self):
        filename = f'{self.fold_subset}_real.csv'
        
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



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    WAVEFAKE_DATASET_PATH = "/mnt/storage/hanhnlh/ccs/dataset/WaveFake"

    real = 0
    fake = 0
    datasets = []
    
    subset = 'train'
    dataset = WaveFakeDataset(WAVEFAKE_DATASET_PATH, subset=subset)
    print(f'Number of samples in {subset} set: {len(dataset.samples)}')
    
    train_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )
    
    # load over the train_loader
    for i, (batch_x, _, _, batch_y) in enumerate(train_loader):
        print("yes")
        break