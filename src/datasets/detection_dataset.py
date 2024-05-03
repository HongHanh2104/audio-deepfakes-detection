import logging
from typing import List, Optional

import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset
from src.datasets.asvspoof_dataset import ASVSpoofDataset
from src.datasets.deepfake_asvspoof_dataset import DeepFakeASVSpoofDataset
from src.datasets.wavefake_dataset import WaveFakeDataset

LOGGER = logging.getLogger()


class DetectionDataset(SimpleAudioFakeDataset):

    def __init__(
        self,
        asvspoof_path=None,
        wavefake_path=None,
        subset: str = "val",
        transform=None,
        oversample: bool = False,
        undersample: bool = False,
        return_label: bool = False,
        return_meta: bool = False,
        return_raw: bool = False
    ):
        super().__init__(
            subset=subset,
            transform=transform,
            return_label=return_label,
            return_meta=return_meta,
            return_raw=return_raw,
        )
        datasets = self._init_datasets(
            asvspoof_path=asvspoof_path,
            wavefake_path=wavefake_path,
            subset=subset,
        )
        self.samples = pd.concat(
            [ds.samples for ds in datasets],
            ignore_index=True
        )

        if oversample:
            self.oversample_dataset()
        elif undersample:
            self.undersample_dataset()


    def _init_datasets(
        self,
        asvspoof_path: Optional[str],
        wavefake_path: Optional[str],
        subset: str,
    ) -> List[SimpleAudioFakeDataset]:
        datasets = []

        if asvspoof_path is not None:
            # asvspoof_dataset = ASVSpoofDataset(asvspoof_path, subset=subset)
            asvspoof_dataset = DeepFakeASVSpoofDataset(asvspoof_path, subset=subset)
            datasets.append(asvspoof_dataset)

        if wavefake_path is not None:
            print("WaveFake dataset")
            wavefake_dataset = WaveFakeDataset(wavefake_path, subset=subset)
            datasets.append(wavefake_dataset)

        

        return datasets


    def oversample_dataset(self):
        
        samples = self.samples.groupby(by=['label'])
        for key, item in samples:
            print(samples.get_group(key), "\n\n")
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        diff_length = spoof_length - bona_length
        
        if diff_length < 0:
            raise NotImplementedError

        if diff_length > 0:
            bonafide = samples.get_group("bonafide").sample(diff_length, replace=True)
            self.samples = pd.concat([self.samples, bonafide], ignore_index=True)
            # print(len(bonafide), spoof_length)
        

    def undersample_dataset(self):
        samples = self.samples.groupby(by=['label'])
        
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        if spoof_length < bona_length:
            raise NotImplementedError

        if spoof_length > bona_length:
            spoofs = samples.get_group("spoof").sample(bona_length, replace=False)
            self.samples = pd.concat([samples.get_group("bonafide"), spoofs], ignore_index=True)
            # print(f"Length of bonafide: {bona_length}")
            # print(f"Length of spoof after undersampling: {len(spoofs)}")

    def get_bonafide_only(self):
        samples = self.samples.groupby(by=['label'])
        self.samples = samples.get_group("bonafide")
        return self.samples

    def get_spoof_only(self):
        samples = self.samples.groupby(by=['label'])
        self.samples = samples.get_group("spoof")
        return self.samples

