# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import json
from pathlib import Path
from os.path import join as path_join

import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample

SAMPLE_RATE = 16000
SPATIAL_SPLITS = ['pretrain', 'preval', 'pretest']
# Spatial jsons store absolute wav paths under JSON_DATA_ROOT, rewritten to spatial_data_root on load.
JSON_DATA_ROOT = "/storage/yotam/ssamba/data"


def load_spatial_index(spatial_data_root):
    """Maps original IEMOCAP relative paths (as in meta_data) to their binaural simulated wavs."""
    relpath2spatial = {}
    for split in SPATIAL_SPLITS:
        with open(path_join(spatial_data_root, f'iemocap_spatial_{split}.json'), 'r') as f:
            for item in json.load(f)['data']:
                wav = item['wav'].replace(JSON_DATA_ROOT, str(spatial_data_root).rstrip('/'), 1)
                # the first binaural IEMOCAP release sampled sources with replacement: refuse it
                assert item['orig_rel_path'] not in relpath2spatial, (
                    f"{item['orig_rel_path']} is simulated more than once in {spatial_data_root}; "
                    "regenerate binaural IEMOCAP with sac-ssamba/data_generation/regen_iemocap.sh")
                relpath2spatial[item['orig_rel_path']] = wav
    return relpath2spatial


class IEMOCAPDataset(Dataset):
    def __init__(self, data_dir, meta_path, pre_load=True, spatial_data_root=None):
        """
        spatial_data_root: if set, each utterance listed in meta_data is loaded from its binaural
        [2, T] simulation (iemocap_spatial_*.json under this dir) instead of the original mono wav.
        Labels, folds and splits still come from meta_data.
        """
        self.data_dir = data_dir
        self.pre_load = pre_load
        with open(meta_path, 'r') as f:
            self.data = json.load(f)
        self.class_dict = self.data['labels']
        self.idx2emotion = {value: key for key, value in self.class_dict.items()}
        self.class_num = len(self.class_dict)
        self.meta_data = self.data['meta_data']
        self.relpath2spatial = load_spatial_index(spatial_data_root) if spatial_data_root else None
        if self.relpath2spatial is not None:
            missing = [info['path'] for info in self.meta_data if info['path'] not in self.relpath2spatial]
            assert not missing, (f'{len(missing)} utterances have no binaural simulation, e.g. {missing[0]}; '
                                 'regenerate binaural IEMOCAP with sac-ssamba/data_generation/regen_iemocap.sh')
            print(f'[IEMOCAPDataset] - Loading binaural IEMOCAP from {spatial_data_root} for {meta_path}')
        _, origin_sr = torchaudio.load(self._wav_path(self.meta_data[0]['path']))
        self.resampler = Resample(origin_sr, SAMPLE_RATE)
        if self.pre_load:
            self.wavs = self._load_all()

    def _wav_path(self, path):
        if self.relpath2spatial is not None:
            return self.relpath2spatial[path]
        return path_join(self.data_dir, path)

    def _load_wav(self, path):
        wav, _ = torchaudio.load(self._wav_path(path))
        # mono -> [T]; binaural keeps both channels -> [2, T]
        wav = self.resampler(wav).squeeze(0)
        return wav

    def _load_all(self):
        wavforms = []
        for info in self.meta_data:
            wav = self._load_wav(info['path'])
            wavforms.append(wav)
        return wavforms

    def __getitem__(self, idx):
        label = self.meta_data[idx]['label']
        label = self.class_dict[label]
        if self.pre_load:
            wav = self.wavs[idx]
        else:
            wav = self._load_wav(self.meta_data[idx]['path'])
        return wav.numpy(), label, Path(self.meta_data[idx]['path']).stem

    def __len__(self):
        return len(self.meta_data)

def collate_fn(samples):
    return zip(*samples)
