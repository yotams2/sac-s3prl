import os
import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio

from pretrain.bucket_dataset import FeatDataset
from pretrain.mockingjay.task import generate_masked_acoustic_model_data

class OnlineAcousticDataset(FeatDataset):
    """
    Dataloader that loads raw audio directly using torchaudio, 
    pads varying-length sequences, and yields (batch_size, seq_len) waveforms.
    """
    def __init__(self, extracter, task_config, bucket_size, file_path, sets, 
                 max_timestep=0, libri_root=None, target_level=-25, **kwargs):
        # We need to normalize temporal length since bucket lengths are usually based on 10ms frame rates (like 160 samples hop).
        max_timestep *= 160
        super(OnlineAcousticDataset, self).__init__(extracter, task_config, bucket_size, file_path, sets, 
                                                    max_timestep, libri_root, **kwargs)
        self.target_level = target_level
        self.sample_length = self.sample_length * 160
    
    def _normalize_wav_decibel(self, wav):
        '''Normalize the signal to the target level'''
        if self.target_level == 'None':
            return wav
        rms = wav.pow(2).mean().pow(0.5)
        scalar = (10 ** (self.target_level / 20)) / (rms + 1e-10)
        wav = wav * scalar
        return wav

    def _load_feat(self, feat_path):
        if self.libri_root is None:
            # Assumes pre-extracted features
            import numpy as np
            return torch.FloatTensor(np.load(os.path.join(self.root, feat_path)))
        else:
            wav, _ = torchaudio.load(os.path.join(self.libri_root, feat_path))
            wav = self._normalize_wav_decibel(wav.squeeze())
            return wav # (seq_len)

    def __getitem__(self, index):
        # Load raw mono acoustic signals, pad sequence
        # We will handle STFT directly inside `pretrain_expert` using the padded batch to minimize memory load here
        x_batch = [self._sample(self._load_feat(x_file)) for x_file in self.X[index]]
        x_pad_batch = pad_sequence(x_batch, batch_first=True) # (batch_size, seq_len)
        
        # S3PRL training relies on returning complex tuples, but for SARSSL we just need the padded features
        return x_pad_batch
