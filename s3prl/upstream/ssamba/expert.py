# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : expert.py

# Authors
# - Leo

import logging

import torch

from .audio import FeatureExtractor

logger = logging.getLogger(__name__)

FBANK_SAMPLE_STRIDE = 160
SAMPLE_RATE = 16000


class UpstreamExpert(torch.nn.Module):
    def __init__(
        self,
        ckpt: str,
        model_size: str,
        window_secs: float = 1.0,
    ):
        super().__init__()
        self.window_secs = window_secs
        self.stride_secs = window_secs
        
        
        model_size_parts = model_size.split("_")
        model_size_base = model_size_parts[0]
        model_type = model_size_parts[1] if len(model_size_parts) > 1 else ""
        
        if model_size == "binaural_sac":
            import sys
            import os
            sys.path.insert(0, '/storage/yotam/ssamba/Vim')
            sys.path.insert(0, '/storage/yotam/ssamba/Vim/vim')
            sys.path.insert(0, '/storage/yotam/ssamba/Vim/mamba-1p1p1')
            sys.path.insert(0, '/storage/yotam/ssamba/src')
            from binaural_dual_stream_sac.downstream_experts import MonauralSpectralUpstreamExpert
            self.model = MonauralSpectralUpstreamExpert(model_checkpoint_path=ckpt)
            self.tstride = 16
            self.vertical_num_patches = 16
            self.model = self.model.cpu()
            self.model_size = model_size
            return
            
        from .ast_models import AMBAModel

        try:
            import timm
            if model_type == 'p':
                print("importing ssast")
                from .ast_models import ASTModel
            else:
                print("importing amba")
                from .ast_models import AMBAModel
            
        except:
            print("error occured importing")
            if model_type =='p':
                logger.error(
                    "SSAST requires 'timm==0.4.5' to work. Please run 'pip install timm==0.4.5'"
                )
                exit(1)

        target_length = int(window_secs * SAMPLE_RATE / FBANK_SAMPLE_STRIDE)
        
        self.preprocessor = FeatureExtractor(
            target_length=target_length, apply_cmvn=False
        )
        
        default_vision_mamba_config = {
            'img_size': (128, target_length),
            'patch_size': 16,
            'stride': 10,
            'embed_dim': 768,
            'depth': 24,
            'rms_norm': True,
            'residual_in_fp32': False,
            'fused_add_norm': False,
            'final_pool_type': 'mean',
            'if_abs_pos_embed': True,
            'if_rope': False,
            'if_rope_residual': False,
            'bimamba_type': "v2",
            'if_cls_token': True,
            'if_devide_out': True,
            'use_middle_cls_token': False,
        }
        if model_size == 'base':
            default_vision_mamba_config['embed_dim']=768
        elif model_size == 'small':
            default_vision_mamba_config['embed_dim']=384
        else:
            default_vision_mamba_config['embed_dim']=192
            
        
        assert model_type in ["p", "a"]
        if model_type == "p":
            self.tstride = 10
            self.model = ASTModel(
                fshape=16,
                tshape=16,
                fstride=10,
                tstride=self.tstride,
                input_tdim=target_length,
                input_fdim=128,
                model_size=model_size,
                pretrain_stage=False,
                load_pretrained_mdl_path=ckpt,
            )
            self.vertical_num_patches = (128 - 16) // 10 + 1  # 12
        else:
            self.tstride = 10
            self.model = AMBAModel(
                fshape=16,
                tshape=16,
                fstride=10,
                tstride=self.tstride,
                input_tdim=target_length,
                input_fdim=128,
                model_size=model_size,
                pretrain_stage=False,
                load_pretrained_mdl_path=ckpt,
                vision_mamba_config = default_vision_mamba_config
            )
            self.vertical_num_patches = (128 - 16) // 10 + 1  # 1

        self.model = self.model.cpu()

    def get_downsample_rates(self, key: str = None) -> int:
        if hasattr(self, 'model_size') and self.model_size == 'binaural_sac':
            return 160 * self.tstride
        return int(FBANK_SAMPLE_STRIDE * self.tstride)

    def forward(self, wavs):
        if hasattr(self, 'model_size') and self.model_size == 'binaural_sac':
            device = self.model.spectral_encoder.v.patch_embed.proj.weight.device
            CHUNK_SAMPLES = 1024 * 160  # 163,840 samples (~10.24 seconds)
            max_wav_len = max(wav.shape[-1] for wav in wavs)
            start_points = list(range(0, max(1, max_wav_len), CHUNK_SAMPLES))

            chunk_h_list = []
            for start in start_points:
                chunk_end = start + CHUNK_SAMPLES
                stft_4ch_list = []
                for wav in wavs:
                    if wav.ndim == 2 and wav.shape[0] >= 2:
                        w_slice = wav[:, start:chunk_end]
                        if w_slice.shape[-1] < CHUNK_SAMPLES:
                            w_slice = torch.nn.functional.pad(w_slice, (0, CHUNK_SAMPLES - w_slice.shape[-1]))
                        wL = w_slice[0].cpu().float()
                        wR = w_slice[1].cpu().float()
                        stft_L = torch.stft(wL, n_fft=512, hop_length=160, return_complex=True)
                        stft_R = torch.stft(wR, n_fft=512, hop_length=160, return_complex=True)
                        x_4ch = torch.stack([stft_L.real, stft_L.imag, stft_R.real, stft_R.imag], dim=0)
                    elif wav.ndim == 1:
                        w_slice = wav[start:chunk_end]
                        if w_slice.shape[-1] < CHUNK_SAMPLES:
                            w_slice = torch.nn.functional.pad(w_slice, (0, CHUNK_SAMPLES - w_slice.shape[-1]))
                        w = w_slice.cpu().float()
                        stft = torch.stft(w, n_fft=512, hop_length=160, return_complex=True)
                        re, im = stft.real, stft.imag
                        x_4ch = torch.stack([re, im, re, im], dim=0)
                    else:
                        w_slice = wav.squeeze(0)[start:chunk_end]
                        if w_slice.shape[-1] < CHUNK_SAMPLES:
                            w_slice = torch.nn.functional.pad(w_slice, (0, CHUNK_SAMPLES - w_slice.shape[-1]))
                        w = w_slice.cpu().float()
                        stft = torch.stft(w, n_fft=512, hop_length=160, return_complex=True)
                        re, im = stft.real, stft.imag
                        x_4ch = torch.stack([re, im, re, im], dim=0)

                    # Frequency dimension to 256
                    if x_4ch.shape[1] > 256:
                        x_4ch = x_4ch[:, :256, :]
                    elif x_4ch.shape[1] < 256:
                        x_4ch = torch.nn.functional.pad(x_4ch, (0, 0, 0, 256 - x_4ch.shape[1]))

                    # Time dimension to 1024
                    if x_4ch.shape[2] < 1024:
                        x_4ch = torch.nn.functional.pad(x_4ch, (0, 1024 - x_4ch.shape[2]))
                    elif x_4ch.shape[2] > 1024:
                        x_4ch = x_4ch[:, :, :1024]

                    stft_4ch_list.append(x_4ch)

                batch_x_4ch = torch.stack(stft_4ch_list, dim=0).to(device)
                # Allow gradients when runner is in upstream_trainable mode (-f)
                h = self.model.spectral_encoder._encode_with_mamba(batch_x_4ch)
                h = h[:, 1:, :] # Skip CLS token -> [B, 1024, 768]
                B = batch_x_4ch.size(0)
                D = h.size(-1)
                # Reshape [B, 16, 64, D] -> Permute to [B, 64, 16, D] -> Flatten frequency bands to [B, 64, 16 * D]
                h = h.reshape(B, 16, 64, D).permute(0, 2, 1, 3).reshape(B, 64, 16 * D).float()
                chunk_h_list.append(h)

            all_h = torch.cat(chunk_h_list, dim=1) if len(chunk_h_list) > 1 else chunk_h_list[0]

            from torch.nn.utils.rnn import pad_sequence
            trimmed_hidden_states = []
            for wav_idx, wav in enumerate(wavs):
                w_len = wav.shape[-1]
                max_h_len = min(all_h.size(1), max(1, round(w_len / self.get_downsample_rates())))
                trimmed_hidden_states.append(all_h[wav_idx, :max_h_len, :])

            padded_h = pad_sequence(trimmed_hidden_states, batch_first=True)
            return {"hidden_states": [padded_h.to(device)]}

        wavs_clean = [wav[0] if wav.ndim == 2 else wav for wav in wavs]
        wavs_len = [len(wav) for wav in wavs_clean]
        max_wav_len = max(wavs_len)
        start_points = list(range(0, max_wav_len, int(self.stride_secs * SAMPLE_RATE)))
        padded_max_wav_len = start_points[-1] + int(self.window_secs * SAMPLE_RATE)
        padded_wavs = [
            torch.cat([wav, wav.new_zeros(padded_max_wav_len - len(wav))])
            for wav in wavs_clean
        ]

        all_features = []
        for start in start_points:
            subwavs = [
                wav[start : start + int(self.window_secs * SAMPLE_RATE)]
                for wav in padded_wavs
            ]
            features = [self.preprocessor(wav.unsqueeze(0)) for wav in subwavs]
            features = torch.stack(
                features, dim=0
            )  # (batch_size, segment_seq_len, hidden_size)
            all_features.append(features)

        all_features = torch.stack(all_features, dim=0)
        num_segment, batch_size, segment_seq_len, hidden_size = all_features.shape

        flatten_features = all_features.reshape(-1, segment_seq_len, hidden_size)
        
        if hasattr(self, 'model_size') and self.model_size == 'binaural_sac':
            x_mono = flatten_features.transpose(1, 2).unsqueeze(1)
            x_4ch = x_mono.repeat(1, 4, 1, 1).to(self.model.spectral_encoder.v.patch_embed.proj.weight.device)
            with torch.no_grad():
                h = self.model.spectral_encoder._encode_with_mamba(x_4ch)
            h = h[:, 1:, :] # Skip CLS token
            hidden_states = [h.cpu()]
        else:
            hidden_states, final_repr = self.model(flatten_features)

        reshaped_hidden_states = [
            (
                h.reshape(num_segment, batch_size, -1, h.size(-1))
                .transpose(
                    0, 1
                )  # (batch_size, num_segment, num_horizon_patch, num_vertical_patch * hidden_size)
                .flatten(
                    1, 2
                )  # (batch_size, num_segment * num_horizon_patch, num_vertical_patch * hidden_size)
                .float()
            )
            for h in hidden_states
        ]

        trimmed_hidden_states = []
        for h in reshaped_hidden_states:
            max_h_len = len(range(0, max_wav_len, self.get_downsample_rates()))
            h = h[:, :max_h_len, :]
            trimmed_hidden_states.append(h)

        return {"hidden_states": trimmed_hidden_states}
