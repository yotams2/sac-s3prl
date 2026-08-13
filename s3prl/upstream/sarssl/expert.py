import sys
import os
from collections import OrderedDict
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

# Need to append the original SAR-SSL codebase to path to load the model properly, 
# because torch.load() might look for the specific SARSSL classes during unpickling or we need to instantiate it.
# We will do this dynamically here.
SAR_SSL_PATH = '/storage/yotam/SAR-SSL/SAR-SSL/code'
if SAR_SSL_PATH not in sys.path:
    sys.path.append(SAR_SSL_PATH)

from model import SARSSL

class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        super().__init__()
        self.name = "SARSSL"
        
        print(f"[{self.name}] - Loading checkpoint: {ckpt}")
        if ckpt is None:
            raise ValueError("No checkpoint specified for SARSSL.")
            
        # Parse external yaml config if supplied (e.g. from the -g flag in run_downstream)
        if model_config is not None and os.path.exists(model_config):
            import yaml
            with open(model_config, 'r') as f:
                config_overrides = yaml.load(f, Loader=yaml.FullLoader)
            kwargs.update(config_overrides)
            
        # Load the checkpoint
        checkpoint = torch.load(ckpt, map_location='cpu')
        
        # We need to recreate the model with the exact same arguments used during pretraining.
        # Checkpoint usually saves "Args" or "args"
        args_dict = checkpoint.get('Args', checkpoint.get('args', None))
        
        # Merge kwargs to allow user overriding parameters like 'embed_use4ds' from CLI/hubconf
        if args_dict is None:
            print("Warning: Could not find Args in checkpoint, using default SARSSL parameters")
            # Default fallback parameters (should match the WSJ pretraining)
            fs = 16000
            win_len = 512
            win_shift_ratio = 0.5
            nfft = 512
            nf = nfft // 2
            
            # Incorporate kwargs to override defaults
            spec_dembed = kwargs.get('spec_dembed', 512)
            spat_dembed = kwargs.get('spat_dembed', 256)
            spectral_enc_input = kwargs.get('spectral_enc_input', 'all')
            embed_use4ds = kwargs.get('embed_use4ds', 'spec_spat')
            
            self.model = SARSSL(
                sig_shape=(nf, 100, 2, 2), # nt was 0, causing assertion logic below `nt/patch_shape[1]` to crash. Provide dummy positive int.
                pretrain=False, 
                pretrain_frozen_encoder=False, # Disable pretrain masking logic for downstream evaluating
                spec_dembed=spec_dembed, 
                spat_dembed=spat_dembed, 
                spectral_enc_input=spectral_enc_input,
                embed_use4ds=embed_use4ds
            )
        else:
            # Reconstruct arguments from the saved namespace/dict
            if not isinstance(args_dict, dict):
                args_dict = vars(args_dict)
                
            # Allow kwargs to override checkpoint arguments
            spec_dembed = kwargs.get('spec_dembed', args_dict.get('spec_dembed', 512))
            spat_dembed = kwargs.get('spat_dembed', args_dict.get('spat_dembed', 256))
            spectral_enc_input = kwargs.get('spectral_enc_input', args_dict.get('spectral_enc_input', 'all'))
            embed_use4ds = kwargs.get('embed_use4ds', args_dict.get('embed_use4ds', 'spec_spat'))
            
            nf = 512 // 2 # Hardcoded nfft=512 from run_pretrain.py
            self.model = SARSSL(
                sig_shape=(nf, 100, 2, 2), # Dummy positive temporal length
                pretrain=False, 
                pretrain_frozen_encoder=False, # Disable pretrain masking logic for downstream evaluating
                spec_dembed=spec_dembed,
                spat_dembed=spat_dembed,
                spectral_enc_input=spectral_enc_input,
                embed_use4ds=embed_use4ds
            )

        # The state dict might be wrapped in 'module.' if it was trained with DataParallel/DDP
        # Looking at SAR-SSL learner.py, the key used is 'model'
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint.get('state_dict', None)))
        if state_dict is None:
            raise KeyError("Could not find state_dict in the checkpoint.")
            
        # Clean state_dict keys if necessary
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        
        # Audio feature extraction parameters used during SARSSL training
        self.win_len = 512
        self.win_shift_ratio = 0.5
        self.nfft = 512
        self.win_shift = int(self.win_len * self.win_shift_ratio)

    def get_downsample_rates(self, key: str) -> int:
        """
        Since the STFT uses a window shift of 256 (512 * 0.5), 
        for 16kHz audio the downsample rate is exactly the window shift.
        """
        return self.win_shift

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        wavs: list of unpadded wavs [wav1, wav2, ...]
        Each wav is sequence length of 1D tensor [seq_len]
        """
        device = wavs[0].device
        
        # Standard S3PRL downstream tasks pass 1D mono audio.
        # SARSSL expects multichannel STFT input of shape: (nbatch, nmic, nf, nt, nreim)
        # We need to compute the STFT for each batch item.
        
        # First, pad the 1D waveforms
        wavs_padded = pad_sequence(wavs, batch_first=True) # (batch_size, max_len)
        batch_size, max_len = wavs_padded.shape
        
        # Duplicate mono to stereo (2 mics) since SARSSL was trained on multichannel
        wavs_stereo = wavs_padded.unsqueeze(-1).expand(-1, -1, 2) # (batch_size, max_len, 2)
        
        # Compute STFT manually as done in STFTLearner / utils_module.STFT
        # We use torch.stft on the batched data
        
        # To match the model's STFT parameters:
        window = torch.hann_window(window_length=self.win_len, device=device)
        
        # STFT output is complex: (batch_size, nf, nt) -> we need real and imag parts
        # torch.stft takes 2D inputs, so we reshape (batch_size * 2, max_len)
        wavs_flat = wavs_stereo.transpose(1, 2).reshape(batch_size * 2, max_len)
        
        stft_out = torch.stft(
            wavs_flat, 
            n_fft=self.nfft, 
            hop_length=self.win_shift, 
            win_length=self.win_len,
            window=window, 
            center=False, 
            normalized=False, 
            return_complex=True
        ) # (batch_size * 2, nf, nt)
        
        nf = stft_out.shape[1]
        nt = stft_out.shape[2]
        
        # Reshape back to (batch_size, 2, nf, nt)
        stft_out = stft_out.view(batch_size, 2, nf, nt)
        
        # SARSSL shape expectation: (nbatch, nmic, nf, nt, nreim)
        # where nreim = 2 (real, imag)
        stft_real = stft_out.real
        stft_imag = stft_out.imag
        
        # Stack real and imag at the last dimension
        stft_formatted = torch.stack((stft_real, stft_imag), dim=-1) # (nbatch, 2, nf, nt, 2)
        
        # SARSSL model wrapper forward expects input passed directly to the encoders
        # But wait, SARSSL.forward when pretrain=False still needs the patch split
        # Let's see how it's done: 'x' is (nbatch, nmic, nf, nt, nreim)
        
        # Update any dummy PatchRecover layers inside the model with the actual dynamic nf and nt
        for module in self.model.modules():
            # PatchRecover defines `output_shape` and uses it in `forward()` to reconstruct frames
            if hasattr(module, 'output_shape') and isinstance(module.output_shape, tuple):
                module.output_shape = (nf, nt)
                
        with torch.no_grad():
            x = stft_formatted # (nbatch, nmic, nf, nt, nreim)
            data = x.permute(0, 2, 3, 4, 1) # (nbatch, nf, nt, nreim, nmic)
            vec_patch = self.model.patch_split(data)  # (nbatch, npatch, dpatch, nreim, nmic)
            
            nbatch = vec_patch.shape[0]
            npatch = vec_patch.shape[1]
            
            if self.model.in_ver in ['separate', 'same']:
                vec_patch_reshape = vec_patch.reshape(nbatch, npatch, -1) # (nbatch, npatch, dpatch*nch) 
                
                if self.model.spectral_enc_input == 'mono':
                    vec_patch_spec = vec_patch[:, :, :, :, 0].reshape(nbatch, npatch, -1)
                    embed_spec, hs_spec = self.model.spec_encoder(vec_patch_spec)
                else:
                    embed_spec, hs_spec = self.model.spec_encoder(vec_patch_reshape)  # (nbatch, npatch, dembed)
                
                embed_spat, hs_spat = self.model.spat_encoder(vec_patch_reshape, add_same_one=False)  # (nbatch, npatch, dembed)
                
            elif self.model.in_ver == 'single_ch_each_patch':
                vec_patch_reshape = torch.cat([vec_patch[:,:,:,:,0], vec_patch[:,:,:,:,1]], dim=1) # (nbatch, npatch*nmic, dpatch, nreim)
                vec_patch_reshape = vec_patch_reshape.reshape(nbatch, npatch*2, -1)  # (nbatch, npatch*nmic, dpatch*nreim)
                embed_spec, hs_spec = self.model.spec_encoder(vec_patch_reshape)  # (nbatch, npatch*nmic, dembed/nmic)
                embed_spec = torch.cat([embed_spec[:, 0:npatch, :], embed_spec[:, npatch:npatch*2, :]], dim=2) # (nbatch, npatch, dembed)
                hs_spec = [torch.cat([h[:, 0:npatch, :], h[:, npatch:npatch*2, :]], dim=2) for h in hs_spec]

                embed_spat, hs_spat = self.model.spat_encoder(vec_patch_reshape)  # (nbatch, npatch*nmic, dembed/nmic)
                embed_spat = torch.cat([embed_spat[:, 0:npatch, :], embed_spat[:, npatch:npatch*2, :]], dim=2) # (nbatch, npatch, dembed)
                hs_spat = [torch.cat([h[:, 0:npatch, :], h[:, npatch:npatch*2, :]], dim=2) for h in hs_spat]
                
            # Discard CLS token sequences if used
            if self.model.use_cls:
                embed_spec = embed_spec[:, :-1, :]
                embed_spat = embed_spat[:, :-1, :]
                if self.model.ds_token == 'cls':
                    hs_spec = [h[:, -1:, :] for h in hs_spec]
                    hs_spat = [h[:, -1:, :] for h in hs_spat]
                else:
                    hs_spec = [h[:, :-1, :] for h in hs_spec]
                    hs_spat = [h[:, :-1, :] for h in hs_spat]
                
            if self.model.embed_use4ds == 'spec_spat':
                embed = torch.cat([embed_spec, embed_spat], dim=-1)
                hidden_states = [torch.cat([s_spec, s_spat], dim=-1) for s_spec, s_spat in zip(hs_spec, hs_spat)]
            elif self.model.embed_use4ds == 'spec':
                embed = embed_spec
                hidden_states = hs_spec
            elif self.model.embed_use4ds == 'spat':
                embed = embed_spat
                hidden_states = hs_spat
            else:
                embed = torch.zeros_like(embed_spec).detach()
                hidden_states = [torch.zeros_like(h).detach() for h in hs_spec]
            
        # S3PRL expects "hidden_states" to be a list of sequence features (batch, sequence_length, hidden)
        return {
            "hidden_states": hidden_states,
        }
