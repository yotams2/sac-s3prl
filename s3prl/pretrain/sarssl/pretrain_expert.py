import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Inject SAR-SSL into path
SAR_SSL_PATH = '/storage/yotam/SAR-SSL/SAR-SSL/code'
if SAR_SSL_PATH not in sys.path:
    sys.path.append(SAR_SSL_PATH)

from model import SARSSL
from pretrain.sarssl.dataset import OnlineAcousticDataset

class UpstreamPretrainExpert(nn.Module):
    """
    The SARSSL pretrain expert natively integrated with S3PRL runner
    """

    def __init__(self, datarc, upstream_config, device='cuda', multi_gpu=False, **kwargs):
        super(UpstreamPretrainExpert, self).__init__()

        self.datarc = datarc
        self.device = device
        self.multi_gpu = multi_gpu

        # Load yaml config
        if type(upstream_config) == str:
            self.upstream_config = yaml.load(open(upstream_config, 'r'), Loader=yaml.FullLoader)
            print('[UpstreamPretrainExpert] - Using upstream config from:', upstream_config)
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            print('[UpstreamPretrainExpert] - Using upstream config from the previous experiment.')
        else:
            raise ValueError
        
        # Audio params exactly like our hub wrapper and original code
        self.win_len = 512
        self.win_shift_ratio = 0.5
        self.nfft = 512
        self.win_shift = int(self.win_len * self.win_shift_ratio)
        nf = self.nfft // 2
        
        self.spec_dembed = self.upstream_config.get('spec_dembed', 512)
        self.spat_dembed = self.upstream_config.get('spat_dembed', 256)
        self.spectral_enc_input = self.upstream_config.get('spectral_enc_input', 'all')
        
        # Init dataloader
        print('[UpstreamPretrainExpert] - Using online preprocessor, on-the-fly STFT feature extraction later during forward()')
        self._get_train_dataloader()

        print('[UpstreamPretrainExpert] - Initializing SARSSL model for pretraining...')
        
        # Note: Pretrain=True. When Pretraining, SARSSL forces sequence masking.
        self.model = SARSSL(
            sig_shape=(nf, 100, 2, 2), # Dummy temporal len
            pretrain=True, 
            pretrain_frozen_encoder=False, 
            spec_dembed=self.spec_dembed,
            spat_dembed=self.spat_dembed,
            spectral_enc_input=self.spectral_enc_input
        )

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print('[UpstreamPretrainExpert] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[UpstreamPretrainExpert] - Number of parameters: ' + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def _get_train_dataloader(self):
        # We don't use a specific kaldi extracter because we compute STFT directly in batch process.
        dataset = OnlineAcousticDataset(
            extracter=None,
            task_config=self.upstream_config.get('task', {}),
            bucket_size=self.datarc['train_batch_size'],
            target_level=self.upstream_config.get('audio', {}).get('target_level', -25),
            **self.datarc
        )
        # S3PRL requires dataset bucketing natively which takes batch_size=1 and collates sequences dynamically
        self.dataloader = DataLoader(dataset, batch_size=1,
                                     shuffle=True, num_workers=self.datarc['num_workers'],
                                     drop_last=False, pin_memory=True, collate_fn=dataset.collate_fn)

    # Interface
    def load_model(self, init_ckpt):
        model_state = init_ckpt.get('model', init_ckpt.get('model_state_dict', None))
        if model_state is not None:
            if self.multi_gpu:
                self.model.module.load_state_dict(model_state, strict=False)
            else:
                self.model.load_state_dict(model_state, strict=False)

    # Interface
    def loss_to_device(self):
        pass # Not applicable for SARSSL

    # Interface
    def add_state_to_save(self, all_states):
        # SARSSL original codebase saves under 'model' key
        all_states['model'] = self.model.state_dict() if not self.multi_gpu else \
                              self.model.module.state_dict()
        all_states['Upstream_Config'] = self.upstream_config
        return all_states

    # Interface
    def get_train_dataloader(self):
        return self.dataloader

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs):
        """
        S3PRL `run_pretrain.py` expects this forward implementation matching tera/mockingjay standard.
        """
        # Data is just the batched pad_sequence waveforms directly from the OnlineAcousticDataset __getitem__
        wavs_padded = data.to(self.device) # (batch_size, max_len)
        
        batch_size, max_len = wavs_padded.shape
        # Fake Stereo
        wavs_stereo = wavs_padded.unsqueeze(-1).expand(-1, -1, 2) # (batch_size, max_len, 2)
        window = torch.hann_window(window_length=self.win_len, device=self.device)
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
        ) 
        
        nf = stft_out.shape[1]
        nt = stft_out.shape[2]
        
        stft_out = stft_out.view(batch_size, 2, nf, nt)
        stft_formatted = torch.stack((stft_out.real, stft_out.imag), dim=-1) # (nbatch, 2, nf, nt, 2)
        
        sarssl_model = self.model.module if self.multi_gpu else self.model
        
        # Re-dimension dynamic output frames to prevent downstream folding mismatches
        for module in sarssl_model.modules():
            if hasattr(module, 'output_shape') and isinstance(module.output_shape, tuple):
                module.output_shape = (nf, nt)
                
        # SARSSL pretrain forward pass returns tuple: (loss, diff, data_vis, loss_cl_spat, loss_cl_spec, obt_loss, loss_gen, loss_feat_sim, loss_spec_feat_sim)
        results = sarssl_model(stft_formatted)
        
        loss = results[0] # Aggregated loss for S3PRL to call backward() on
        data_vis = results[2] # Dict containing 'mask', 'pred', 'tar' (reproduced images/mats)
        
        if global_step % log_step == 0 and data_vis is not None:
            # We will plot SARSSL internal variables to S3PRL tensorboard records iteratively
            if 'mask' in data_vis: records['mask_spec'] = data_vis['mask']
            if 'pred' in data_vis: records['pred_spec'] = data_vis['pred']
            if 'tar' in data_vis: records['true_spec'] = data_vis['tar']
            if 'debug_stats' in data_vis:
                for label, val in data_vis['debug_stats'].items():
                    # Manually tracking debug losses might not fit image type plotting, skipping to avoid tensorboard crashes. S3PRL scalar tracking operates elsewhere
                    pass 
            
        return loss, records

    # interface
    def on_before_zero_grad(self):
        pass
    
    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        from utility.audio import plot_spectrogram_to_numpy
        # S3PRL tensorboard integration requires images
        for key, values in records.items():
            if torch.is_tensor(values):
                values = values.cpu().detach().numpy()
            
            # SARSSL produces [B, F, T] patches, just sum pool feature dimensions to render 2D specs
            if len(values.shape) > 2:
                # Plot the first item in the batch
                values = plot_spectrogram_to_numpy(values[0].sum(axis=-1)) # Hueristic
            else:
                values = plot_spectrogram_to_numpy(values)

            logger.add_image(
                f'{prefix}{key}',
                values,
                global_step=global_step
            )
