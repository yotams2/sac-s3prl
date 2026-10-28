"""
s3prl upstream for GRAM-T (https://github.com/labhamlet/GRAM-T), used as a binaural baseline
against the binaural SAC-SSAMBA model.

The encoder itself lives in sac-ssamba/src/gramt_wrapper so the same preprocessing is shared with
the run_amba.py (Speech Commands) pipeline. Set SAC_SSAMBA_SRC if sac-ssamba is not a sibling of
this s3prl checkout.

Binaural [2, T] waveforms are encoded as-is; mono [T] waveforms are duplicated to both ears.
"""
import os
import sys

import torch
from torch.nn.utils.rnn import pad_sequence

SAMPLE_RATE = 16000

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SAC_SSAMBA_SRC = os.environ.get(
    "SAC_SSAMBA_SRC", os.path.join(os.path.dirname(_REPO_ROOT), "sac-ssamba", "src")
)


class UpstreamExpert(torch.nn.Module):
    def __init__(self, model_id: str, **kwargs):
        super().__init__()
        if SAC_SSAMBA_SRC not in sys.path:
            sys.path.insert(0, SAC_SSAMBA_SRC)
        from gramt_wrapper import GRAMTEncoder

        self.model = GRAMTEncoder(model_id=model_id, input_sample_rate=SAMPLE_RATE)

    def get_downsample_rates(self, key: str = None) -> int:
        return self.model.downsample_rate

    def forward(self, wavs):
        hidden_states, n_tokens = self.model(wavs)
        hidden_states = [
            pad_sequence([h[i, :n] for i, n in enumerate(n_tokens)], batch_first=True)
            for h in hidden_states
        ]
        return {"hidden_states": hidden_states}
