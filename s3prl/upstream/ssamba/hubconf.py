# -*- coding: utf-8 -*-
# @Time    : 8/25/21 5:25 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : hubconf.py

# Authors
# - Leo

import os
from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert

def _get_window_secs(default_secs, provided_secs):
    if "SSAMBA_WINDOW_SECS" in os.environ:
        return float(os.environ["SSAMBA_WINDOW_SECS"])
    return provided_secs

# Frame-based SSAST
# 1s for speech commands, 6s for IEMOCAP, 10s for SID
def ssamba_baseline(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    window_secs = _get_window_secs(10.0, window_secs)
    ckpt = "/storage/yotam/ssamba/src/pretrain/exp/amba-base-f16-t16-b16-lr1e-4-m300-pretrain_joint-librispeech/models/best_audio_model.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)

def ssamba_sac(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    window_secs = _get_window_secs(10.0, window_secs)
    ckpt = "/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-m300-lam1.0-tau0.3-sig1.0-librispeech/models/best_audio_model.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)

def ssamba_sac_feat_sid(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    window_secs = _get_window_secs(10.0, window_secs)
    ckpt = "/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-lam1.0-sig1.0-feat_sid-librispeech/models/best_audio_model.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)

def ssamba_sac_feat_sid_lam0_02_sig2_0(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    window_secs = _get_window_secs(10.0, window_secs)
    ckpt = "/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-lam0.02-sig2.0-feat_sid-librispeech/models/best_audio_model.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)

def ssamba_sac_feat_emo_lam0_02_sig2_0(refresh: bool = False, window_secs: float = 6.0, **kwargs):
    window_secs = _get_window_secs(6.0, window_secs)
    ckpt = "/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-lam0.02-sig2.0-feat_emo-librispeech/models/best_audio_model.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)

def ssamba_sac_feat_universal(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    window_secs = _get_window_secs(10.0, window_secs)
    ckpt = "/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-lam0.02-sig1.0-feat_universal-librispeech/models/best_audio_model.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)

def ssamba_sac_feat_universal_fixed(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    window_secs = _get_window_secs(10.0, window_secs)
    ckpt = "/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-lam0.02-sig1.0-feat_universal_fixed_norm_layer-librispeech/models/best_audio_model.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)

def ssamba_sac_feat_universal_mode_sqrt_dim(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    window_secs = _get_window_secs(10.0, window_secs)
    ckpt = "/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-lam0.02-sig1.0-feat_universal-mode_sqrt_dim-librispeech/models/best_audio_model.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)

def ssamba_sac_feat_universal_mode_offline_global_median(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    window_secs = _get_window_secs(10.0, window_secs)
    ckpt = "/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-lam0.02-sig1.0-feat_universal-mode_offline_global_median-librispeech/models/best_audio_model.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)

def ssamba_sac_feat_universal_mode_offline_global_median_exp3_multi_query(refresh: bool = False, window_secs: float = 10.0, **kwargs):
    window_secs = _get_window_secs(10.0, window_secs)
    ckpt = "/storage/yotam/ssamba/src/sac/exp/sac-base-f16-t16-b16-lr1e-4-lam0.02-sig1.0-feat_universal-mode_offline_global_median-librispeech-exp3_multi_query/models/best_audio_model.pth"
    return _UpstreamExpert(ckpt, "base_a", window_secs)

def ssamba_local(ckpt, refresh: bool = False, window_secs: float = 10.0, **kwargs):
    window_secs = _get_window_secs(10.0, window_secs)
    if str(ckpt).startswith("http"):
        ckpt = _urls_to_filepaths(ckpt, refresh=refresh)
    return _UpstreamExpert(ckpt, "base_a", window_secs)
