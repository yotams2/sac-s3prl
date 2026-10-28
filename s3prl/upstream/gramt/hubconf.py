from .expert import UpstreamExpert as _UpstreamExpert


def gramt_binaural(refresh: bool = False, **kwargs):
    """GRAM-T binaural, 16x8 (freq x time) patches. The default variant used in the GRAM-T paper."""
    return _UpstreamExpert("labhamlet/gramt-binaural-frame")


def gramt_binaural_time(refresh: bool = False, **kwargs):
    """GRAM-T binaural, 128x2 full-band time patches."""
    return _UpstreamExpert("labhamlet/gramt-binaural-time")


def gramt_mono(refresh: bool = False, **kwargs):
    """GRAM-T mono. Binaural input is averaged to mono."""
    return _UpstreamExpert("labhamlet/gramt-mono")


def gramt_local(ckpt, refresh: bool = False, **kwargs):
    """ckpt: HF repo id or a local directory containing a downloaded labhamlet/gramt-* snapshot."""
    return _UpstreamExpert(ckpt)
