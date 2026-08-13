from .expert import UpstreamExpert as _UpstreamExpert

def sarssl(*args, **kwargs):
    """
    Registers the sarssl upstream model in the S3PRL Hub.
    """
    return _UpstreamExpert(*args, **kwargs)
