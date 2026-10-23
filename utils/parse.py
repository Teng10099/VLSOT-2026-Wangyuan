import argparse
def dict_to_namespace(d: dict):
    """
    Convert a dict to a namespace.
    Args:
        d (dict): input dict.
    """
    n = argparse.Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(n, k, dict_to_namespace(v))
        else:
            setattr(n, k, v)
    return n