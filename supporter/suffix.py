import time


def get_suffix(suffix: None):
    if suffix is None or suffix == '':
        return time.strftime("%m%d_%H%M%S", time.localtime())
    return suffix
