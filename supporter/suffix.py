import time


def get_time_suffix(suffix: str = None):
    if suffix is None or suffix == '':
        return time.strftime("%m%d_%H%M%S", time.localtime())
    return suffix
