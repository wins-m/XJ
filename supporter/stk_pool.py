import pandas as pd
import numpy as np


def get_stk_pool(conf: dict, stk_pool: str, bd=None, ed=None) -> pd.DataFrame:
    """Get multiplier (True: index constituent, nan: not in)"""
    if (stk_pool is None) or (stk_pool == 'NA'):
        stk_pool = 'A'

    df = pd.read_csv(conf['idx_constituent'].format(stk_pool), index_col=0, parse_dates=True)
    if bd is not None:
        df = df[bd:]
    if ed is not None:
        df = df[:ed]

    return (~df.isna()).replace(False, np.nan)


def keep_in_stk_pool(df, in_stk_pool) -> pd.DataFrame:
    """Return df only in stk pool (shape like in_stk_pool and NA when not in)"""
    res = (df.reindex_like(in_stk_pool) * in_stk_pool)  # .dropna(axis=1, how='all')
    print(f'Keep in pool: src{df.shape} -> tgt{res.shape}')
    return res
