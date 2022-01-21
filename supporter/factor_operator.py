import pandas as pd
import numpy as np
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.neu import get_neutralize_sector, get_neutralize_sector_size, get_winsorize, get_standardize


# def keep_pool_stk(df: pd.DataFrame, pool_multi: pd.DataFrame):
#     """
#     只保留股池内的因子值，其余为空
#     :param df: 原始的因子值面板，dtype=float
#     :param pool_multi: 股池乘子，1或nan，dtype=float
#     :return: 新因子面板，不在股池内则为空值，在则保留原始的因子值
#     """
#     return df.reindex_like(pool_multi) * pool_multi


def read_single_factor(path: str, begin_date=None, end_date=None, dtype=None, pool_multi=None, hdf_k=None) -> pd.DataFrame:
    """
    从磁盘csv文件读取2D格式的因子面板
    :param path: 因子csv文件地址
    :param begin_date: 可选，开始日期（注意预留若干timedelta）
    :param end_date: 可选，结束日期
    :param dtype: 可选，指定除0行0列外的类型，必须统一（解决自动识别类型不统一）
    :param pool_multi: 可选，股池乘子，1或nan，dtype=float
    :return: 因子面板
    """
    if hdf_k is None:
        df = pd.read_csv(path, index_col=0, parse_dates=True, dtype=dtype)
    else:
        df = pd.read_hdf(path, index_col=0, parse_dates=True, dtype=dtype, key=hdf_k)

    if pool_multi is not None:
        df = df.reindex_like(pool_multi) * pool_multi  # keep_pool_stk(df, pool_multi)
    if begin_date is not None:
        df = df.loc[begin_date:]
    if end_date is not None:
        df = df.loc[:end_date]
    return df if dtype is None else df.astype(dtype)


def factor_neutralization(df: pd.DataFrame, neu_mtd='n', ind_path=None, mv_path=None) -> pd.DataFrame:
    """
    返回单因子fval中性化后的结果
    :param neu_mtd: 中性化方式，仅标准化(n)，按行业(i)，行业+市值(iv)
    :param ind_path: 行业分类2D面板，由get_data.py获取，tradingdate,stockcode,<str>
    :param mv_path: 市值2D面板，由get_data.py获取，tradingdate,stockcode,<float>
    :param df: 待中性化的因子值面板
    :return: 中性化后的因子

    """
    fv0 = get_standardize(get_winsorize(df))  # standardized factor panel
    fv1 = pd.DataFrame()  # result factor panel
    # plt.hist(fv0.iloc[-1]); plt.show()
    if neu_mtd == 'n':
        return fv0
    elif 'i' in neu_mtd:
        indus = pd.read_csv(ind_path, index_col=0, parse_dates=True, dtype=str)
        indus = indus.reindex_like(df).fillna(method='ffill')  # 新交易日未知行业，沿用上一交易日
        if neu_mtd == 'i':  # 行业中性化
            print('NEU INDUS...')
            fv1 = get_neutralize_sector(fv0, indus)
        elif neu_mtd == 'iv':  # 行业&市值中性化
            size = read_single_factor(mv_path)
            size = size.reindex_like(df)
            lnsize = size.apply(np.log)
            # stdlnsize = get_standardize(get_winsorize(lnsize))
            print('NEU INDUS & MKTVAL...')
            fv1 = get_neutralize_sector_size(fval=fv0, indus=indus, stdlnsize=lnsize)
        return fv1
    #
    return fv1

