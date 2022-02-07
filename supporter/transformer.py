"""
中性化模块

"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.regression.linear_model import OLS


def get_winsorize_sr(sr: pd.Series, nsigma=3) -> pd.Series:
    """对series缩尾"""
    df = sr.copy()
    md = df.median()
    mad = 1.483 * df.sub(md).abs().median()
    up = df.apply(lambda k: k > md + mad * nsigma)
    down = df.apply(lambda k: k < md - mad * nsigma)
    df[up] = df[up].rank(pct=True).multiply(mad * 0.5).add(md + mad * (0.5 + nsigma))
    df[down] = df[down].rank(pct=True).multiply(mad * 0.5).add(md - mad * (0.5 + nsigma))
    return df


def get_standardize(df):
    """按日标准化"""
    if (df.notnull().sum(axis=1) <= 1.).any():
        import warnings
        warnings.warn('there are {} days only has one notna data'.format((df.notnull().sum(axis=1) <= 1.).sum()))
    #
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, 1), axis=0)
        # 注意：有时会出现某一天全部为0的情况。此时标准化会带来问题


def get_winsorize(df, nsigma=3):
    """去极值缩尾"""
    df = df.copy().astype(float)
    md = df.median(axis=1)
    mad = 1.483 * (df.sub(md, axis=0)).abs().median(axis=1)
    up = df.apply(lambda k: k > md + mad * nsigma)
    down = df.apply(lambda k: k < md - mad * nsigma)
    df[up] = df[up].rank(axis=1, pct=True).multiply(mad * 0.5, axis=0).add(md + mad * (0.5 + nsigma), axis=0)
        # mad*nsigma后0.5mad大小分布
    df[down] = df[down].rank(axis=1, pct=True).multiply(mad * 0.5, axis=0).add(md - mad * (0.5 + nsigma), axis=0)
        # -1*mad*nsigma前0.5mad大小分布

    return df


def get_neutralize_sector_size(fval: pd.DataFrame, stdlnsize: pd.DataFrame, indus: pd.DataFrame) -> pd.DataFrame:
    """单因子行业和市值中性化"""
    factoro = fval[~fval.index.duplicated()].copy()
    factoro[np.isinf(factoro)] = np.nan
    cols = [i for i in factoro.columns if (i in stdlnsize.columns) and (i in indus.columns)]
    factoro = factoro.loc[:, cols]
    dic = {}
    date = factoro.index[100]
    for date in tqdm(factoro.index):
        try:
            industry = indus.loc[date, cols]
        except:
            break
        #indus = indus.loc[date, cols]
        z = pd.get_dummies(industry)
        s = stdlnsize.loc[date, cols]

        x = pd.concat([z, s], axis=1, sort=True)
        y = factoro.loc[date].sort_index(ascending=True)
        mask = (y.notnull()) & (x.notnull().all(axis=1))

        x1 = x[mask]
        y1 = y[mask]

        if len(y1) == 0:
            continue
        else:
            est = OLS(y1, x1).fit()
            dic[date] = est.resid
    #
    newfval = get_standardize(get_winsorize(pd.DataFrame.from_dict(dic, 'index')))
    newfval = newfval.reindex_like(fval)
    newfval[fval.isnull()] = np.nan
    #
    return newfval


def get_neutralize_sector(fval: pd.DataFrame, indus: pd.DataFrame) -> pd.DataFrame:
    """单因子面板的行业中性化"""
    factoro = fval.copy()
    dic = {}
    date = factoro.index[0]
    for date in tqdm(factoro.index):
        #未控制factoro的日期切片#
        try:
            x_indus = indus.loc[date, :]
        except:
            break  # indus里不含有该日期，即已经超出了范围（不限制enddate，日期为最新的交易）
        # mask
        x = pd.get_dummies(x_indus)
        y = factoro.loc[date, :]
        mask = y.notnull() & x.notnull().all(axis=1)
        x1 = x.sort_index()[mask]
        y1 = y.sort_index()[mask].astype('float')
        if len(y1) == 0:
            continue
        # 用行业所无法解释的residual部分
        est = OLS(y1, x1).fit()
        dic[date] = est.resid
        #
    newfactor = get_standardize(get_winsorize(pd.DataFrame.from_dict(dic, 'index')))# .reindex_like(factoro)
    newfactor[factoro.reindex_like(newfactor).isnull()] = np.nan

    return newfactor