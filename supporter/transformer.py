"""
中性化模块

"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from statsmodels.regression.linear_model import OLS


def visit_2d_v(td, stk, df, shift=0):
    """
    按行、列标访问2D面板，行为stockcode，列为tradingdate，支持shift
    :param td: 日期
    :param stk: 股票
    :param df: 2D因子面板
    :param shift: 滞后若干交易日
    :return: 表值
    """
    td_idx = -1
    try:
        td_idx = df.index.get_loc(td) + shift
    except KeyError:
        print(f'KeyError: ({td}, {stk})')
        return np.nan
    finally:
        if (td_idx < 0) or (td_idx > len(df)):
            return np.nan
        return df.iloc[td_idx, :].loc[stk]


def column_look_up(tgt, src, delay=0, kw='r_1', msg=None, c0='tradingdate', c1='stockcode'):
    """
    从2D面板src获取stacked表tgt关键字kw的列值，根据tradingdate和stockcode
    :param tgt: 目标表，包含列tradingdate和stockcode
    :param src: 来源表，2D，列为tradingdate，行为stockcode
    :param delay: 滞后日
    :param kw: 新增到src的列名
    :param msg: 缺失百分比的提示
    :param c0: 2D中列名(tradingdate)
    :param c1: 2D中行名(stockcode)
    :return: 新加列的表
    """
    print(f'{kw}...')

    # # Method 1
    # tgt[kw] = np.nan
    # for ri in tqdm(tgt.index):
    #     tgt.loc[ri, kw] = visit_2d_v(tgt.loc[ri, c0], tgt.loc[ri, c1], src, shift=delay)

    # # Method 2
    tgt[kw] = tgt[[c0, c1]].apply(lambda s: visit_2d_v(s.iloc[0], s.iloc[1], src, shift=delay), axis=1)

    # # Method 3
    # tmp = src.shift(delay).unstack().reset_index()
    # tmp.columns = [c0, c1, kw]
    # tgt = tgt.merge(tmp, on=[c0, c1], how='left')

    print(f'nan:{tgt[kw].isna().mean() * 100: 6.2f} % {"not found in source table" if msg is None else msg}')
    return tgt


def get_winsorize_sr(sr: pd.Series, nsigma=3) -> pd.Series:
    """对series缩尾"""
    sr1 = sr.copy()
    md = sr1.median()
    mad = 1.483 * sr1.sub(md).abs().median()
    up = sr1.apply(lambda k: k > md + mad * (0 + nsigma))
    down = sr1.apply(lambda k: k < md - mad * (0 + nsigma))
    sr1[up] = sr1[up].rank(pct=True).multiply(mad * 0.5).add(md + mad * (0 + nsigma))
    sr1[down] = sr1[down].rank(pct=True).multiply(mad * 0.5).add(md - mad * (0 + nsigma))
    return sr1


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


def cvg_f_fill(fr: pd.DataFrame, w=10, q=.75, ishow=False, notify=False) -> pd.DataFrame:
    """F-Fill if Low Coverage: 日覆盖率低于过去w日均值的q倍时填充"""
    beta_covered_stk_num = fr.index.get_level_values(0).value_counts().sort_index()
    mask_l_cvg = beta_covered_stk_num < (beta_covered_stk_num.shift(1).rolling(w).mean() * q)
    rep_tds = beta_covered_stk_num[mask_l_cvg]
    # print(rep_tds)
    tds = fr.index.get_level_values(0).unique()
    tds = pd.Series(tds, index=tds)
    # td = rep_tds.index[0]
    for td in rep_tds.index:
        td_1 = tds[:td].iloc[-2]
        td1 = tds[td:].iloc[1]
        if notify:
            print(td.strftime('%Y-%m-%d'), '->', td_1.strftime('%Y-%m-%d'), len(fr.loc[td]), '->', len(fr.loc[td_1]))
        fr = pd.concat([fr.loc[:td_1], fr.loc[td_1:td_1].rename(index={td_1: td}), fr.loc[td1:]])
    if ishow:
        from matplotlib import pyplot as plt
        plt.plot(beta_covered_stk_num)
        plt.plot(fr.index.get_level_values(0).value_counts().sort_index())
        plt.tight_layout()
        plt.title(f'F-Fill Coverage Lower than {q} * (past {w}d mean)')
        plt.show()
        plt.close()
    return fr


def df_union_sub(df0: pd.DataFrame, df1: pd.DataFrame) -> pd.DataFrame:
    """Align index and columns and calculate df0 - df1"""
    assets0 = df0.columns
    assets1 = df1.columns
    views0 = df0.index
    views1 = df1.index
    assets = assets0.union(assets1)
    views = views0.union(views1)
    df0 = pd.DataFrame(df0, columns=assets, index=views).fillna(0)
    df1 = pd.DataFrame(df1, columns=assets, index=views).fillna(0)
    res = df0.sub(df1)
    return res
