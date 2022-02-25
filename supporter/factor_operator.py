"""
(created by swmao on Jan. 21st)
因子统一处理，读取、中性化等

"""
import pandas as pd
import numpy as np
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.transformer import get_neutralize_sector, get_neutralize_sector_size, get_winsorize, get_standardize
from matplotlib import pyplot as plt
from scipy import stats

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


def factor_neutralization(fv: pd.DataFrame, neu_mtd='n', ind_path=None, mv_path=None) -> pd.DataFrame:
    """
    返回单因子fval中性化后的结果。依次进行
    - n/i/iv: 去极值（缩尾）｜标准化
    - i/iv: 行业中性化
    - iv: 市值中性化

    :param neu_mtd: 中性化方式，仅标准化(n)，按行业(i)，行业+市值(iv)
    :param ind_path: 行业分类2D面板，由get_data.py获取，tradingdate,stockcode,<str>
    :param mv_path: 市值2D面板，由get_data.py获取，tradingdate,stockcode,<float>
    :param fv: 待中性化的因子值面板
    :return: 中性化后的因子

    """
    fv0 = get_standardize(get_winsorize(fv))  # standardized factor panel
    fv1 = pd.DataFrame()  # result factor panel
    # plt.hist(fv0.iloc[-1]); plt.show()
    if neu_mtd == 'n':
        return fv0
    elif 'i' in neu_mtd:
        indus = pd.read_csv(ind_path, index_col=0, parse_dates=True, dtype=str)
        indus = indus.reindex_like(fv).fillna(method='ffill')  # 新交易日未知行业，沿用上一交易日
        if neu_mtd == 'i':  # 行业中性化
            print('NEU INDUS...')
            fv1 = get_neutralize_sector(fval=fv0, indus=indus)
        elif neu_mtd == 'iv':  # 行业&市值中性化
            size = pd.read_csv(mv_path, index_col=0, parse_dates=True, dtype=float)
            size = size.reindex_like(fv)
            size_ln = size.apply(np.log)
            size_ln_std = size_ln  # size_ln_std = get_standardize(get_winsorize(size_ln))
            print('NEU INDUS & MKT_SIZE...')
            fv1 = get_neutralize_sector_size(fval=fv0, indus=indus, stdlnsize=size_ln_std)
        return fv1
    #
    return fv1


def get_idx_weight(pool_path, begin_date, end_date, mtd='idx'):
    """获取城成分股权重 等权(ew)，按指数成分股权重(idx)，返回值行绝对值之和为1"""
    df = pd.read_csv(pool_path, index_col=0, parse_dates=True, dtype=float)
    df = df.loc[begin_date: end_date]
    df1 = df
    if mtd == 'idx':
        df1 = df.apply(lambda s: s / np.nansum(s), axis=1)
    elif mtd == 'ew':
        df1 = (df.T * 0 + 1 / df.count(axis=1)).T
    assert round(df1.abs().sum(axis=1).prod(), 2) == 1
    return df1


def get_long_short_group(df: pd.DataFrame, ngroups: int, save_path=None) -> pd.DataFrame:
    """
    因子值替换为分组标签
    :param df: 因子值（经过标准化、中性化）
    :param ngroups: 分组数(+)；多头/空头内资产数量取负值(-)，若股池不够大，重叠部分不持有
    :param save_path: 若有，分组情况存到本地
    :return: 因子分组，nan仍为nan，其余为分组编号 0~(分组数-1)
    """
    res = None
    if ngroups < 0:
        cnt = df.count(axis=1)
        nm = (cnt + 2 * ngroups).abs()
        lg = (cnt - nm) / 2  # 上阈值
        hg = lg + nm  # 下阈值
        rnk = df.rank(axis=1)  # , method='first')
        res = rnk * 0 + 1  # 中间组
        res[rnk.apply(lambda s: s <= lg, axis=0)] = 0  # 空头
        res[rnk.apply(lambda s: s >= hg, axis=0)] = 2  # 多头
        # res = rnk * 0  # 一般
        # res[rnk.apply(lambda s: s >= hg, axis=0)] = 1  # 多头
    elif ngroups == 1:
        res = df
    else:
        res = df.rank(axis=1, pct=True).applymap(lambda x: x // (1 / ngroups))

    if save_path is not None:
        res.to_csv(save_path)

    return res


def portfolio_statistics_from_weight(weight, cost_rate, all_ret, save_path=None):
    """对持仓计算结果"""
    res = pd.DataFrame(index=weight.index)
    res['NStocks'] = (weight.abs() > 0).sum(axis=1)
    res['Turnover'] = weight.diff().abs().sum(axis=1)
    res['Return'] = (all_ret.reindex_like(weight) * weight).sum(axis=1)
    res['Return_wc'] = res['Return'] - res['Turnover'] * cost_rate
    res['Wealth(cumsum)'] = res['Return'].cumsum().add(1)
    res['Wealth_wc(cumsum)'] = res['Return_wc'].cumsum().add(1)
    res['Wealth(cumprod)'] = res['Return'].add(1).cumprod()
    res['Wealth_wc(cumprod)'] = res['Return_wc'].add(1).cumprod()
    if save_path:
        res.to_csv(save_path)
    return res


def portfolio_panels_from_weight(w_long, w_short, w_long_short, idx_weight, cost_rate, all_ret, path_format,
                                 save_tables, fbegin_date, fend_date) -> dict:
    """根据持仓和收益,得到组合表现面板"""
    panel_long = portfolio_statistics_from_weight(w_long, cost_rate, all_ret, save_path=path_format.format(
        'PanelLong.csv') if save_tables else None)
    panel_short = portfolio_statistics_from_weight(w_short, cost_rate, all_ret, save_path=path_format.format(
        'PanelShort.csv') if save_tables else None)
    panel_long_short = portfolio_statistics_from_weight(w_long_short, cost_rate, all_ret, save_path=path_format.format(
        'PanelLongShort.csv') if save_tables else None)
    panel_baseline = portfolio_statistics_from_weight(idx_weight.loc[fbegin_date:fend_date], cost_rate, all_ret)
    all_panels = {'long_short': panel_long_short,
                  'long': panel_long,
                  'short': panel_short,
                  'baseline': panel_baseline}

    return all_panels


def panel_long_short_return(ret_group, ret_baseline, save_path=None) -> pd.DataFrame:
    """
    由各分组收益以及基准收益，准备多空绝对收益的面板
    :param ret_group: 各分组收益，最左边因子值最小，做空，最右边因子值最大，做多
    :param ret_baseline: 对比基准的收益，index与ret_group相同
    :param save_path: 包含该参数，则存到指定的csv文件
    :return: 多空绝对收益面板

    """
    long_short_return_nc = pd.DataFrame(index=ret_group.index)
    long_short_return_nc['long_short'] = ret_group.iloc[:, -1] - ret_group.iloc[:, 0]
    long_short_return_nc['long'] = ret_group.iloc[:, -1]  # 因子最大组，做多
    long_short_return_nc['short'] = ret_group.iloc[:, 0]
    long_short_return_nc['baseline'] = ret_baseline

    if save_path is not None:
        long_short_return_nc.to_csv(save_path)

    return long_short_return_nc


def panel_long_short_absolute(long_short_return_nc, ishow=False, title='', save_path=None, cumsum=True) -> pd.DataFrame:
    """多空累计绝对收益面板"""
    if cumsum:
        long_short_absolute_nc = long_short_return_nc.cumsum()  # 用累加
    else:
        long_short_absolute_nc = long_short_return_nc.add(1).cumprod()  # 用累乘

    if save_path is not None:
        long_short_absolute_nc.plot(figsize=(10, 5), grid=True, title=title)
        plt.savefig(save_path)
        if ishow:
            plt.show()
        else:
            plt.close()

    return long_short_absolute_nc


def panel_long_short_excess(long_short_absolute_nc, ishow=False, title='Long-Short Excess Result',
                            save_path=None) -> pd.DataFrame:
    """
    由多头、空头、多空、基准账户，计算时期内超额收益面板
    :param title: 标题
    :param long_short_absolute_nc: 多空累计收益面板
    :param ishow: 是否展示图片
    :param save_path: 非None则存对应文件
    :return: 多空超额收益面板
    """
    long_short_excess_nc = pd.DataFrame(index=long_short_absolute_nc.index)
    long_short_excess_nc['long_short'] = long_short_absolute_nc['long_short'] - long_short_absolute_nc['baseline']
    long_short_excess_nc['long'] = long_short_absolute_nc['long'] - long_short_absolute_nc['baseline']
    long_short_excess_nc['short'] = long_short_absolute_nc['short'] - long_short_absolute_nc['baseline']

    if save_path is not None:
        long_short_excess_nc.plot(figsize=(10, 5), grid=True, title=title)
        plt.savefig(save_path)
        if ishow:
            plt.show()
        else:
            plt.close()

    return long_short_excess_nc


def plot_rtns_group(ret_group: pd.DataFrame, ishow=False, save_path=None, cumsum=True):
    """分层收益情况"""
    if cumsum:
        ret_group_cumulative = ret_group.cumsum()
    else:
        ret_group_cumulative = ret_group.add(1).cumprod()

    if save_path is not None:
        ret_group_cumulative.plot(grid=True, figsize=(16, 8), linewidth=3, title="Group Test Result")
        plt.savefig(save_path)
        if ishow:
            plt.show()
        else:
            plt.close()

    return ret_group_cumulative


def cal_weight_from_long_short_group(long_short_group, ngroups, idx_weight, fbegin_date, fend_date, holddays, ic_mean=0,
                                     save_path=None):
    """Position weight: long, short, long-short 对holdday处理交易频率"""

    def weight_with_holdday(w: pd.DataFrame, hd) -> pd.DataFrame:
        """hd日交易一次，调整持仓权重"""
        return w.iloc[::hd].reindex_like(w).fillna(method='ffill')

    ngroups = 2 if ngroups == 1 else ngroups  # 分为0或1若是同质信号
    long_position = (long_short_group == ngroups - 1) * idx_weight.loc[fbegin_date:fend_date]
    long_position = long_position.apply(lambda s: s / s.sum(), axis=1)
    short_position = (long_short_group == 0) * idx_weight.loc[fbegin_date:fend_date]
    short_position = short_position.apply(lambda s: s / s.sum(), axis=1)
    long_short_position = (long_position - short_position).apply(lambda s: s / s.abs().sum(), axis=1)
    assert round(long_short_position.dropna(how='all').abs().sum(axis=1).prod(), 4) == 1

    w_long = weight_with_holdday(long_position, holddays)
    w_short = weight_with_holdday(short_position, holddays)
    w_long_short = weight_with_holdday(long_short_position, holddays)
    if (ngroups != 1) and (ic_mean < 0):
        w_long, w_short, w_long_short = w_short, w_long, -w_long_short

    if save_path:
        w_long.to_csv(save_path.format('long'))
        w_short.to_csv(save_path.format('short'))
        w_long_short.to_csv(save_path.format('long_short'))

    return w_long, w_short, w_long_short


def cal_long_short_group_rtns(long_short_group, ret, idx_weight, ngroups, save_path=None) -> pd.DataFrame:
    """
    计算各组收益并存储
    :param long_short_group:
    :param ret:
    :param idx_weight:
    :param ngroups:
    :param save_path:
    :return:
    """

    long_short_group = long_short_group.reindex_like(ret)
    if idx_weight is None:
        idx_weight = ret * 0 + 1
    else:
        idx_weight = idx_weight.reindex_like(ret)

    gn = 3 if ngroups < 0 else 2 if ngroups == 1 else ngroups
    ret_group = pd.DataFrame(index=ret.index)
    gi = 0
    for gi in range(gn):
        mask = (long_short_group == gi)
        ret1 = ret[mask]
        w1 = idx_weight[mask].apply(lambda s: s / np.nansum(s), axis=1)
        ret_group['$Group_{' + str(gi) + '}$'] = (ret1 * w1).sum(axis=1)

    if save_path is not None:  # 保存多空分组收益
        ret_group.to_csv(save_path)

    return ret_group


def cal_total_ret_group(ret_group, ishow=False, save_path=None) -> pd.DataFrame:
    """由分组收益面板，计算分组总收益"""
    ret_group_total = ret_group.add(1).prod().add(-1)

    if save_path is not None:
        ret_group_total.plot(figsize=(10, 5), title="Total Return of Group")
        plt.savefig(save_path)
        if ishow:
            plt.show()
        else:
            plt.close()

    return ret_group_total


def cal_sr_max_drawdown(df: pd.Series, ishow=False, title=None, save_path=None, kind='cumprod') -> pd.DataFrame:
    """计算序列回撤"""
    cm = df.cummax()
    mdd = pd.DataFrame(index=df.index)
    mdd[f'{df.name}_maxdd'] = (df / cm - 1) if kind == 'cumprod' else (df - cm)

    if save_path is not None:
        try:
            mdd.plot(kind='area', figsize=(10, 5), grid=True, color='y', alpha=.5, title=title)
        except ValueError:
            mdd[mdd > 0] = 0
            mdd.plot(kind='area', figsize=(10, 5), grid=True, color='y', alpha=.5, title=title)
        finally:
            plt.savefig(save_path)
            if ishow:
                plt.show()
            else:
                plt.close()

    return mdd


def cal_yearly_return(data, ishow=False, title='Annual Return', save_path=None) -> pd.DataFrame:
    """计算收益面板（列为datetime）在历年的总收益面板"""
    year_group = data.index.to_series().apply(lambda dt: dt.year)
    aret = data.groupby(year_group).apply(lambda s: np.prod(s + 1) - 1)

    if save_path is not None:
        aret.plot.bar(figsize=(10, 5), title=title)
        # rolling_yearly_ret.plot(figsize=(10, 5))
        plt.grid(axis='y')
        plt.savefig(save_path)
        if ishow:
            plt.show()
        else:
            plt.close()

    return aret


def cal_yearly_sharpe(data, ishow=False, title='Annual Sharpe', save_path=None, y_len=240) -> pd.DataFrame:
    """计算年度夏普（240天年化）"""
    year_group = data.index.to_series().apply(lambda dt: dt.year)
    asharp = data.groupby(year_group).apply(lambda s: s.mean() / s.std() * np.sqrt(y_len / s.count()))

    if save_path is not None:
        asharp.plot.bar(figsize=(10, 5), title=title)
        # asharp.plot(figsize=(10, 5))
        plt.grid(axis='y')
        plt.savefig(save_path)
        if ishow:
            plt.show()
        else:
            plt.close()

    return asharp


def cal_turnover_long_short(long_short_group, idx_weight, ngroups, ishow=False, save_path=None) -> pd.DataFrame:
    """由多空分组分别计算long, short, long_short, baseline的换手率和权重"""

    def group_weight(df: pd.DataFrame, w: pd.DataFrame, g: int) -> pd.DataFrame:
        """分组内各股权重，空仓则为0：df(分组标签2D), w(全市场权重2D), g(第g组)"""
        return w[df == g].fillna(0)

    def rescale_weight(df: pd.DataFrame, axis=1) -> pd.DataFrame:
        """对每行的权重值标准化，使绝对值总和为1"""
        return df.apply(lambda s: s / np.nansum(s.abs()), axis=axis)

    def turnover_from_weight(w: pd.DataFrame) -> pd.DataFrame:
        """由权重计算双边换手率"""
        return w.diff().abs().sum(axis=1)

    w_long = rescale_weight(group_weight(long_short_group, idx_weight, max(ngroups - 1, 1)))  # 最大编号组，做多
    w_short = rescale_weight(group_weight(long_short_group, idx_weight, 0))  # 0编号组，做空
    w_long_short = rescale_weight(w_long - w_short)  # 多头资金量与空头资金量1:1，合并换手率

    dw = pd.DataFrame()
    dw['long_short'] = turnover_from_weight(w_long_short)
    dw['long'] = turnover_from_weight(w_long)
    dw['short'] = turnover_from_weight(w_short)
    dw['baseline'] = turnover_from_weight(rescale_weight(idx_weight))

    if save_path is not None:
        dw.to_csv(save_path)
        dw[['long', 'short']].plot(figsize=(10, 5), grid=True, title='Turnover')
        plt.savefig(save_path.replace('.csv', '.png'))
        if ishow:
            plt.show()
        else:
            plt.close()

    return dw


def cal_ic(fv_l1, ret, lag=1, ranked=False, ishow=False, save_path=None) -> pd.DataFrame:
    """计算IC：昨日因子值与当日收益Pearson相关系数"""
    mtd = 'spearman' if ranked else 'pearson'
    ic = fv_l1.shift(lag - 1).corrwith(ret, axis=1, drop=False, method=mtd)  # lag-1: fv_l1是滞后的因子
    ic = pd.DataFrame(ic)
    ic.columns = ['IC']

    if save_path is not None:
        ic.plot.hist(figsize=(10, 5), bins=50, title='IC distribution')
        plt.savefig(save_path)
        if ishow:
            plt.show()
        else:
            plt.close()

    return ic


def cal_ic_stat(data):
    """获取IC的统计指标"""
    data = data.dropna()
    t_value, p_value = stats.ttest_1samp(data, 0)  # 计算ic的t
    pdata = data[data >= 0]
    ndata = data[data < 0]
    data_stat = list(zip(
        data.mean(), data.std(), data.skew(), data.kurt(), t_value, p_value,
        pdata.mean(), ndata.mean(), pdata.std(), ndata.std(),
        ndata.isna().mean(), pdata.isna().mean(), data.mean() / data.std()
    ))
    data_stat = pd.DataFrame(data_stat).T
    data_stat.columns = data.columns
    data_stat.index = ['mean', 'std', 'skew', 'kurt', 't_value', 'p_value',
                       'mean+', 'mean-', 'std+', 'std-', 'pos ratio', 'neg ratio', 'IR']
    #
    return data_stat


def cal_ic_decay(fval_neutralized, ret, maxlag=20, ishow=False, save_path=None) -> pd.DataFrame:
    """计算IC Decay，ret为滞后一期的因子值，ret为当日收益率"""
    from tqdm import tqdm
    ic_decay = {0: np.nan}
    print("Calculating IC Decay...")
    for k in tqdm(range(1, maxlag + 1)):
        ic_decay[k] = cal_ic(fval_neutralized, ret, lag=k, ranked=True, ishow=False).mean().values[0]
    res = pd.DataFrame.from_dict(ic_decay, orient='index', columns=['IC_mean'])

    if save_path is not None:
        res.plot.bar(figsize=(10, 5), title='IC Decay')
        plt.savefig(save_path)
        if ishow:
            plt.show()
        else:
            plt.close()

    return res


def cal_result_stat(df: pd.DataFrame, save_path: str = None, kind='cumsum') -> pd.DataFrame:
    """
    对日度收益序列df计算相关结果
    :param df: 值为日收益率小r，列index为日期DateTime
    :param save_path: 存储名（若有）
    :param kind: 累加/累乘
    :return: 结果面板
    """
    if kind == 'cumsum':
        df1 = df.cumsum() + 1
    elif kind == 'cumprod':
        df1 = df.add(1).cumprod()
    else:
        raise ValueError(f"""Invalid kind={kind}, only support('cumsum', 'cumprod')""")
    data = df.copy()
    data['Date'] = data.index
    data['SemiYear'] = data['Date'].apply(lambda s: f'{s.year}-H{s.month // 7 + 1}')
    res: pd.DataFrame = data.groupby('SemiYear')[['Date']].last().reset_index()
    res.index = res['Date']
    res['Cash'] = (2e7 * df1.loc[res.index]).round(1)
    res['UnitValue'] = df1.loc[res.index]
    res['TotalRet'] = res['UnitValue'] - 1
    res['PeriodRetOnBookSize'] = res['UnitValue'].pct_change()
    res.iloc[0, -1] = res['UnitValue'].iloc[0] - 1
    res['PeriodSharpe'] = df.groupby(data.SemiYear).apply(lambda s: s.mean() / s.std() * np.sqrt(240)).values
    mdd = df1 / df1.cummax() - 1 if kind == 'cumprod' else df1 - df1.cummax()
    res['PeriodMaxDD'] = mdd.groupby(data.SemiYear).min().values
    res['PeriodCalmar'] = res['PeriodRetOnBookSize'] / res['PeriodMaxDD'].abs()
    res['TotalMaxDD'] = mdd.min().values[0]
    res['TotalSharpe'] = (df.mean() / df.std() * np.sqrt(240)).values[0]
    res['AverageCalmar'] = res['TotalSharpe'] / res['TotalMaxDD'].abs()
    res['TotalAnnualRet'] = (df1.iloc[-1] ** (240 / len(df1)) - 1).values[0]

    res['Date'] = res['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    res = res.set_index('SemiYear')
    if save_path is not None:
        res.to_excel(save_path.replace('.csv', '.xlsx'))
    return res


