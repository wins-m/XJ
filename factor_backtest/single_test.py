"""
(created by swmao on Jan. 17th)

"""
import os, sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.factor_operator import *
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from datetime import timedelta


def get_idx_weight(pool_path, begin_date, end_date, mtd='idx'):
    """获取城成分股权重 等权(ew)，按指数成分股权重(idx)，返回值行绝对值之和为1"""
    df = read_single_factor(pool_path, begin_date, end_date, float)
    df1 = df
    if mtd == 'idx':
        df1 = df.apply(lambda s: s / np.nansum(s), axis=1)
    elif mtd == 'ew':
        df1 = (df.T * 0 + 1 / df.count(axis=1)).T
    assert round(df1.abs().sum(axis=1).prod(), 2) == 1
    return df1


def get_long_short_group(df: pd.DataFrame, ngroups: int) -> pd.DataFrame:
    """
    因子值替换为分组标签
    :param df: 因子值（经过标准化、中性化）
    :param ngroups: 分组数(+)；多头/空头内资产数量取负值(-)，若股池不够大，重叠部分不持有
    :return: 因子分组，nan仍为nan，其余为分组编号 0~(分组数-1)
    """
    res = None
    if ngroups < 0:
        cnt = df.count(axis=1)
        nm = (cnt + 2 * ngroups).abs()
        lg = (cnt - nm) / 2  # 上阈值
        hg = lg + nm  # 下阈值
        rnk = df.rank(axis=1)  #, method='first')
        res = rnk * 0 + 1  # 中间组
        res[rnk.apply(lambda s: s <= lg, axis=0)] = 0  # 空头
        res[rnk.apply(lambda s: s >= hg, axis=0)] = 2  # 多头
        # res = rnk * 0  # 一般
        # res[rnk.apply(lambda s: s >= hg, axis=0)] = 1  # 多头
    elif ngroups == 1:
        return df
    else:
        res = df.rank(axis=1, pct=True).applymap(lambda x: x // (1 / ngroups))

    return res


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


def panel_long_short_excess(long_short_absolute_nc, ishow=False, title='Long-Short Excess Result', save_path=None) -> pd.DataFrame:
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


def cal_long_short_group_rtns(long_short_group, ret, idx_weight, ngroups, save_path: None) -> pd.DataFrame:
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
        w1 = idx_weight[mask].apply(lambda s: s/np.nansum(s), axis=1)
        ret_group['$Group_{'+str(gi)+'}$'] = (ret1 * w1).sum(axis=1)

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


def cal_sr_max_drawdown(df: pd.Series, ishow=False, title=None, save_path=None) -> pd.DataFrame:
    """计算序列回撤"""
    cm = df.cummax()
    mdd = pd.DataFrame(index=df.index)
    mdd[f'{df.name}_maxdd'] = df / cm - 1

    if save_path is not None:
        mdd.plot(kind='area', figsize=(10, 5), grid=True, color='y', alpha=.5, title=title)
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
    asharp = data.groupby(year_group).apply(lambda s: s.mean() / s.std() * np.sqrt(y_len/s.count()))

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

    w_long = rescale_weight(group_weight(long_short_group, idx_weight, max(ngroups-1, 1)))  # 最大编号组，做多
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


def cal_ic(fv_l1, ret, lag=1, rankIC=False, ishow=False, save_path=None) -> pd.DataFrame:
    """计算IC：昨日因子值与当日收益Pearson相关系数"""
    mtd = 'spearman' if rankIC else 'pearson'
    ic = fv_l1.shift(lag-1).corrwith(ret, axis=1, drop=False, method=mtd)
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
        ic_decay[k] = cal_ic(fval_neutralized, ret, lag=k, rankIC=True, ishow=False).mean().values[0]
    res = pd.DataFrame.from_dict(ic_decay, orient='index', columns=['IC_mean'])

    if save_path is not None:
        res.plot.bar(figsize=(10, 5), title='IC Decay')
        plt.savefig(save_path)
        if ishow:
            plt.show()
        else:
            plt.close()

    return res


def cal_result_stat(df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    """
    对日度收益序列df计算相关结果
    :param df: 值为日收益率小r，列index为日期DateTime
    :param save_path: 存储名（若有）
    :return: 结果面板
    """
    df1 = df.add(1).cumprod()
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
    res['PeriodSharpe'] = df.groupby(data.SemiYear).apply(lambda s: s.mean()/s.std()*np.sqrt(240)).values
    mdd = df1 / df1.cummax() - 1
    res['PeriodMaxDD'] = mdd.groupby(data.SemiYear).min().values
    res['PeriodCalmar'] = res['PeriodRetOnBookSize'] / res['PeriodMaxDD'].abs()
    res['TotalMaxDD'] = mdd.min().values[0]
    res['TotalSharpe'] = (df.mean() / df.std() * np.sqrt(240)).values[0]
    res['AverageCalmar'] = res['TotalSharpe'] / res['TotalMaxDD'].abs()
    res['TotalAnnualRet'] = (df1.iloc[-1]**(240/len(df1)) - 1).values[0]
    if save_path is not None:
        res.to_excel(save_path.replace('.csv', '.xlsx'))
    return res


# %%
def single_test(conf: dict):
    """单因子回测"""
    neu_mtd = conf['neu_mtd']
    stk_pool = conf['stk_pool']
    stk_w = conf['stk_w']
    csv_path = conf['factorscsv_path']
    res_path = conf['factorsres_path']
    idx_constituent = conf['idx_constituent']
    tradeable_path = conf['a_list_tradeable']
    ind_citic_path = conf['ind_citic']
    marketvalue_path = conf['marketvalue']
    ngroups = conf['ngroups']
    close_path = conf['closeAdj']
    save_tables = conf['save_tables']
    save_plots = conf['save_plots']
    ishow = conf['ishow']
    begin_date = pd.to_datetime(conf['begin_date'])
    end_date = pd.to_datetime(conf['end_date'])
    cost_rate = float(conf['tc'])
    all_factornames = [k for k, v in conf['fnames'].items() if v == 1]
    with_updown = 'tradeable' + conf['with_updown']
    save_suffix = conf['save_suffix']
    print("CONFIG LOADED", neu_mtd, stk_pool)
    # ishow = True

    # Tradeable Sifter (a_list_tradeable)
    a_list_tradeable = read_single_factor(tradeable_path, begin_date - timedelta(60), end_date, bool, hdf_k=with_updown)
    tradeable_multiplier = a_list_tradeable.replace(False, np.nan).dropna(axis=1, how='all')

    # idx_weight: a_list_tradeable @ stk_pool @ stk_w
    if stk_pool is None or stk_pool == 'NA':  # 未指定股池，则由全市场的`a_list_tradeable`生成mask
        idx_weight = tradeable_multiplier.apply(lambda s: s/s.sum(), axis=1)
        pool_multiplier = tradeable_multiplier * 1
    else:
        idx_weight = get_idx_weight(idx_constituent.format(stk_pool), begin_date - timedelta(60), end_date, stk_w)  # 股池内权重
        pool_multiplier = idx_weight * 0 + 1
    idx_weight = idx_weight.loc[begin_date:end_date]  # sum(axis=1)均为1

    # Stock Returns
    close = read_single_factor(close_path, begin_date - timedelta(5), end_date, float, pool_multiplier)  # 日收盘价
    all_ret: pd.DataFrame = close.pct_change().loc[begin_date:end_date]  # 收盘价日收益率，停牌第二天存在误差

    # Baseline
    ret_baseline = (all_ret * idx_weight).sum(axis=1)  # Return

    # Loop All Factors
    fname = 'first_report'
    for fname in all_factornames:

        # Result File Format: *args -> path_format
        save_path_ = f"""{res_path}{fname}_{neu_mtd}_{stk_pool}{stk_w}_{ngroups}g""" + save_suffix
        os.makedirs(save_path_, exist_ok=True)
        path_format = save_path_ + "/{}"

        # Factor Value: *args -> fval
        fval = read_single_factor(f'{csv_path}{fname}.csv', begin_date - timedelta(5), end_date, float, pool_multiplier)
        fval: pd.DataFrame = fval.shift(1).loc[begin_date:]  # 滞后一天，以昨日的因子值参与计算
        fval.head().sum(axis=1)  # check first date
        fval = fval * tradeable_multiplier.loc[begin_date:]
        fval = fval.astype(float)
        ret = all_ret.reindex_like(fval)  # returns aligned with factor value

        if ngroups == 1:
            fval_neutralized = fval.copy()
            long_short_group = fval.copy()
        else:
            # Factor Neutralization: fval, neu_mtd, ind_citic_path, marketvalue_path -> fval_neutralized, ret
            fval_neutralized = factor_neutralization(fval, neu_mtd, ind_citic_path, marketvalue_path)
            # Group Label Panel: fval_neutralized, ngroups -> long_short_group
            long_short_group = get_long_short_group(fval_neutralized, ngroups)

        # Group Absolute Return: long_short_group, ret, idx_weight, ngroups -> ret_group, GroupRtns.csv
        save_path = path_format.format('GroupRtns.csv') if save_tables else None
        ret_group = cal_long_short_group_rtns(long_short_group, ret, idx_weight, ngroups, save_path)

        # Turnover: long_short_turnover -> LSTurnover.csv
        save_path = path_format.format('LSTurnover.csv') if save_tables else None
        long_short_turnover = cal_turnover_long_short(long_short_group, idx_weight, ngroups, ishow, save_path)

        # IC (distribution): fval_neutralized, ret -> IC, IC.png
        save_path = path_format.format('IC.png') if save_plots else None
        ic = cal_ic(fval_neutralized, ret, rankIC=False, ishow=ishow, save_path=save_path)
        save_path = path_format.format('ICRank.png') if save_plots else None
        rank_ic = cal_ic(fval_neutralized, ret, rankIC=True, ishow=ishow, save_path=save_path)

        if (ngroups != 1) and (ic.isna().prod().values[0] != 1) and (rank_ic.isna().prod().values[0] != 1):
            # IC Stats: ic -> ICStat.csv
            ic_stat = pd.DataFrame()
            ic_stat['IC'] = cal_ic_stat(data=ic)
            ic_stat['Rank IC'] = cal_ic_stat(data=rank_ic)
            ic_stat = ic_stat.astype('float16')
            if save_tables:
                ic_stat.to_csv(path_format.format('ICStat.csv'))
            else:
                print(ic_stat)

            # IC Decay: fval_neutralized, ret -> ICDecay.png
            save_path = path_format.format('ICDecay.png') if save_plots else None
            cal_ic_decay(fval_neutralized, ret, 20, ishow, save_path)

            # Cumulated IC_IR

        """
        save_tables, save_plots, ishow, cost_rate, save_tables
        path_format, ret_group, ret_baseline, long_short_turnover
        
        """
        # Group Test Result: ret_group -> ResGroup.png
        save_path = path_format.format('ResGroup.png') if save_plots else None
        plot_rtns_group(ret_group, ishow, save_path)

        # Total Return of Group: ret_group -> TotalRtnsGroup.png
        save_path = path_format.format('TotalRtnsGroup.png') if save_plots else None
        cal_total_ret_group(ret_group, ishow, save_path)

        # Long-Short Absolute Return No Cost: ret_group, ret_baseline -> long_short_return_nc, LSRtnsNC.csv
        save_path = path_format.format('LSRtnsNC.csv') if save_tables else None
        long_short_return_nc = panel_long_short_return(ret_group, ret_baseline, save_path)

        # Long Only Statistics No Cost: long_short_return_nc -> ResLongNC.csv
        save_path = path_format.format('ResLongNC.csv') if save_tables else None
        cal_result_stat(long_short_return_nc[['long']], save_path)

        if ngroups != 1:
            # Short Only Statistics No Cost: long_short_return_nc -> ResShortNC.csv
            save_path = path_format.format('ResShortNC.csv') if save_tables else None
            cal_result_stat(long_short_return_nc[['short']], save_path)

            # Long-Short Statistics No Cost: long_short_return_nc -> ResLongShortNC.csv
            save_path = path_format.format('ResLongShortNC.csv') if save_tables else None
            cal_result_stat(long_short_return_nc[['long_short']], save_path)

        # Long-Short Absolute Result No Cost: long_short_return_nc -> long_short_absolute_nc, LSAbsResNC.png
        title = 'Long-Short Absolute Result No Cost'
        save_path = path_format.format('LSAbsResNC.png') if save_plots else None
        long_short_absolute_nc = panel_long_short_absolute(long_short_return_nc, ishow, title, save_path)

        # Long-Short Excess Result No Cost
        title = 'Long-Short Excess Result No Cost'
        save_path = path_format.format('LSExcResNC.png') if save_plots else None
        long_short_excess_nc = panel_long_short_excess(long_short_absolute_nc, ishow, title, save_path)

        # Long-Absolute MaxDrawdown No Cost
        title = 'Long-Absolute MaxDrawdown No Cost'
        save_path = path_format.format('LMddNC.png') if save_plots else None
        cal_sr_max_drawdown(long_short_absolute_nc['long'], ishow, title, save_path)

        if ngroups != 1:
            # Long-Short-Absolute MaxDrawdown No Cost
            title = 'Long-Short-Absolute MaxDrawdown No Cost'
            save_path = path_format.format('LSMddNC.png') if save_plots else None
            cal_sr_max_drawdown(long_short_absolute_nc['long_short'], ishow, title, save_path)

        # Long-Short Absolute Return With Cost
        long_short_return_wc = long_short_return_nc - long_short_turnover * cost_rate
        if save_tables:
            long_short_return_wc.to_csv(path_format.format('LSRtnsWC.csv'))
            #

        # Long Only Statistics With Cost
        save_path = path_format.format('ResLongWC.csv') if save_tables else None
        cal_result_stat(long_short_return_wc[['long']], save_path)

        if ngroups != 1:
            # Short Only Statistics With Cost
            save_path = path_format.format('ResShortWC.csv') if save_tables else None
            cal_result_stat(long_short_return_wc[['short']], save_path)

            # Long-Short Statistics With Cost
            save_path = path_format.format('ResLongShortWC.csv') if save_tables else None
            cal_result_stat(long_short_return_wc[['long_short']], save_path)

        # Long-Short Absolute Result With Cost
        title = 'Long-Short Absolute Result With Cost'
        save_path = path_format.format('LSAbsResWC.png') if save_plots else None
        long_short_absolute_wc = panel_long_short_absolute(long_short_return_wc, ishow, title, save_path)

        # Long-Short Excess Result With Cost
        title = 'Long-Short Excess Result With Cost'
        save_path = path_format.format('LSExcResWC') if save_tables else None
        long_short_excess_wc = panel_long_short_excess(long_short_absolute_wc, ishow, title, save_path)

        # Long-Absolute MaxDrawdown With Cost
        title = 'Long-Absolute MaxDrawdown With Cost'
        save_path = path_format.format('LMddWC.png') if save_plots else None
        cal_sr_max_drawdown(long_short_absolute_wc['long'], ishow, title, save_path)

        if ngroups != 1:
            # Long-Short-Absolute MaxDrawdown With Cost
            title = 'Long-Short-Absolute MaxDrawdown With Cost'
            save_path = path_format.format('LSMddWC.png') if save_plots else None
            cal_sr_max_drawdown(long_short_absolute_wc['long_short'], ishow, title, save_path)

        # Annual Return No Cost
        title = 'Annual Return No Cost'
        save_path = path_format.format('LSYRtnsNC.png') if save_plots else None
        cal_yearly_return(long_short_return_nc, ishow, title, save_path)

        # Annual Return With Cost
        title = 'Annual Return With Cost'
        save_path = path_format.format('LSYRtnsWC.png') if save_plots else None
        cal_yearly_return(long_short_return_wc, ishow, title, save_path)

        # Annual Yearly Sharpe No Cost
        title = 'Annual Yearly Sharpe No Cost'
        save_path = path_format.format('LSYSharpNC.png') if save_plots else None
        cal_yearly_sharpe(long_short_return_nc, ishow, title, save_path)

        # Annual Yearly Sharpe With Cost
        title = 'Annual Yearly Sharpe With Cost'
        save_path = path_format.format('LSYSharpWC.png') if save_plots else None
        cal_yearly_sharpe(long_short_return_wc, ishow, title, save_path)

        #
    #


if __name__ == '__main__':
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    single_test(conf)
