"""
(created by swmao on Jan. 17th)
(Feb. 11th)
- move functions to supporter.backtester
(Feb. 22nd)
- use object method rather than function
"""
import os
import sys
import yaml

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from factor_backtest.backtester import *


# %%
def main():
    # %%
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    # conf_path = './config2.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    # conf = clip_backtest_conf(conf)
    # %%
    single_test(conf)


def single_test(conf: dict):
    """单因子回测, conf is a sub one"""
    conf = clip_backtest_conf(conf)
    # %% config for backtest
    csv_path = conf['csv_path']
    res_path = conf['res_path']
    idx_constituent = conf['idx_constituent']
    tradeable_path = conf['tradeable_path']
    ind_citic_path = conf['ind_citic_path']
    marketvalue_path = conf['marketvalue_path']
    close_path = conf['close_path']
    open_path = conf['open_path']

    test_mode = conf['test_mode']
    exclude_tradeable = conf['exclude_tradeable']
    neu_mtd = conf['neu_mtd']
    stk_pool = conf['stk_pool']
    stk_w = conf['stk_w']
    return_kind = conf['return_kind']
    ngroups = conf['ngroups']
    holddays = conf['holddays']
    cost_rate = conf['cost_rate']

    begin_date = pd.to_datetime(conf['begin_date'])
    end_date = pd.to_datetime(conf['end_date'])
    begin_date_nd60 = pd.to_datetime(conf['begin_date_nd60'])

    save_tables, save_plots, ishow = conf['save_tables'], conf['save_plots'], conf['ishow']
    # ishow = True
    save_suffix = conf['save_suffix']
    all_factornames = conf['all_factornames']

    suffix = f"""{neu_mtd}_{stk_pool}_{stk_w}_{ngroups}g_{return_kind}_{holddays}hd({save_suffix})"""
    print("CONFIG LOADED", suffix)

    # %% tradeable_multiplier 股池筛选乘子
    def read_tradeable(hdf_k='ipo'):
        """可改写装饰器，获取可交易乘子，不可为False，可为True"""
        df = read_single_factor(tradeable_path, begin_date_nd60, end_date, bool, hdf_k=hdf_k)
        # df = df.replace(False, np.nan).astype(float)
        if return_kind == 'ctc':
            df = df.shift(1).iloc[1:]  # 因子是昨日的，昨日收盘价买入今日收盘价卖出的return，可交易是昨日可否买入
        return df

    # tradeable_multiplier = read_single_factor(tradeable_path, begin_date_nd60, end_date, bool, hdf_k=with_updown)
    tradeable_multiplier = read_tradeable(hdf_k='ipo')  # 上市其内所有为基准股池
    for tradeable_key, status in exclude_tradeable.items():  # 叠加停用股池
        if status == 0:
            print(f'Sift tradeable stocks via `{tradeable_key}`')
            tradeable_multiplier &= read_tradeable(hdf_k=tradeable_key)
    tradeable_multiplier.replace(False, np.nan, inplace=True)  # 股池乘子，不可交易为nan
    tradeable_multiplier.dropna(axis=1, how='all', inplace=True)  # 去除全空列（股票）

    # %% idx_weight: tradeable @ stk_pool @ stk_w
    if stk_pool == 'NA' or stk_pool is None or stk_pool == '':  # 未指定股池
        idx_weight = tradeable_multiplier.astype(float)  # 全市场可交易股`tradeable_multiplier`等权重
    else:  # 声明了特殊的股池
        # stk_pool = 'CSI500'
        idx_weight = get_idx_weight(idx_constituent.format(stk_pool), begin_date_nd60, end_date, stk_w)  # 股池内权重
        idx_weight = idx_weight * tradeable_multiplier.reindex_like(idx_weight)
    idx_weight = idx_weight.fillna(0).apply(lambda s: s / s.abs().sum(), axis=1).astype(float)  # 权重之和为1
    # idx_weight = idx_weight.loc[begin_date: end_date]  # 限制回测期时间范围
    tradeable_multiplier = tradeable_multiplier.reindex_like(idx_weight)  # 缩小 tradeable_multiplier 到 idx_weight
    assert round(idx_weight.abs().sum(axis=1).prod(), 4) == 1  # 大盘/指数全股仓位权重绝对值之和为1

    # %% Stock Returns: 可行日度收益
    if return_kind == 'ctc':  # 今日收益率：昨日信号，昨日收盘买入，今日收盘卖出
        sell_price = pd.read_csv(close_path, index_col=0, parse_dates=True)
    elif return_kind == 'oto':  # 今日收益率：昨日信号，今日开盘买入，明日开盘卖出
        sell_price = pd.read_csv(open_path, index_col=0, parse_dates=True).shift(-1)
    else:
        raise ValueError(f"""Invalid config.return_kind {return_kind}""")
    all_ret: pd.DataFrame = sell_price.pct_change().reindex_like(idx_weight) * tradeable_multiplier

    # %% Loop All Factors
    fname = all_factornames[0]
    save_folders = dict()
    # %%
    for fname in all_factornames:
        # %%
        save_folders[fname] = f'{fname}_{suffix}'

        # 存储目录管理
        print(f'\nBacktest `{fname}`...')
        ic_mean = 0  # 默认值，更新后用于判断因子方向

        save_path_ = f"""{res_path}{fname}_{suffix}/"""
        os.makedirs(save_path_, exist_ok=True)
        path_format = save_path_ + '{}'  # save_path_save_path_ + fname + '.{}'

        fval = pd.read_csv(f'{csv_path}{fname}.csv', dtype=float, index_col=0, parse_dates=True)
        signal = Signal(data=fval, bd=begin_date, ed=end_date, neu=None)
        signal.shift_1d(d_shifted=1)  # 滞后一天，以昨日的因子值参与计算
        signal.keep_tradeable(tradeable_multiplier.loc[begin_date: end_date])
        fbegin_date = signal.get_fbegin()
        fend_date = signal.get_fend()

        # %%
        if test_mode == '3':  # 由持仓计算持仓组合结果
            portfolio_l = Portfolio(w=signal.get_fv())
            portfolio_l.cal_panel_result(cr=cost_rate, ret=all_ret)
            portfolio_l.get_panel(path_format.format('PanelLong.csv'))
            portfolio_l.plot_cumulative_returns(ishow=ishow, path=path_format.format('LAbsRes.png'), kind='cumsum')
            portfolio_l.get_half_year_stat(wc=False, path=path_format.format('ResLongNC.csv'))
            portfolio_l.get_half_year_stat(wc=True, path=path_format.format('ResLongWC.csv'))
            portfolio_l.plot_max_drawdown(ishow=ishow, path=path_format.format('LMddNC.png'), wc=False, kind='cumsum')
            portfolio_l.plot_max_drawdown(ishow=ishow, path=path_format.format('LMddWC.png'), wc=True, kind='cumsum')
            print(f'Graphs & Tables Saved in {path_format}')
            continue

        # 存回测相关config
        conf1 = conf.copy()
        conf1['all_factornames'] = fname
        conf1['fbegin_date'] = fbegin_date.strftime('%Y-%m-%d')
        conf1['fend_date'] = fend_date.strftime('%Y-%m-%d')
        with open(path_format.format('config.yaml'), 'w', encoding='utf-8') as f:
            f.write(yaml.safe_dump(conf1))

        if test_mode == '2':  # 由多空分组计算多/空组合结果
            long_short_group = pd.read_csv(path_format.format('LSGroup.csv'), index_col=0, parse_dates=True)
            ic_mean = pd.read_csv(path_format.format('ICStat.csv'), index_col=0).loc['mean', 'IC']
            strategy = Strategy(sgn=signal, ng=ngroups)
            strategy.ls_group = long_short_group
        else:  # 因子处理 & 因子统计（IC） & 多空分组（分组收益）
            if ngroups != 1:  # 非事件信号（0-1）
                signal.neutralize_by(neu_mtd, ind_citic_path, marketvalue_path)
                signal.cal_ic(all_ret)
                signal.cal_ic_statistics()
                signal.cal_ic_decay(all_ret=all_ret, lag=20)

                ic_mean = signal.get_ic_mean(ranked=True)
                print(signal.get_ic_stat(path_format.format('ICStat.csv') if save_tables else None))
                print(signal.get_ic_decay(path_format.format('ICDecay.csv') if save_tables else None).iloc[1:6])

                if save_plots:
                    signal.plot_ic(ishow, path_format)
                    signal.plot_ic_decay(ishow, path_format.format('ICDecay.png'))

            # Long-Short-Group Strategy
            strategy = Strategy(sgn=signal, ng=ngroups)
            strategy.cal_long_short_group()
            strategy.cal_group_returns(all_ret, idx_weight)
            if save_tables:
                strategy.get_ls_group(path_format.format('LSGroup.csv'))
                strategy.get_group_returns(path_format.format('GroupRtns.csv'))
            if save_plots:
                strategy.plot_group_returns(ishow, path_format.format('ResGroup.png'))
                strategy.plot_group_returns_total(ishow, path_format.format('TotalRtnsGroup.png'))
            # long_short_group = strategy.get_ls_group()

        if test_mode == '0':  # 只计算分组，不进行多空回测
            print(f'Graphs & Tables Saved in {path_format}')
            continue

        rvs = (ic_mean < 0)
        strategy.cal_long_short_panels(idx_weight, holddays, rvs, cost_rate, all_ret)
        # all_panels = strategy.get_ls_panels(path_format if save_tables else None)

        if save_plots:
            strategy.plot_long_short_turnover(ishow, path_format.format('LSTurnover.png'))
            strategy.plot_cumulative_returns(ishow, path_format.format('LSAbsResNC.png'), False, 'cumsum', False)
            strategy.plot_cumulative_returns(ishow, path_format.format('LSAbsResWC.png'), True, 'cumsum', False)
            strategy.plot_cumulative_returns(ishow, path_format.format('LSExcResNC.png'), False, 'cumsum', True)
            strategy.plot_cumulative_returns(ishow, path_format.format('LSExcResWC.png'), True, 'cumsum', True)
            strategy.portfolio['long'].plot_max_drawdown(ishow, path_format.format('LMddNC.png'), wc=False)
            strategy.portfolio['long'].plot_max_drawdown(ishow, path_format.format('LMddWC.png'), wc=True)
            if ngroups != 1:
                strategy.portfolio['long_short'].plot_max_drawdown(ishow, path_format.format('LSMddNC.png'), wc=False)
                strategy.portfolio['long_short'].plot_max_drawdown(ishow, path_format.format('LSMddWC.png'), wc=True)

        if save_tables:
            for wc in [False, True]:
                strategy.get_portfolio_statistics(kind='long', wc=wc, path_f=path_format)
                if ngroups != 1:
                    strategy.get_portfolio_statistics(kind='short', wc=wc, path_f=path_format)
                    strategy.get_portfolio_statistics(kind='long_short', wc=wc, path_f=path_format)

        print(f'Graphs & Tables Saved in {path_format}')
    # %%
    print(save_folders)


# %%
if __name__ == '__main__':
    main()
