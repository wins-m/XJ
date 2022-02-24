"""
(created by swmao on Jan. 17th)
(Feb. 11th)
- move functions to supporter.backtester
(Feb. 22nd)
-
"""
import os
import sys
import yaml
import time
from datetime import timedelta

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.backtester import *


# %%
def clip_backtest_conf(conf: dict):
    res = {
        'csv_path': conf['factorscsv_path'],
        'res_path': conf['factorsres_path'],
        'idx_constituent': conf['idx_constituent'],
        'tradeable_path': conf['a_list_tradeable'],
        'ind_citic_path': conf['ind_citic'],
        'marketvalue_path': conf['marketvalue'],
        'close_path': conf['closeAdj'],
        'open_path': conf['openAdj'],
        'test_mode': str(conf['test_mode']),
        'exclude_tradeable': conf['exclude_tradeable'],
        'neu_mtd': conf['neu_mtd'],
        'stk_pool': conf['stk_pool'],
        'stk_w': conf['stk_w'],
        'return_kind': conf['return_kind'],
        'ngroups': conf['ngroups'],
        'holddays': conf['holddays'],
        'cost_rate': float(conf['tc']),
        'begin_date': conf['begin_date'],
        'end_date': conf['end_date'],
        'save_tables': conf['save_tables'],
        'save_plots': conf['save_plots'],
        'ishow': conf['ishow'],
        'all_factornames': pd.read_excel(conf['factors_tested'], index_col=0).loc[1:1].iloc[:, 0].to_list(),
        'save_suffix': conf['save_suffix'] if conf['save_suffix'] != '' else time.strftime("%m%d_%H%M%S", time.localtime()),
        'begin_date_nd60': (pd.to_datetime(conf['begin_date']) - timedelta(60)).strftime('%Y-%m-%d')
    }
    # res['fbegin_end'] = df[['F_NAME', 'F_BEGIN', 'F_END']].set_index('F_NAME').apply(lambda s: (s.iloc[0],
    # s.iloc[1]), axis=1).to_dict()

    return res


# %%
def main():
    # con'f
    # conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf_path = './config2.yaml'
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
    # all_factornames = [k for k, v in conf['fnames'].items() if v == 1]
    # with_updown = 'tradeable' + conf['with_updown']
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
    tradeable_multiplier.dropna(axis=1, how='all', inplace=True)  # 去除全空的列（股票）

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

    # Baseline
    # ret_baseline = (all_ret * idx_weight).sum(axis=1)  # Return
    # ret_baseline.add(1).cumprod().plot(); plt.show()

    # %% Loop All Factors
    fname = all_factornames[0]
    save_folders = dict()
    # %%
    for fname in all_factornames:
        save_folders[fname] = f'{fname}_{suffix}'
        # %% 存储目录管理
        print(f'\nBacktest `{fname}`...')
        ic_mean = 0  # 默认值，更新后用于判断因子方向

        save_path_ = f"""{res_path}{fname}_{suffix}/"""
        os.makedirs(save_path_, exist_ok=True)
        path_format = save_path_ + '{}'  # save_path_save_path_ + fname + '.{}'

        fval = pd.read_csv(f'{csv_path}{fname}.csv', dtype=float, index_col=0, parse_dates=True)
        signal = Signal(data=fval, bd=begin_date, ed=end_date, neu=None, ishow=ishow)
        signal.shift_1d(d_shifted=1)  # 滞后一天，以昨日的因子值参与计算
        signal.keep_tradeable(tradeable_multiplier.loc[begin_date: end_date])
        fbegin_date = signal.get_fbegin()
        fend_date = signal.get_fend()

        if test_mode == '3':  # 由持仓计算持仓组合结果
            holding_weight = signal.get_fv()
            # 改为：以weight为基础的Portfolio对象
            portfolio_statistics_from_weight(holding_weight, cost_rate, all_ret, path_format.format('PanelLong.csv'))

        elif test_mode in '012':  # 进行因子回测，产生策略表现的图表
            conf1 = conf.copy()
            conf1['all_factornames'] = fname
            conf1['fbegin_date'] = fbegin_date.strftime('%Y-%m-%d')
            conf1['fend_date'] = fend_date.ed.strftime('%Y-%m-%d')
            with open(path_format.format('config.yaml'), 'w', encoding='utf-8') as f:
                f.write(yaml.safe_dump(conf1))

            if test_mode == '2':  # 由多空分组计算多/空组合结果
                long_short_group = pd.read_csv(path_format.format('LSGroup.csv'), index_col=0, parse_dates=True)
                ic_mean = pd.read_csv(path_format.format('ICStat.csv'), index_col=0).loc['mean', 'IC']
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

                # %% Long-Short-Group Strategy
                strategy = Strategy(sgn=signal, ng=ngroups, ishow=ishow)
                strategy.cal_long_short_group()
                strategy.cal_group_returns(all_ret, idx_weight)
                if save_tables:
                    strategy.get_ls_group(path_format.format('LSGroup.csv'))
                    strategy.get_group_returns(path_format.format('GroupRtns.csv'))
                if save_plots:
                    strategy.plot_group_returns(path_format.format('ResGroup.png'))
                    strategy.plot_group_returns_total(path_format.format('TotalRtnsGroup.png'))
                long_short_group = strategy.get_ls_group()

            if test_mode == '0':  # 只计算分组，不进行多空回测
                continue

            # Long-Short Strategy
            save_path = path_format.format('positionWeight_{}.csv') if save_tables else None
            w_long, w_short, w_long_short = cal_weight_from_long_short_group(
                long_short_group, ngroups, idx_weight, fbegin_date, fend_date, holddays, ic_mean, save_path)

            # Portfolio Turnover, NStocks, Return, Wealth
            all_panels = portfolio_panels_from_weight(w_long, w_short, w_long_short, idx_weight, cost_rate, all_ret,
                                                      path_format, save_tables, fbegin_date, fend_date)
            # # Turnover: long_short_turnover -> LSTurnover.csv
            if save_plots:
                save_path = path_format.format('LSTurnover.png')
                long_short_turnover = pd.concat([df['Turnover'].rename(k) for k, df in all_panels.items()], axis=1)
                long_short_turnover[['long', 'short']].plot(figsize=(10, 5), grid=True, title='Turnover')
                plt.savefig(save_path)
                if ishow:
                    plt.show()
                else:
                    plt.close()
            # long_short_turnover = cal_turnover_long_short(long_short_group, idx_weight, ngroups, ishow, save_path)
            # long_short_turnover = pd.concat([df['Turnover'].rename(k) for k, df in all_panels.items()], axis=1)

            # Return

            # # Long-Short Absolute Return No Cost: ret_group, ret_baseline -> long_short_return_nc, LSRtnsNC.csv
            # save_path = path_format.format('LSRtnsNC.csv') if save_tables else None
            # long_short_return_nc = panel_long_short_return(ret_group, ret_baseline, save_path)
            long_short_return_nc = pd.concat([df['Return'].rename(k) for k, df in all_panels.items()], axis=1)

            # # Long-Short Absolute Return With Cost
            # long_short_return_wc = long_short_return_nc - long_short_turnover * cost_rate
            # if save_tables:
            #     long_short_return_wc.to_csv(path_format.format('LSRtnsWC.csv'))
            long_short_return_wc = pd.concat([df['Return_wc'].rename(k) for k, df in all_panels.items()], axis=1)

            # Long-Short Absolute Result No Cost: long_short_return_nc -> long_short_absolute_nc, LSAbsResNC.png
            title = 'Long-Short Absolute Result No Cost'
            save_path = path_format.format('LSAbsResNC.png') if save_plots else None
            long_short_absolute_nc = panel_long_short_absolute(long_short_return_nc, ishow, title, save_path)

            # Long-Short Absolute Result With Cost
            title = 'Long-Short Absolute Result With Cost'
            save_path = path_format.format('LSAbsResWC.png') if save_plots else None
            long_short_absolute_wc = panel_long_short_absolute(long_short_return_wc, ishow, title, save_path)

            # Long-Short Excess Result No Cost
            title = 'Long-Short Excess Result No Cost'
            save_path = path_format.format('LSExcResNC.png') if save_plots else None
            panel_long_short_excess(long_short_absolute_nc, ishow, title, save_path)
            # long_short_excess_nc = long_short_return_nc.iloc[:, :3] - long_short_return_nc.iloc[:,
            # 3].values.reshape(-1, 1)

            # Long-Short Excess Result With Cost
            title = 'Long-Short Excess Result With Cost'
            save_path = path_format.format('LSExcResWC') if save_tables else None
            panel_long_short_excess(long_short_absolute_wc, ishow, title, save_path)
            # long_short_excess_wc = long_short_return_wc.iloc[:, :3] - long_short_return_wc.iloc[:,
            # 3].values.reshape(-1, 1)

            # max drawdown
            # Long-Absolute MaxDrawdown No Cost
            title = 'Long-Absolute MaxDrawdown No Cost'
            save_path = path_format.format('LMddNC.png') if save_plots else None
            cal_sr_max_drawdown(long_short_absolute_nc['long'], ishow, title, save_path)

            if ngroups != 1:
                # Long-Short-Absolute MaxDrawdown No Cost
                title = 'Long-Short-Absolute MaxDrawdown No Cost'
                save_path = path_format.format('LSMddNC.png') if save_plots else None
                cal_sr_max_drawdown(long_short_absolute_nc['long_short'], ishow, title, save_path)

            # Long-Absolute MaxDrawdown With Cost
            title = 'Long-Absolute MaxDrawdown With Cost'
            save_path = path_format.format('LMddWC.png') if save_plots else None
            cal_sr_max_drawdown(long_short_absolute_wc['long'], ishow, title, save_path)

            if ngroups != 1:
                # Long-Short-Absolute MaxDrawdown With Cost
                title = 'Long-Short-Absolute MaxDrawdown With Cost'
                save_path = path_format.format('LSMddWC.png') if save_plots else None
                cal_sr_max_drawdown(long_short_absolute_wc['long_short'], ishow, title, save_path)

            # Portfolio Statistics
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

            # Annual Result

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
            print(f'Graphs & Tables Saved in {path_format}')
        else:
            raise ValueError(f'Invalid test_mode {test_mode}')
    # %%
    print(save_folders)


# %%
if __name__ == '__main__':
    main()
