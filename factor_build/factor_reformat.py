"""
(created by swmao on Jan. 21st)
各类因子的预处理，做成可进入回测的2D面板
"""

import pandas as pd
import numpy as np
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.factor_operator import read_single_factor


def adjust_factor_pe_residual(conf: dict):
    """
    调整pe_residual计算结果为因子形式
    - 去除industry列，相同个股前后拼接
    - 去重、去空
    - 取反：PE被高估的股票发生回归，应该取反

    """
    from supporter.factor_operator import read_single_factor

    tradedates = pd.read_csv(conf['tdays_d'], index_col=0)
    tradedates = tradedates.loc['2013-01-01':'2021-12-31']  # conf['begin_date'], conf['end_date']
    tradedates = tradedates.index.to_list()

    filename = 'pe_residual_20130101_20211231.csv'
    df = pd.DataFrame()
    for filename in conf['tables']['pe_tables'][1:]:
        df = read_single_factor(path = conf['factorscsv_path'] + filename)
        df = df.set_index(df.iloc[:, 0].rename('stockcode')).iloc[:, 1:].T
        duplicated_stockcode = df.columns[df.columns.duplicated()].to_list()
        #
        if len(duplicated_stockcode) > 0:
            stk = duplicated_stockcode[0]
            for stk in duplicated_stockcode:
                tmp = df[stk]
                res = pd.DataFrame()
                for ri in range(tmp.shape[1]):
                    res = pd.concat([res, tmp.iloc[:, ri:ri + 1].dropna(axis=0)], axis=0)
                df.loc[:, stk] = res
        #
        df = df.drop_duplicates()
        df = -df  # PE被高估的股票发生回归，应该取反
        df.loc[tradedates].astype(float).to_csv(
            conf['factorscsv_path'] + '_'.join(filename.split('_')[:-2]) + '.csv')
        # df.to_csv(conf['factorscsv_path'] + '_'.join(filename.split('_')[:-2]) + '.csv')


def adjust_event_first_report(conf: dict, dur=5, kind=None):
    path = conf['event_first_report']
    csv_path = conf['factorscsv_path']
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # df['000650.SZ'].dropna()

    # 预处理，T0若新上市/停牌/涨跌停，删除信号
    if kind is not None:
        tradeable = read_single_factor(conf['a_list_tradeable'], conf['begin_date'], conf['end_date'], hdf_k=kind)
        df = df.reindex_like(tradeable) * tradeable.replace(False, np.nan)

    if dur > 1:
        df = df.fillna(method='ffill', limit=dur-1)
    elif dur == 1:
        pass
    else:
        raise ValueError(f'Invalid dur: {dur}')
    # df['000650.SZ'].dropna()
    df = df.fillna(0)
    # df = df * template.replace(False, np.nan)  # 不在此处筛选Tradeable

    # tmp = df.loc['2018-07-06']
    # tmp = tmp[tmp == 1]
    # # tmp2 = pd.read_clipboard(sep=',', index_col=0)
    # (len(tmp), len(tmp2))
    # set(tmp.index) - set(tmp2['stockcode'])
    # set(tmp2['stockcode']) - set(tmp.index)
    # tmp.sum()
    # df.loc['2018-07-09'].sum()

    df.to_csv(csv_path + f"""first_report_dur{dur}{'_' + kind if kind else ''}.csv""")
    print(f"""first_report_dur{dur}{'_' + kind if kind else ''}.csv""", 'saved.')


def event_condition_panel(conf, dur=3, threshold=100, prior='CAR_3'):
    event_panel = pd.DataFrame(pd.read_hdf(conf['event_first_report2'], key='event_first_report', index_col=0))
    # condition 1: 机构关注少于5
    event_panel1 = event_panel[event_panel.instnum < 5]
    # condition 2: 事件当天未发生涨停
    event_panel2 = event_panel1[event_panel1.maxUp != 1]
    def _investigate():
        # 看事前CAR和事后CAR3的相关性
        for _other in ['CAR_120', 'CAR_60', 'CAR_40', 'CAR_20', 'CAR_10', 'CAR_5', 'CAR_4', 'CAR_3', 'CAR_2', 'AR_1']:
            print(_other + '\t', event_panel2['CAR3'].corr(event_panel2[_other], method='pearson'),
                  '\t', event_panel2['CAR3'].corr(event_panel2[_other], method='spearman'))
        # 单日截面内排名
        from matplotlib import pyplot as plt
        _RANK_CAR_3_CROSS = event_panel2.groupby('tradingdate')['CAR_3'].rank(pct=True)
        _G_RANK_CAR_3_CROSS = _RANK_CAR_3_CROSS // .1
        event_panel2['CAR3'].groupby(_G_RANK_CAR_3_CROSS).mean().plot.bar()
        plt.show()
        #
        _RANK_CAR_3_TS = event_panel2['CAR_3'].rank(pct=True)
        _G_RANK_CAR_3_TS = _RANK_CAR_3_TS // .1
        event_panel2['CAR3'].groupby(_G_RANK_CAR_3_TS).mean().plot.bar()
        plt.show()
        #
        import statsmodels.formula.api as sm
        fm = 'CAR3~' + '+'.join(['CAR_120', 'CAR_60', 'CAR_20', 'CAR_3'])
        ols_res = sm.ols(formula=fm, data=event_panel2).fit()
        ols_res.summary()
        event_panel['X-4'] = event_panel.apply(lambda s: 3 * s['CAR_120'] - 4 * s['CAR_60'] + 5 * s['CAR_20'] - 11 * s['CAR_3'], axis=1)
        event_panel.to_hdf(conf['event_first_report2'], key='event_first_report')
    def _investigate1():
        panel = event_panel2.copy()
        # panel['RANK_CAR_3'] = panel.groupby('tradingdate')['CAR_3'].rank(pct=True)
        # signal = panel.pivot(index='tradingdate', columns='stockcode', values='RANK_CAR_3')
        # signal_dur = signal.fillna(method='ffill', limit=dur-1)
        # filename = 'first_report_CAR_3_rank_dur3'
        # signal_dur.to_csv(conf['factorscsv_path'] + filename + '.csv')

        panel['RANK_CAR_10'] = panel.groupby('tradingdate')['CAR_10'].rank(pct=True)
        signal = panel.pivot(index='tradingdate', columns='stockcode', values='RANK_CAR_10')
        signal_dur = signal.fillna(method='ffill', limit=dur-1)
        filename = 'first_report_CAR_10_rank_dur3'
        signal_dur.to_csv(conf['factorscsv_path'] + filename + '.csv')
    def _investigate2():
        panel = event_panel2.copy()

        signal = panel.pivot(index='tradingdate', columns='stockcode', values='CAR_3')
        signal_dur = signal.fillna(method='ffill', limit=dur-1)
        signal_dur = signal_dur.rank(axis=1, pct=True)
        filename = 'first_report_CAR_3_rank3_dur3'
        signal_dur.to_csv(conf['factorscsv_path'] + filename + '.csv')

        signal = panel.pivot(index='tradingdate', columns='stockcode', values='CAR_10')
        signal_dur = signal.fillna(method='ffill', limit=dur - 1)
        signal_dur = signal_dur.rank(axis=1, pct=True)
        filename = 'first_report_CAR_10_rank3_dur3'
        signal_dur.to_csv(conf['factorscsv_path'] + filename + '.csv')
    # condition 3: 日内截面上，X日累计超额收益 排在后50%
    _RANK_CAR_X = event_panel2.groupby('tradingdate')[prior].rank(pct=True)
    # event_panel3 = event_panel2[_RANK_CAR_X <= threshold/100]  # CAR_X更小的threshold%（反转）
    # event_panel3 = event_panel2[_RANK_CAR_X > (threshold-20)/100]  # CAR_X更大的threshold%（反转）
    event_panel3 = event_panel2[(_RANK_CAR_X <= threshold/100) & (_RANK_CAR_X > (threshold-20)/100)]  # CAR_X更大的threshold%（反转）
    # signal
    instnum_1d = event_panel3.pivot(index='tradingdate', columns='stockcode', values='instnum')
    signal_1d = instnum_1d / instnum_1d
    signal_dur = signal_1d.fillna(method='ffill', limit=dur-1).fillna(0)
    # filename = f'first_report_i5_{threshold}{prior}_0up_dur3'
    # filename = f'first_report_i5_R{threshold}{prior}_0up_dur3'
    filename = f'first_report_i5_G{threshold}{prior}_0up_dur3'
    signal_dur.to_csv(conf['factorscsv_path'] + filename + '.csv')
    print(filename)


# %%
if __name__ == '__main__':
    # %%
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    # %%
    # adjust_factor_pe_residual(conf)
    # %%
    # kind = 'updown'
    # dur = 3
    # # for d in range(1, 6):
    # for kind in [None, 'updown', 'updown_open', 'up', 'up_open']:
    #     adjust_event_first_report(conf, dur=dur, kind=kind)
    # %%
    for threshold in [100, 80, 60, 40, 20]:
        # event_condition_panel(conf, dur=3, threshold=threshold, prior='CAR_10')
        event_condition_panel(conf, dur=3, threshold=threshold, prior='CAR_3')
        # event_condition_panel(conf, dur=3, threshold=threshold, prior='X-4')

