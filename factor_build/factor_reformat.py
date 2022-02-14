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


# %%
if __name__ == '__main__':
    # %%
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    # %%
    # adjust_factor_pe_residual(conf)
    # %%
    kind = 'updown'
    dur = 3
    # for d in range(1, 6):
    for kind in [None, 'updown', 'updown_open', 'up', 'up_open']:
        adjust_event_first_report(conf, dur=dur, kind=kind)
