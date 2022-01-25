"""
(created by swmao on Jan. 21st)
各类因子的预处理，做成可进入回测的2D面板
"""

import pandas as pd
import numpy as np
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


def adjust_event_first_report(conf: dict):
    path = conf['event_first_report']
    csv_path = conf['factorscsv_path']
    dur = 5
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.fillna(method='ffill', limit=dur)
    template = read_single_factor(conf['a_list_tradeable'], hdf_k='tradeable')
    df = df.reindex_like(template).fillna(0)
    df = df * template.replace(False, np.nan)
    df.to_csv(csv_path + 'first_report.csv')


if __name__ == '__main__':
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    #
    # adjust_factor_pe_residual(conf)

