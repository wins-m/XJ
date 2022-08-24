"""
(created on March 21st by swmao)
- 计算胜率
"""
import pandas as pd
import numpy as np
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.backtester import *
from factor_backtest.single_test import single_test


def win_percentage(price_adj: pd.DataFrame, eve2d: pd.DataFrame, dur: int, save_path: str) -> pd.DataFrame:
    """
    根据开仓信号eve2d和持仓时长dur计算胜率
    :param price_adj: 可行的调整后价格，日期对应的是当天买入
    :param eve2d: 开仓信号
    :param dur: 持有dur天后卖出
    :param save_path: 胜率xlsx文件存储位置
    :return: 胜率计算的面板
    """
    long_price = price_adj.reindex_like(eve2d) * eve2d
    short_price = price_adj.shift(-dur).reindex_like(eve2d) * eve2d

    s_l_p = (short_price - long_price)
    win_r_sr = ((s_l_p > 0).sum(axis=1) / s_l_p.count(axis=1)).rename('daily')
    year_month = s_l_p.index.to_series().apply(lambda x: x.strftime('%Y-%m')).rename('month')
    year = year_month.apply(lambda x: x.split('-')[0]).rename('year')
    win_r_df = pd.DataFrame([year, year_month, win_r_sr]).T.reset_index()

    tmp = s_l_p.groupby(year_month).apply(lambda s: (s > 0).sum().sum() / s.count().sum())
    tmp = tmp.rename('monthly').reset_index()
    win_r_df = win_r_df.merge(tmp, on='month', how='left')

    tmp = s_l_p.groupby(year).apply(lambda s: (s > 0).sum().sum() / s.count().sum())
    tmp = tmp.rename('yearly').reset_index()
    win_r_df = win_r_df.merge(tmp, on='year', how='left')

    win_r_all = (s_l_p > 0).sum().sum() / s_l_p.count().sum()
    win_r_df['whole_period'] = win_r_all

    win_r_df.to_excel(save_path)
    return win_r_df


# %
def main():
    # %
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    # %  计算胜率等
    src_path = conf['data_path'] + 'event_first_report.csv'  # 格式和服务器一致
    event = pd.read_csv(src_path)
    eve2d = event.pivot('tradingdate', 'stockcode', 'fv')
    eve2d.index = pd.to_datetime(eve2d.index)

    kind = 'ctc'
    save_path_ = conf['factorsres_path'] + 'event_first_report/{}'
    if kind == 'ctc':
        price_adj = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True)
    else:
        raise ValueError

    win_percentage(price_adj, eve2d, dur=3, save_path=save_path_.format('daily_wp.xlsx'))

    # % single test
    eve2d3f = eve2d.fillna(method='ffill', limit=2)


