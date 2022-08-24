"""
(created by swmao on Aug. 24th)

"""
import os
from datetime import timedelta
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sqlalchemy import create_engine

_PATH = '/home/swmao/event_analysis/'
_CONF = {

    # 结果目录
    'save_path_': _PATH + "/result0824/{}",

    # CSV path, 0/1 signal: tradingdate, stockcode (Attention: 需要额外提供！)
    # e.g. (column "stockname" is not necessary)
    #   id,fv,stockcode,stockname,tradingdate
    #   160105000338,1,000338.SZ,潍柴动力,2016-01-05
    #   160105000516,1,000516.SZ,国际医学,2016-01-05
    #   160105000898,1,000898.SZ,鞍钢股份,2016-01-05
    #   160105002034,1,002034.SZ,旺能环境,2016-01-05
    #   160105002200,1,002200.SZ,ST云投,2016-01-05
    'event_signal_file': f"{_PATH}event_first_report.csv",

    # CSV path, daily close to close return; caution: all event-related asset should have corresponding close return
    'close_adj_file': f"{_PATH}/stk_marketdata_closeAdj.csv",

    # Trading date
    'tdays_d': f"{_PATH}/tdays_d.csv",

    # 计算胜率时的, 拟持仓天数
    'hold_duration': 1,

    # CSV path for tradeable status: 0/1 or TRUE/FALSE csv file path; not consider if None
    'tradeable_status': None,

}
_ENG = {'user': 'intern01',
        'password': 'rh35th',
        'host': '192.168.1.104',
        'port': '3306',
        'dbname': 'jeffdatabase'}


def get_remote(close_adj_file, tdays_d, bd, ed):

    def conn_mysql(eng: dict):
        """根据dict中的服务器信息，连接mysql"""
        user = eng['user']
        password = eng['password']
        host = eng['host']
        port = eng['port']
        dbname = eng['dbname']
        engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}?charset=UTF8MB4')
        return engine

    def mysql_query(query, engine, telling=True) -> pd.DataFrame:
        """mysql接口，返回DataFrame"""
        if telling:
            print(query)
        return pd.read_sql_query(query, engine)

    engine_jeff = conn_mysql(_ENG)
    if os.path.exists(close_adj_file):
        close_adj = pd.read_csv(close_adj_file, index_col=0, parse_dates=True)
        bd0 = bd if isinstance(bd, str) else bd.strftime('%Y-%m-%d')
        ed0 = close_adj.index[0].strftime('%Y-%m-%d')
        bd1 = close_adj.index[-1].strftime('%Y-%m-%d')
        ed1 = ed if isinstance(ed, str) else ed.strftime('%Y-%m-%d')

        df0 = mysql_query(
            query=f"SELECT tradingdate,stockcode,close*adjfactor AS closeAdj"
                  f" FROM jeffdatabase.stk_marketdata"
                  f" WHERE tradingdate>='{bd0}'"
                  f" AND tradingdate<'{ed0}'",
            engine=engine_jeff
        )
        df0 = df0.pivot('tradingdate', 'stockcode', 'closeAdj')
        df0.index = pd.to_datetime(df0.index)

        df1 = mysql_query(
            query=f"SELECT tradingdate,stockcode,close*adjfactor AS closeAdj"
                  f" FROM jeffdatabase.stk_marketdata"
                  f" WHERE tradingdate>'{bd1}'"
                  f" AND tradingdate<='{ed1}'",
            engine=engine_jeff
        )
        df1 = df1.pivot('tradingdate', 'stockcode', 'closeAdj')
        df1.index = pd.to_datetime(df1.index)

        res = pd.concat([df0, close_adj, df1])
        res.to_csv(close_adj_file)

    else:
        df = mysql_query(
            query=f"SELECT tradingdate,stockcode,close*adjfactor AS closeAdj"
                  f" FROM jeffdatabase.stk_marketdata"
                  f" WHERE tradingdate>='{bd}'"
                  f" AND tradingdate<='{ed}'",
            engine=engine_jeff
        )
        df = df.pivot('tradingdate', 'stockcode', 'closeAdj')
        df.index = pd.to_datetime(df.index)
        df.to_csv(close_adj_file)

    # tdays_d
    df = mysql_query(
        query=f"SELECT tradingdate"
              f" FROM jeffdatabase.tdays_d"
              f" WHERE tradingdate>='{bd}'"
              f" AND tradingdate<='{ed}'",
        engine=engine_jeff
    )
    df.to_csv(tdays_d, header=None, index=None)


class TradeDate(object):

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.last_date = self.data.index[-1]  # 最后一个交易日

    def tradedate_delta(self, date, gap=0) -> tuple:
        """对自然日date找gap个交易日以后的日期，gap为负表示向前
            - 若超出已有范围，则取self.data中第一个/最后一个交易日日期
            - 返回日期，以及实际间隔的交易日数量（因为可能超出范围）
        """
        try:
            l1 = self.data.index.get_loc(date)
        except KeyError:
            d0 = pd.to_datetime(date)
            if d0 >= self.last_date:
                print(f'`{date}`超出范围')
                return None, None
            else:
                print(f'`{date}`不是交易日，替换以下一个交易日')
                d1 = d0 + timedelta(1)
                date1 = d1.date().__str__()
                return self.tradedate_delta(date=date1, gap=gap)
        l0 = max(0, l1 + gap)  #
        l0 = min(l0, len(self.data) - 1)
        return self.data.index[l0], abs(l1 - l0)


def table_ar_adjacent_events(
        event: pd.DataFrame,
        adj_close: pd.DataFrame,
        TD: TradeDate,
        save_path: str,
        gap0: int = -20,
        gap1: int = 20,
        drop_daily_ret_over=0.20,
) -> pd.DataFrame:
    """计算事件超额收益"""

    adjret = adj_close.pct_change()  # 复权收盘价收益

    adjret[adjret.abs() > drop_daily_ret_over] = np.nan  # 直接去除日收益异常值

    adjret_mkt = pd.DataFrame(adjret.apply(lambda s: s.mean(), axis=1), columns=['mkt'])  # 大市等权收益

    # % 表：记录id，事件日+-20日（不够则空），复权收盘价收益率（当日比昨日）
    event_adjacent_returns = pd.DataFrame(columns=event.id, index=range(gap0, gap1 + 1))
    for i_row in tqdm(range(15000, len(event))):  # TODO
        row = event.iloc[i_row, :]
        stk_id = row['stockcode']
        event_date = row['tradingdate']
        date0, l_gap = TD.tradedate_delta(event_date, gap0)
        date1, r_gap = TD.tradedate_delta(event_date, gap1)
        # return within date range
        adjacent_ret = adjret.loc[date0:date1, stk_id:stk_id]
        adjacent_mret = adjret_mkt.loc[date0:date1]
        # ar = r_avg - r_mkt
        adjacent_ar = adjacent_ret - adjacent_mret.values.reshape(-1, 1)
        assert adjacent_ar.__len__() == l_gap + r_gap + 1
        adjacent_ar.index = range(-l_gap, r_gap + 1)

        row_id = f"{event_date.replace('-', '')}{stk_id.split('.')[0]}"
        adjacent_ar.columns = [row_id]
        # adjacent_ar.reindex_like(event_adjacent_returns.loc[:, row_id:row_id])
        event_adjacent_returns.loc[-l_gap:r_gap + 1, row_id:row_id] = adjacent_ar.values
    # 存表
    event_adjacent_returns.to_csv(save_path)

    return event_adjacent_returns


def graph_ar_car(
        save_path_,
        ret_ab,
        gap_range,
        fig_size=(10, 5),
):
    """绘制时间前后 超额收益、累计超额收益 均值图"""

    def plot_ar_car(ar, save_path=None):
        car = ar.cumsum()
        car = car - car.loc[0]

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        ax.bar(ar.index, ar.values, width=.8, color='k')
        ax.axvline(x=0, ls=':', color='r')

        ax2 = ax.twinx()
        ax2.stackplot(car.index, car.values, alpha=.2, color='b')
        # fig.legend(loc=1)
        plt.grid()
        plt.title('event abnormal returns')
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()

    # abnormal return adjacent
    if isinstance(ret_ab, str):
        event_abnormal_returns = pd.read_csv(ret_ab, index_col=0)
    elif isinstance(ret_ab, pd.DataFrame):
        event_abnormal_returns = ret_ab
    else:
        raise Exception('ret_ab')

    # 等权平均
    ret_gap_nna_mean = event_abnormal_returns.mean(axis=1)
    # 绘图
    for gap0, gap1 in gap_range:
        plot_ar_car(ret_gap_nna_mean.loc[gap0:gap1], save_path_.format(f'ar_car({gap0},{gap1}).png'))


def graph_corr_d_ar_cumsum(
        save_path_,
        ret_ab,
        gap0=-20,
        gap1=20,
        corr_mtd='spearman',
        fig_size=(20, 16),
        ishow=False,
):
    """绘制correlation coefficient热力图。计算方式
        - 事前j天，日后k天
        - -j天到-1天该公司股票复权收盘价总收益；事后1天到k天复权收盘价总收益
    """

    def corr_heat_map(x, title, xlabel, ylabel):
        f, ax = plt.subplots(figsize=fig_size)
        sns.heatmap(x,
                    annot=True if (gap1 - gap0) <= 40 else False,
                    cmap='RdBu',
                    ax=ax,
                    annot_kws={'size': 9, 'weight': 'bold', 'color': 'white'})
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(save_path + f'{title}.png')
        if ishow:
            plt.show()
        else:
            plt.close()

    save_path = save_path_.format('')
    if isinstance(ret_ab, str):
        event_abnormal_returns = pd.read_csv(ret_ab, index_col=0)
    elif isinstance(ret_ab, pd.DataFrame):
        event_abnormal_returns = ret_ab
    else:
        raise Exception('ret_ab illegal')

    # 超额收益相关性
    corr_mat_aa = event_abnormal_returns.T.corr(method=corr_mtd)
    corr_mat_aa = corr_mat_aa.loc[gap0:0, 0:gap1]
    corr_mat_aa.to_excel(save_path + f'AR-AR Corr ({corr_mtd}).xlsx')
    corr_heat_map(x=corr_mat_aa,
                  title=f'AR-AR Corr({gap0},{gap1})({corr_mtd})',
                  xlabel='AR',
                  ylabel='AR')

    # 累计超额收益（不计入事件当天）
    cum_sum_ret0 = event_abnormal_returns.loc[-1:gap0:-1, :].cumsum().loc[::-1].T
    cum_sum_ret1 = event_abnormal_returns.loc[1:gap1, :].cumsum().T
    cum_sum_ret = pd.concat([cum_sum_ret0, cum_sum_ret1], axis=1)

    # 累计超额收益相关性
    corr_mat_cc = cum_sum_ret.corr(method=corr_mtd)
    corr_mat_cc = corr_mat_cc.loc[gap0:-1, 1:gap1]  # CAR0 is nonsense
    corr_mat_cc.to_excel(save_path + f'CAR-CAR Corr({corr_mtd}).xlsx')
    corr_heat_map(x=corr_mat_cc,
                  title=f'CAR-CAR Corr({gap0},{gap1})({corr_mtd})',
                  xlabel='CAR',
                  ylabel='CAR')

    # 事前超额收益 事后累计超额收益
    corr_mat_ca = pd.concat(
        [
            event_abnormal_returns.T.loc[:, :0],
            cum_sum_ret.loc[:, 1:]
        ],
        axis=1
    ).corr(method=corr_mtd)
    corr_mat_ca = corr_mat_ca.loc[1:gap1, gap0:0]
    corr_mat_ca.to_excel(save_path + f'CAR-AR Corr({corr_mtd}).xlsx')
    corr_heat_map(x=corr_mat_ca,
                  title=f'CAR-AR Corr({gap0},{gap1})({corr_mtd})',
                  xlabel='AR',
                  ylabel='CAR')

    # 事前累计超额收益 事后超额收益
    corr_mat_ac = pd.concat(
        [
            cum_sum_ret.loc[:, :-1],
            event_abnormal_returns.T.loc[:, 0:]
        ],
        axis=1
    ).corr(method=corr_mtd)
    corr_mat_ac = corr_mat_ca.loc[0:gap1, gap0:-1]
    corr_mat_ac.to_excel(save_path + f'AR-CAR Corr({corr_mtd}).xlsx')
    corr_heat_map(x=corr_mat_ac,
                  title=f'AR-CAR Corr({gap0},{gap1})({corr_mtd})',
                  xlabel='CAR',
                  ylabel='AR')


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


def main():
    conf = _CONF.copy()

    # Event panel
    print('Loading local event panel...')
    event = pd.read_csv(conf['event_signal_file'])
    eve2d = event.pivot('tradingdate', 'stockcode', 'fv')
    eve2d.index = pd.to_datetime(eve2d.index)

    # Download AdjClose and t_days
    print('Loading remote close_adj and tdays_d...')
    get_remote(close_adj_file=conf['close_adj_file'],
               tdays_d=conf['tdays_d'],
               bd=eve2d.index[0].strftime('%Y-%m-%d'),
               ed=(eve2d.index[-1] + timedelta(60)).strftime('%Y-%m-%d'))

    # Price adjusted
    price_adj = pd.read_csv(conf['close_adj_file'], index_col=0, parse_dates=True)
    if conf['tradeable_status'] is not None:
        tradeable = pd.read_csv(conf['tradeable_status'], index_col=0, parse_dates=True)
        price_adj = price_adj * tradeable.reindex_like(price_adj).fillna(1)

    # Result dir
    save_path_ = conf['save_path_']
    os.makedirs(save_path_.format(''), exist_ok=True)

    # 异常收益存储地址
    return_abnormal_path = save_path_.format('event_abnormal_returns.csv')
    print(f'Abnormal return saved in {return_abnormal_path}')

    # Trading date parser
    tradedates = pd.read_csv(conf['tdays_d'], header=None, index_col=0, parse_dates=True)
    TD = TradeDate(tradedates)

    # 计算事件前后异常收益
    print('Calculating adjacent abnormal returns...')
    # ret_ab = \
    table_ar_adjacent_events(
        event=event,
        adj_close=price_adj,
        TD=TD,
        save_path=return_abnormal_path,
        gap0=-20,
        gap1=20,
        drop_daily_ret_over=0.11
    )

    # 事件前后超额收益图像 AR CAR
    print('Plot average adjacent abnormal returns...')
    ret_ab = return_abnormal_path  # ret_ab
    graph_ar_car(
        save_path_=save_path_,
        ret_ab=ret_ab,
        gap_range=[(-20, 20), (-10, 10), (-5, 5)],
        fig_size=(10, 5),
    )

    # 事件前后超额收益相关性
    print('Plot correlation graphs...')
    graph_corr_d_ar_cumsum(
        save_path_=save_path_,
        ret_ab=return_abnormal_path,
        gap0=-20,
        gap1=20,
        corr_mtd='spearman',  # or 'pearson'
        fig_size=(20, 16),
        ishow=False
    )

    graph_corr_d_ar_cumsum(
        save_path_=save_path_,
        ret_ab=return_abnormal_path,
        gap0=-20,
        gap1=20,
        corr_mtd='pearson',
        fig_size=(20, 16),
        ishow=False
    )

    # 计算胜率
    print('Calculating winning percentage...')
    hold_duration = conf['hold_duration']
    win_percentage(
        price_adj=price_adj,
        eve2d=eve2d,
        dur=hold_duration,
        save_path=save_path_.format(f'win_percent(hd={hold_duration}).xlsx')
    )

    print('exit')


if __name__ == '__main__':
    main()
