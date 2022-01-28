"""
(created by swmao on Jan. 26th)
首次研报超额收益研究，生成几张图
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from datetime import timedelta
import seaborn as sns
import os


class TradeDate(object):

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.last_date = self.data.index[-1]

    def tradedatedelta(self, date, gap=0):
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
                return self.tradedatedelta(date=date1, gap=gap)
        l0 = max(0, l1 + gap)
        l0 = min(l0, len(self.data) - 1)
        return self.data.index[l0], abs(l1-l0)


def table_ar_adjacent_events(conf: dict, gap=20, drop_dret_over=.20):
    """计算机构首次关注event（原instnum<5)前后-gap~gap+1日的超额收益，大市为等权"""
    data_path = conf['data_path']
    res_path = conf['factorsres_path']
    save_path = res_path + 'event_first_report/'
    os.makedirs(save_path, exist_ok=True)
    # 存储地址
    save_path = save_path + 'event_abnormal_returns.csv'  # 'event_adjacent_returns.csv'
    # 事件记录：stockcode, tradingdate, fv(==1)
    event = pd.read_csv(data_path + 'event_first_report.csv')
    # 机构关注数
    instnum = pd.read_csv(conf['instnum_180'], index_col=0, parse_dates=True)
    # 交易日列表
    tradedates = pd.read_csv(conf['tdays_d'], header=None, index_col=0, parse_dates=True)
    TD = TradeDate(tradedates)
    # ipo 60 天后
    ipo60 = pd.DataFrame(pd.read_hdf(conf['a_list_tradeable'], key='ipo60')).replace(False, np.nan)
    # 复权收盘价
    adjclose = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True)
    # 复权收盘价收益
    adjret = adjclose.pct_change()
    # 去除新上市60日
    adjret = adjret * ipo60.reindex_like(adjclose)
    # 去除异常值
    adjret[adjret.abs() > drop_dret_over] = np.nan
    # 异常收益（去除大市等权）
    adjret_mkt = adjret.apply(lambda s: s.mean(), axis=1)
    # adjret_ab = adjret.apply(lambda s: s - s.mean(), axis=1)  # 异常大，为什么？
    # 表：记录id，事件日+-120日（不够则空），复权收盘价收益率（当日比昨日）
    event_adjacent_returns = pd.DataFrame(columns=event.id, index=range(-gap, gap+1))
    for irow in tqdm(range(len(event))):  # 15996个事件，规模极大
        row = event.iloc[irow, :]
        # break
        stk_id = row['stockcode']
        event_date = row['tradingdate']
        row_id = row['id']
        date0, l_gap = TD.tradedatedelta(event_date, -gap)
        date1, r_gap = TD.tradedatedelta(event_date, gap)
        if (instnum.loc[event_date, stk_id] > 5) or (date0 is None):
            # 计入新机构已有超过5加关注，或者事件日超出日期范围
            continue
        adjacent_ret = adjret.loc[date0:date1, stk_id:stk_id]
        adjacent_mret = adjret_mkt.loc[date0:date1]
        adjacent_ar = adjacent_ret - adjacent_mret.values.reshape(-1, 1)
        # adjacent_ar = adjret_ab.loc[date0:date1, stk_id:stk_id]  # 为何速度慢很多？？
        assert adjacent_ar.__len__() == l_gap + r_gap + 1
        adjacent_ar.index = range(-l_gap, r_gap+1)
        adjacent_ar.columns = [row_id]
        # adjacent_ar.reindex_like(event_adjacent_returns.loc[:, row_id:row_id])
        event_adjacent_returns.loc[-l_gap:r_gap+1, row_id:row_id] = adjacent_ar
    # 存表
    event_adjacent_returns.to_csv(save_path)


def graph_ar_car(conf):

    def plot_ar_car(ar, save_path=None):
        car = ar.cumsum()
        car = car - car.loc[0]

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.bar(ar.index, ar.values, width=.8, color='k')
        ax.axvline(x=0, ls=':', color='r')

        ax2 = ax.twinx()
        ax2.stackplot(car.index, car.values, alpha=.2, color='b')
        # fig.legend(loc=1)
        plt.grid()
        plt.title('Event Abnormal Returns')
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)
            plt.close()

    # data_path = conf['data_path']
    res_path = conf['factorsres_path']
    save_path = res_path + 'event_first_report/'
    os.makedirs(save_path, exist_ok=True)

    event_abnormal_returns = pd.read_csv(save_path + 'event_abnormal_returns.csv', index_col=0, parse_dates=True)
    """tmp = event_abnormal_returns.copy()
    tmp.apply(lambda s: (s > s.median() + s.std() * 8).sum(), axis=1).plot(); plt.show()
    tmp.apply(lambda s: (s.median() + s.std() * 8), axis=1).plot(); plt.show()
    (tmp > drop_dret_over).sum(axis=1).plot(); plt.show()"""
    # 范围内
    ret_gap_nna = event_abnormal_returns  # event_abnormal_returns.dropna(axis=1)
    """
    print(event_abnormal_returns.shape, ret_gap_nna.shape)
    plt.hist(ret_gap_nna.values.reshape(-1, 1))
    plt.show()
    """
    # 等权平均
    ret_gap_nna_mean = ret_gap_nna.mean(axis=1)
    # 绘图
    ar = ret_gap_nna_mean.copy()
    for gap in [240, 120, 60, 30, 15]:
        plot_ar_car(ar.loc[-gap:gap], f'{save_path}AR_CAR_{gap}.png')
        #


def graph_corr_d_ar_cumsum(conf, ishow=False):
    """
    绘制相关热力图。计算方式
    - 事前j天，日后k天
    - -j天到-1天该公司股票复权收盘价总收益；事后1天到k天复权收盘价总收益

    """
    # data_path = conf['data_path']
    res_path = conf['factorsres_path']
    save_path = res_path + 'event_first_report/'
    os.makedirs(save_path, exist_ok=True)
    event_abnormal_returns = pd.read_csv(save_path + 'event_abnormal_returns.csv', index_col=0, parse_dates=True)

    cumsumret0 = event_abnormal_returns.loc[-1:-120:-1, :].cumsum().loc[::-1].T
    cumsumret1 = event_abnormal_returns.loc[1:120, :].cumsum().T
    cumsumret = pd.concat([cumsumret0, cumsumret1], axis=1)

    corrmatrix = cumsumret.corr(method='pearson')
    subcorrmatrix = corrmatrix.loc[-120:-1, 1:120].iloc[::-1]
    subcorrmatrix.to_excel(save_path + 'ar_cumsum_corr(pearson).xlsx')

    corrmatrix2 = cumsumret.corr(method='spearman')
    subcorrmatrix2 = corrmatrix2.loc[-120:-1, 1:120].iloc[::-1]
    subcorrmatrix2.to_excel(save_path + 'ar_cumsum_corr(spearman).xlsx')

    gap, data = 25, subcorrmatrix
    for gap in [25, 120]:
        for mtd, data in zip(['pearson', 'spearman'], [subcorrmatrix, subcorrmatrix2]):
            x = data.iloc[:gap, :gap]
            annot = True if gap <= 25 else False
            f, ax = plt.subplots(figsize=(20, 16))
            sns.heatmap(x, annot=annot, cmap='RdBu', ax=ax, annot_kws={'size': 9, 'weight': 'bold', 'color': 'white'})
            title = f'corr{gap}_{mtd}'
            plt.title(title)
            plt.savefig(save_path+f'{title}.png')
            if ishow:
                plt.show()
            else:
                plt.close()
    #


def graph_dist_d_ar_afterwards(conf, ishow=False):
    """日超额收益的分布"""
    # data_path = conf['data_path']
    res_path = conf['factorsres_path']
    save_path = res_path + 'event_first_report/'

    os.makedirs(save_path, exist_ok=True)
    event_abnormal_returns = pd.read_csv(save_path + 'event_abnormal_returns.csv', index_col=0, parse_dates=True)

    # % Violin Plot
    for di in [5, 20, 60]:
        df = event_abnormal_returns.loc[1:di]
        df = df.stack().reset_index()[['level_0', 0]]
        df.columns = ['days', 'AR']

        plt.figure(figsize=(10 if di < 21 else di // 2, 10))
        sns.violinplot(x='days', y='AR', data=df)
        plt.grid()
        plt.savefig(save_path+f'violin{di}')
        if ishow:
            plt.show()
        else:
            plt.close()

    # % Box Plot
    df = event_abnormal_returns.loc[1:5].T
    df.plot.box(title='Daily Abnormal Return After Event')
    plt.savefig(save_path+'ARbox.png')
    if ishow:
        plt.show()
    else:
        plt.close()


def table_2d_one_day(conf):
    """K个2d面板，若有事件，则为事件后第K天的超额收益"""
    # data_path = conf['data_path']
    res_path = conf['factorsres_path']
    save_path = res_path + 'event_first_report/'
    os.makedirs(save_path, exist_ok=True)

    event_abnormal_returns = pd.read_csv(save_path + 'event_abnormal_returns.csv', index_col=0, parse_dates=True)

    # 000001 -> 000001.SZ
    ipo60 = pd.DataFrame(pd.read_hdf(conf['a_list_tradeable'], key='ipo60'))
    stock_code_dir = ipo60.columns.to_frame()
    stock_code_dir.index = stock_code_dir.loc[:, 0].apply(lambda x: x[:6]).values

    di = 1
    for di in tqdm(range(21)):
        sr = event_abnormal_returns.loc[di]
        #
        df = pd.DataFrame(sr)
        df.columns = ['ar']
        df['id'] = df.index
        # tradingdate格式datetime
        df['tradingdate'] = df.id.apply(lambda x: pd.to_datetime(x[:8]))
        # 还原stockcode交易所后缀
        df['stockcode'] = df.id.apply(lambda x: stock_code_dir.loc[x[-6:]].values[0])
        # Pivot
        df2d = df.pivot(index='tradingdate', columns='stockcode', values='ar')
        # drop all NA columns
        df2d = df2d.dropna(axis=1, how='all')
        # save
        df2d.to_hdf(save_path + 'one_day_AR_after_event_2d.hdf', key=f'D{di}')


if __name__ == '__main__':
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    gap = 240

    # table_ar_adjacent_events(conf, gap, drop_dret_over=0.10)
    # graph_ar_car(conf)
    # graph_corr_d_ar_cumsum(conf)
    # graph_dist_d_ar_afterwards(conf, ishow=False)
    # table_2d_one_day(conf)