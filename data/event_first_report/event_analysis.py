"""
(created by swmao on Jan. 26th)
- 首次研报超额收益研究，生成几张图
(modified on Feb. 9th)
- __main__中folder指定所有结果的目录名(在factors_res下)
- 添加注释

TODO: 整合 增加胜率计算！

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
        return self.data.index[l0], abs(l1-l0)


def graph_ar_car(conf, folder='event_first_report/'):
    """绘制时间前后 超额收益、累计超额收益 均值图"""

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
    save_path = res_path + folder
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


def graph_corr_d_ar_cumsum(conf, ishow=False, folder='event_first_report/'):
    """绘制相关热力图。计算方式
        - 事前j天，日后k天
        - -j天到-1天该公司股票复权收盘价总收益；事后1天到k天复权收盘价总收益

    """
    # data_path = conf['data_path']
    res_path = conf['factorsres_path']
    save_path = res_path + folder
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


def graph_dist_d_ar_afterwards(conf, ishow=False, folder='event_first_report/'):
    """日超额收益的分布"""
    # data_path = conf['data_path']
    res_path = conf['factorsres_path']
    save_path = res_path + folder

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


def table_2d_one_day(conf, folder='event_first_report/'):
    """K个2d面板，若有事件，则为事件后第K天的超额收益"""
    # data_path = conf['data_path']
    res_path = conf['factorsres_path']
    save_path = res_path + folder
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


# %%
def table_ar_adjacent_events(conf: dict, gap=20, drop_dret_over=.20, folder='event_first_report/'):
    """计算事件超额收益
        - 机构首次关注event（原instnum<5)前后-gap~gap+1日，大市为等权

    """
    # %%
    data_path = conf['data_path']
    res_path = conf['factorsres_path']
    tradeable_path = conf['a_list_tradeable']
    save_path = res_path + folder
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
    # 可交易：ipo 60 天后
    ipo60 = pd.DataFrame(pd.read_hdf(conf['a_list_tradeable'], key='ipo60')).replace(False, np.nan)
    # 复权收盘价
    adjclose = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True)
    # 复权收盘价收益
    adjret_raw = adjclose.pct_change()
    # 去除新上市60日
    adjret = adjret_raw * ipo60.reindex_like(adjclose)
    # 去除异常值
    # adjret_raw = adjret.copy()  # 备份原始值
    # adjret[adjret.abs() > drop_dret_over] = np.nan
    # 大市等权收益
    adjret_mkt = pd.DataFrame(adjret.apply(lambda s: s.mean(), axis=1), columns=['mkt'])
    # 异常收益（去异常的大市等权）
    adjret_ab = adjret.apply(lambda s: s - s.mean(), axis=1)  # 异常大，特殊公司，此处保留异常

    """
    # %% eventid为index的面板
    event_panel = event[['id', 'tradingdate', 'stockcode', 'stockname']].copy().sort_values('id')
    mask = event_panel.tradingdate <= '2021-12-31'
    event_panel = event_panel[mask]

    # %% 按列新增（所有事件）
    def visit_2d_v(td, stk, df, shift=0):
        td_idx = -1
        try:
            td_idx = df.index.get_loc(td) + shift
        except KeyError:
            print(f'KeyError: ({td}, {stk})')
            return np.nan
        finally:
            if (td_idx < 0) or (td_idx > len(df)):
                return np.nan
            return df.iloc[td_idx, :].loc[stk]

    def column_look_up(tgt, src, delay=-1, kw='r_1', msg='not found in source table'):
        key = tgt[['tradingdate', 'stockcode']]
        print(f'{kw}...')
        tgt[kw] = key.apply(lambda s: visit_2d_v(s.iloc[0], s.iloc[1], src, shift=delay), axis=1)
        print(f"nan:{tgt[kw].isna().mean() * 100: 6.2f} % {msg}")
        return tgt

    # instnum
    event_panel = column_look_up(tgt=event_panel, src=instnum, delay=0, kw='instnum')
    # max updown
    maxUp = (1 - pd.DataFrame(pd.read_hdf(tradeable_path, key='up')))  # 1: 当天涨停
    maxUpO = (1 - pd.DataFrame(pd.read_hdf(tradeable_path, key='up_open')))  # 1: 当天一字涨停
    maxDown = (1 - pd.DataFrame(pd.read_hdf(tradeable_path, key='down')))  # 1: 当天跌停
    maxDownO = (1 - pd.DataFrame(pd.read_hdf(tradeable_path, key='down_open')))  # 1: 当天一字跌停

    event_panel = column_look_up(tgt=event_panel, src=maxUp, delay=0, kw='maxUp')
    event_panel = column_look_up(tgt=event_panel, src=maxUpO, delay=0, kw='maxUpO')
    event_panel = column_look_up(tgt=event_panel, src=maxDown, delay=0, kw='maxDown')
    event_panel = column_look_up(tgt=event_panel, src=maxDownO, delay=0, kw='maxDownO')

    # %% 按行新增（指定时间编号or日期股票，给出所有）
    # T-120 ~ T+20 R
    td, stk = '2017-01-03', '000001.SZ'
    shift_, shift = -120, 20

    def visit_2d_ls(df, td, stk, shift_=0, shift=0) -> np.array:
        '''返回列表，长度为 shift - shift_ + 1'''
        td_i0 = df.index.get_loc(td)
        td_i_ = td_i0 + shift_
        td_i1 = td_i0 + shift
        td_i_, na_ = (0, - td_i_) if (td_i_ < 0) else (td_i_, 0)
        td_i1, na1 = (len(df) - 1, td_i1 - len(df) + 1) if (td_i1 >= len(df)) else (td_i1, 0)
        res = df[stk].iloc[td_i_:td_i1+1].to_list()
        res = [np.nan for _ in range(na_)] + res + [np.nan for _ in range(na1)]
        return np.array(res)

    def event_indices(td, stk):
        CAR_120 = visit_2d_ls(adjret_ab, td, stk, -120, -1).sum()
        CAR_60 = visit_2d_ls(adjret_ab, td, stk, -60, -1).sum()
        CAR_40 = visit_2d_ls(adjret_ab, td, stk, -40, -1).sum()
        CAR_20 = visit_2d_ls(adjret_ab, td, stk, -20, -1).sum()
        CAR_10 = visit_2d_ls(adjret_ab, td, stk, -10, -1).sum()
        CAR_5, CAR_4, CAR_3, CAR_2, AR_1 = visit_2d_ls(adjret_ab, td, stk, -5, -1)[::-1].cumsum()[::-1]

        AR0_5 = visit_2d_ls(adjret_ab, td, stk, 0, 5)
        AR0 = AR0_5[0]
        AR1 = AR0_5[1]
        CAR2 = AR0_5[1:3].sum()
        CAR3 = AR0_5[1:4].sum()
        CAR4 = AR0_5[1:5].sum()
        CAR5 = AR0_5[1:6].sum()
        return [CAR_120, CAR_60, CAR_40, CAR_20, CAR_10, CAR_5, CAR_4, CAR_3, CAR_2, AR_1, AR0, AR1, CAR2, CAR3, CAR4, CAR5]

    res_tmp = []
    for i_row in tqdm(event_panel.iterrows()):
        td = TD.tradedate_delta(i_row[1].tradingdate)[0]
        stk = i_row[1].stockcode
        res = event_indices(td, stk)
        res_tmp.append([i_row[1].id] + res)
    tmp = pd.DataFrame(res_tmp, columns=['id',
                                         'CAR_120', 'CAR_60', 'CAR_40', 'CAR_20', 'CAR_10',
                                         'CAR_5', 'CAR_4', 'CAR_3', 'CAR_2', 'AR_1',
                                         'AR0', 'AR1', 'CAR2', 'CAR3', 'CAR4', 'CAR5'])
    event_panel = event_panel[
        ['id', 'tradingdate', 'stockcode', 'stockname', 'instnum', 'maxUp', 'maxUpO', 'maxDown', 'maxDownO']
    ].merge(tmp, on='id', how='left')

    event_panel.to_hdf(conf['data_path'] + 'event_panel.h5', key='event_first_report')

    # car_10_2d = event_panel.pivot(index='tradingdate', columns='stockcode', values='CAR_10')
    # car_10_2d.count(axis=1).rolling(20).mean().plot(); plt.show()

    # %%
    # absolute return
    r141 = visit_2d_ls(adjret, td, stk, shift_, shift); len(r141)
    # excess return
    ar141 = visit_2d_ls(adjret_ab, td, stk, shift_, shift); len(ar141)
    # market return
    mr141 = visit_2d_ls(adjret_mkt, td, 'mkt', shift_, shift); len(mr141)
    # keep 2 from 3
    r141 - mr141 - ar141
    # cumulative absolute return
    cr140 = np.append(np.cumsum(r141[-shift_:0:-1])[::-1], np.cumsum(r141[-shift_+1:])); len(cr140)
    # cumulative excess return
    car140 = np.append(np.cumsum(ar141[-shift_:0:-1])[::-1], np.cumsum(ar141[-shift_+1:])); len(car140)

    pass


    # %%
    # Daily Return (ctc, -120~15)  16报错?
    for _delay in tqdm(range(-120, 16)):
        event_panel = column_look_up(tgt=event_panel, src=adjret, delay=_delay, kw=f"r{str(_delay).replace('-','_')}")
    # %% Daily Excess Return
    for _delay in tqdm(range(-120, 16)):
        event_panel = column_look_up(tgt=event_panel, src=adjret_ab, delay=_delay, kw=f"ar{str(_delay).replace('-','_')}")

    # %% check holding
    # path_noupdown = '/mnt/c/Users/Winst/Documents/factors_res/first_report_dur3_noupdown_n_NAew_1g(回测中不筛涨跌停)/LSGroup.csv'
    path_noupdown = '/mnt/c/Users/Winst/Documents/factors_res/first_report_dur3_updown_n_NAew_1g(回测中去新上市和停牌)/LSGroup.csv'
    # path_withupdown = '/mnt/c/Users/Winst/Documents/factors_res/first_report_dur3_withupdown_n_NAew_1g(回测中不筛涨跌停)/LSGroup.csv'
    path_withupdown = '/mnt/c/Users/Winst/Documents/factors_res/first_report_dur3_n_NAew_1g(回测中去新上市和停牌)/LSGroup.csv'

    df0 = pd.read_csv(path_noupdown, index_col=0, parse_dates=True)
    df1 = pd.read_csv(path_withupdown, index_col=0, parse_dates=True)
    td, td1 = '2021-10-11', '2021-10-08'
    tmp = pd.concat([df0.loc[td], df1.loc[td]], axis=1); tmp.columns=['no', 'with']
    difference_of_holding_weight = tmp[df1.loc[td] - df0.loc[td] > 0]
    sig0 = '/mnt/c/Users/Winst/Documents/factors_csv/first_report_dur3_updown.csv'
    sig1 = '/mnt/c/Users/Winst/Documents/factors_csv/first_report_dur3.csv'
    sig = pd.concat([pd.read_csv(sig0,index_col=0,parse_dates=True).loc[td1],
                     pd.read_csv(sig1,index_col=0,parse_dates=True).loc[td1]],axis=1)
    sig.columns = ['no', 'with']
    difference_of_raw_signal = sig[sig['with'] - sig['no'] != 0].dropna(axis=0)
    assert set(difference_of_raw_signal.index) == set(difference_of_holding_weight.index)
    print(difference_of_raw_signal.index.to_list())
    tmp = adjret[['002170.SZ', '300179.SZ', '300472.SZ']]
    mask = (tmp.index >= '2021-09-27') & (tmp.index <= '2021-10-11')
    tmp = tmp[mask]
    tmp = tmp.sum(axis=1) / df1.loc[td].sum()  # 去权重

    mask = (event_panel.tradingdate >= '2021-09-29') & (event_panel.tradingdate < '2021-10-11')
    # & (event_panel[['MUD0', 'OMUD0']].sum(axis=1) > 0)
    tmp = event_panel[mask].copy()

    tmp.to_csv('/home/swmao/tmp.csv', encoding='GBK')

    # %%
    hdf_file = '/mnt/c/Users/Winst/Documents/factors_res/event_first_report/' + 'event_panel.h5'
    event_panel = pd.DataFrame(pd.read_hdf(hdf_file, key='r141ar141'))

    # %%
    hdf_file = '/mnt/c/Users/Winst/Documents/data_local/' + 'event_panel.h5'
    event_panel.to_hdf(hdf_file, key='r141ar141')
    
    """

    # %% 表：记录id，事件日+-120日（不够则空），复权收盘价收益率（当日比昨日）
    event_adjacent_returns = pd.DataFrame(columns=event.id, index=range(-gap, gap+1))
    for irow in tqdm(range(len(event))):  # 15996个事件，规模极大
        row = event.iloc[irow, :]
        # break
        stk_id = row['stockcode']
        event_date = row['tradingdate']
        row_id = row['id']
        date0, l_gap = TD.tradedate_delta(event_date, -gap)
        date1, r_gap = TD.tradedate_delta(event_date, gap)
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


# %%
if __name__ == '__main__':
    # %%
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    folder = 'event_first_report/'
    drop_dret_over = 0.10

    # %%
    table_ar_adjacent_events(conf, gap=240, drop_dret_over=drop_dret_over, folder=folder)
    graph_ar_car(conf, folder=folder)
    graph_corr_d_ar_cumsum(conf, ishow=False, folder=folder)
    graph_dist_d_ar_afterwards(conf, ishow=False, folder=folder)
    table_2d_one_day(conf, folder=folder)
