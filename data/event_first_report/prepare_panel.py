"""
(created by swmao on Feb. 17th)
从数据库的原始文件，准备事件面板
- instnum: stk_west_instnum_180 fv(tradingdate, stockcode) 7.6% nan 无记录（无关注）
    - 用哪个？ stk_west_eps_instnum _close _open _tomorrow
- maxUp: stk_maxupordown 事件日 涨停[maxupordown=1](1) 不涨停(0) 非交易日(nan)
- maxUpO: stk_maxupordown, stk_marketdata 时间日 一字涨停[(maxupordown=1)&(open=close)] 非一字涨停(0) 非交易日(nan)
- maxDown: 跌停，同上
- maxDownO: 一字跌停，同上
- ipo_date: 上市日期
- delist_date: 退市日期
- lastST: 最近一个停牌日
- isTradeDay: 是交易日
- R_120, R_119, ..., R_1, R0, R_1, ..., R20: 事件对象-120~20日绝对收益
- CR_120, CR_119, ..., CR_20: 事件对象-120~20日累计绝对收益
- AR_120, AR_119, ..., AR0, AR1, ..., AR20: -120~20日超额收益
- CAR_120, CAR_119, ..., CAR0, ..., CAR20: -120~20日累计超额收益
(Feb. 21st)
将（二）分组更新到面板，二分组规则：只有一个事件，则H组L组均持有；事件数为奇数，交替0101令L组/H组多一个事件，使得两组事件数平衡
-
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta, datetime
import os

# %%
import yaml

conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
folder = 'event_first_report/'

data_path = conf['data_path']
res_path = conf['factorsres_path']
tradeable_path = conf['a_list_tradeable']
save_path = res_path + folder
os.makedirs(save_path, exist_ok=True)
print(save_path)

# %%
# 事件面板
event = pd.read_csv(data_path + 'event_first_report.csv')
# 机构关注数
instnum = pd.read_csv(conf['instnum_180'], index_col=0, parse_dates=True)
# # 可交易：ipo 60 天后
# ipo60 = pd.DataFrame(pd.read_hdf(conf['a_list_tradeable'], key='ipo60')).replace(False, np.nan)
# ipo日期
stk_ipo_date = pd.read_csv(conf['stk_ipo_date']).set_index('stockcode')
# # 停牌
# a_list_suspendsymbol = pd.read_csv(conf['a_list_suspendsymbol'])
# 复权收盘价
adjclose = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True)
# 复权收盘价收益
adjret = adjclose.pct_change()
# 大市等权收益
adjret_mkt = pd.DataFrame(adjret.apply(lambda s: s.mean(), axis=1), columns=['mkt'])
# 异常收益（去异常的大市等权）
adjret_ab = adjret.apply(lambda s: s - s.mean(), axis=1)  # 异常大，特殊公司，此处保留异常
# 交易日列表
tradedates = adjret.iloc[:, 0:0].copy()  # pd.read_csv(conf['tdays_d'], header=None, index_col=0, parse_dates=True)

print(event.head())


# # 去除新上市60日
# adjret = adjret_raw * ipo60.reindex_like(adjclose)

# # 去除异常值
# drop_dret_over=.20
# adjret_raw = adjret.copy()  # 备份原始值
# adjret[adjret.abs() > drop_dret_over] = np.nan


# %% 交易日
class TradeDate(object):

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.last_date = self.data.index[-1]  # 最后一个交易日

    def tradedate_delta(self, date, gap=0, saying=False) -> tuple:
        """对自然日date找gap个交易日以后的日期，gap为负表示向前
            - 若超出已有范围，则取self.data中第一个/最后一个交易日日期
            - 返回日期，以及实际间隔的交易日数量（因为可能超出范围）
        """
        try:
            l1 = self.data.index.get_loc(date)
        except KeyError:
            d0 = pd.to_datetime(date)
            if d0 >= self.last_date:
                if saying:
                    print(f'`{date}`超出范围')
                return None, None
            else:
                if saying:
                    print(f'`{date}`不是交易日，替换以下一个交易日')
                d1 = d0 + timedelta(1)
                date1 = d1.date().__str__()
                return self.tradedate_delta(date=date1, gap=gap)
        l0 = max(0, l1 + gap)  #
        l0 = min(l0, len(self.data) - 1)
        return self.data.index[l0], abs(l1 - l0)  # 事件记录：stockcode, tradingdate, fv(==1)


TD = TradeDate(tradedates)

# %% eventid为index的面板
event_panel = event[['id', 'tradingdate', 'stockcode', 'stockname']].copy().sort_values('id').set_index('id')
# mask = event_panel.tradingdate <= '2021-12-31'
# event_panel = event_panel[mask]
event_panel.tail()


# %% 按列新增（所有事件）
# event_panel_bak = event_panel.copy()  # 备份

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


def column_look_up(tgt, src, delay=-1, kw='r_1', msg=None):
    key = tgt[['tradingdate', 'stockcode']]
    print(f'{kw}...')
    tgt[kw] = key.apply(lambda s: visit_2d_v(s.iloc[0], s.iloc[1], src, shift=delay), axis=1)
    if msg is None:
        msg = 'not found in source table'
    print(f'nan:{tgt[kw].isna().mean() * 100: 6.2f} % {msg}')
    return tgt


# instnum
event_panel = column_look_up(tgt=event_panel, src=instnum, delay=0, kw='instnum')

# max updown
maxUp = (1 - pd.DataFrame(pd.read_hdf(tradeable_path, key='up')))  # 1: 当天涨停
maxUpO = (1 - pd.DataFrame(pd.read_hdf(tradeable_path, key='up_open')))  # 1: 当天一字涨停
maxDown = (1 - pd.DataFrame(pd.read_hdf(tradeable_path, key='down')))  # 1: 当天跌停
maxDownO = (1 - pd.DataFrame(pd.read_hdf(tradeable_path, key='down_open')))  # 1: 当天一字跌停

event_panel = column_look_up(tgt=event_panel, src=maxUp, delay=0, kw='maxUp')  # 
event_panel = column_look_up(tgt=event_panel, src=maxUpO, delay=0, kw='maxUpO')
event_panel = column_look_up(tgt=event_panel, src=maxDown, delay=0, kw='maxDown')
event_panel = column_look_up(tgt=event_panel, src=maxDownO, delay=0, kw='maxDownO')

# """确定涨跌停时，可能遇到周末事件；发生为1，未发生为0，非交易日为nan"""
# event_panel['maxUpO'].value_counts()
# event_panel['maxUpO'].isna().sum()
# event_panel.shape

assert stk_ipo_date.index.value_counts().sort_values()[0] == 1
event_panel[['ipo_date', 'delist_date']] = event_panel.stockcode.apply(
    lambda x: stk_ipo_date.loc[x, 'ipo_date':'delist_date'])

# %% 最后一个停牌期
# alldays_d = pd.read_csv(conf['alldays_d'], header=None)
# a_list_suspendsymbol = pd.read_csv(conf['a_list_suspendsymbol'])
# df_template = pd.DataFrame(index=alldays_d.iloc[:, 0], columns=stk_ipo_date.index)
# tmp = a_list_suspendsymbol.pivot(index='tradingdate', values='tradingdate', columns='stockcode')
# tmp = tmp.reindex_like(df_template).fillna(method='ffill')
# tmp.to_csv(conf['date_last_suspend_alldays'])
# date_last_suspend_alldays = tmp
date_last_suspend_alldays = pd.read_csv(conf['date_last_suspend_alldays'], index_col=0, parse_dates=True, dtype=str)
event_panel = column_look_up(tgt=event_panel, src=date_last_suspend_alldays, delay=0, kw='lastST')

# Method 2:
# tmp = a_list_suspendsymbol.set_index('stockcode')['tradingdate']

# def f(s, tmp):
#     if s.stockcode in tmp.index:
#         df = tmp.loc[s.stockcode]
#         df = df[df <= s.tradingdate]
#         if len(df) < 1:
#             res = np.nan
#         elif isinstance(df, str):
#             res = df
#         else:
#             res = df.iloc[-1]
#     else:
#         res = np.nan
#     return res

# event_panel['lastST'] = event[['stockcode', 'tradingdate']].apply(lambda s: f(s,tmp), axis=1)


# %% 表：记录id，事件日+-120日（不够则空），复权收盘价收益率（当日比昨日）
gap_lhs = -120
gap_rhs = 20
event_adjacent_ar = pd.DataFrame(columns=event.id, index=range(gap_lhs, gap_rhs + 1, 1))
event_adjacent_r = pd.DataFrame(columns=event.id, index=range(gap_lhs, gap_rhs + 1, 1))
for irow in tqdm(range(len(event))):  # 16099个事件，规模极大
    row = event.iloc[irow, :]
    # break
    stk_id = row['stockcode']
    event_date = row['tradingdate']
    row_id = row['id']
    date0, l_gap = TD.tradedate_delta(event_date, gap_lhs)
    date1, r_gap = TD.tradedate_delta(event_date, gap_rhs)
    # if (instnum.loc[event_date, stk_id] > 5) or (date0 is None):
    #     # 计入新机构已有超过5加关注，或者事件日超出日期范围
    #     continue
    adjacent_ret = adjret.loc[date0:date1, stk_id:stk_id]
    adjacent_mret = adjret_mkt.loc[date0:date1]
    adjacent_ar = adjacent_ret - adjacent_mret.values.reshape(-1, 1)
    # adjacent_ar = adjret_ab.loc[date0:date1, stk_id:stk_id]  # 为何速度慢很多？？
    assert l_gap == -gap_lhs  # adjacent_ar.__len__() == l_gap + r_gap + 1
    # if r_gap < gap_rhs:
    #     adjacent_ret = adjacent_ret.append(pd.DataFrame([np.nan] * (gap_rhs - r_gap), index=range(r_gap, gap_rhs)))   
    #     adjacent_ar = adjacent_ar.append(pd.DataFrame([np.nan] * (gap_rhs - r_gap), index=range(r_gap, gap_rhs)))  
    # adjacent_ar.index = range(-l_gap, r_gap+1)
    # adjacent_ar.columns = [row_id]
    # adjacent_ret.index = range(-l_gap, r_gap+1)
    # adjacent_ret.columns = [row_id]
    # adjacent_ar.reindex_like(event_adjacent_returns.loc[:, row_id:row_id])
    event_adjacent_ar.loc[-l_gap:r_gap, row_id:row_id] = adjacent_ar.values
    event_adjacent_r.loc[-l_gap:r_gap, row_id:row_id] = adjacent_ret.values

for df, col_name in zip([event_adjacent_ar.copy(), event_adjacent_r.copy()], ['AR', 'R']):
    df = df.T
    df.columns = df.columns.to_series().apply(lambda x: f'{col_name}{x}'.replace('-', '_'))
    df = df.reset_index()
    event_panel = event_panel.merge(df, on='id', how='left')


# %% """计算 CAR CR"""
def cal_cols_CAR(event_panel, key='AR'):
    if f'C{key}_1' in event_panel.columns:
        raise IndexError(f'C{key}_X already in columns')
    df = event_panel.copy()
    tmp = event_panel.loc[:, f'{key}_120':f'{key}_1'].iloc[:, ::-1].cumsum(axis=1).iloc[:, ::-1]
    tmp.columns = [f'C{key}_{x}' for x in range(120, 0, -1)]
    df = pd.concat([df, tmp], axis=1)
    tmp = event_panel.loc[:, f'{key}1':f'{key}20'].cumsum(axis=1)
    tmp.columns = [f'C{key}{x}' for x in range(1, 21)]
    df = pd.concat([df, tmp], axis=1)
    return df


event_panel = cal_cols_CAR(event_panel, key='AR')
event_panel = cal_cols_CAR(event_panel, key='R')
event_panel.ipo_date = pd.to_datetime(event_panel.ipo_date)
event_panel.tradingdate = pd.to_datetime(event_panel.tradingdate)
event_panel.lastST = pd.to_datetime(event_panel.lastST)
tdays_d = pd.read_csv(conf['tdays_d'], header=None, index_col=0, parse_dates=True)
event_panel['isTradeDay'] = event_panel.tradingdate.apply(lambda x: x in tdays_d.index)


# %% 筛选
def mask_efficiency(mask):
    l = len(mask)
    l1 = mask.sum()
    print(f"{l - l1} excluded from {l} rows, left: {l1 / l * 100:.2f} %")


mask_ipo90 = (event_panel.tradingdate - event_panel.ipo_date).apply(lambda x: x.days > 90)  # 新上市90天以上
mask_efficiency(mask_ipo90)
mask_suspend = (event_panel.tradingdate - event_panel.lastST).apply(lambda x: x.days > 0)  # 停牌后7天以上
mask_efficiency(mask_suspend)
mask_maxup = event_panel.maxUp != 1  # 当天不能涨停
mask_efficiency(mask_maxup)
mask_isTD = event_panel.isTradeDay  # 事件不在交易日发生（共66个，不予考虑）
mask_efficiency(mask_isTD)
mask_special = (event_panel.stockcode != '000792.SZ')
mask_efficiency(mask_special)

mask_sum = mask_ipo90 & mask_suspend & mask_maxup & mask_isTD & mask_special
mask_efficiency(mask_sum)
event_panel['Tradeable'] = mask_sum
panel = event_panel[mask_sum].copy().reset_index()
print(panel.shape)
# 可交易事件数量（日度）
tmp = event_panel.groupby('tradingdate')['Tradeable'].sum()
tmp.name = 'TradeableCount'
tmp = tmp.reset_index()
event_panel = event_panel.merge(tmp, on='tradingdate', how='left')


# %% 二分组
def corr_in_panel(df: pd.DataFrame, tgt: str, src: list, minum=0, method='spearman', yby=False):
    """
    探索指标相关性，
    :param df: 子事件池，根据Tradeable筛选
    :param tgt: 目标column
    :param src: 其他column，可能与tgt相关
    :param minum: 只绘制相关度绝对值大于阈值的柱状图
    :param method: 计算相关性系数的方式，宜采用spearman秩相关
    :param yby: 是否逐年计算（一年一柱）
    :return: 相关性系数
    """
    from matplotlib import pyplot as plt

    df = df.set_index('tradingdate')[[tgt] + src].dropna(axis=0).astype(float)
    if yby:
        df = df[df.index < '2022-01-01']
        year = df.index.to_series().apply(lambda x: x.year)
        corr = df.groupby(year).corr(method=method)
        tmp = corr[tgt]
        tmp = tmp[tmp.abs() > minum]
        tmp.unstack().iloc[:, 1:].T.plot.bar(figsize=(20, 8), title='Correlation with ' + tgt)
        plt.show()
    else:
        corr = df.corr(method=method)
        tmp = corr[tgt].iloc[1:]
        tmp = tmp[tmp.abs() > minum]
        tmp.plot.bar(title='Correlation with ' + tgt)
        plt.show()
    return corr


def cal_col_event_count(panel):
    dummy_odd = panel['TradeableCount'] % 2 == 1
    int_odd = dummy_odd.astype(int)
    panel['odd_adj'] = panel['TradeableCount'] // 2 + (int_odd.cumsum() % 2) * dummy_odd
    panel['single'] = (panel['TradeableCount'] == 1)
    return panel


event_panel = cal_col_event_count(event_panel)


# panel = event_panel[mask_sum].copy().reset_index(drop=True)
# template = panel[['id', 'tradingdate', 'stockcode']].copy()
# tmp = panel.groupby('tradingdate')['tradingdate'].count()
# tmp.name = 'event_count'
# tmp = pd.DataFrame(tmp).reset_index()
# event_count = template.merge(tmp, on='tradingdate', how='left')
# event_count['odd'] = (event_count.event_count % 2 == 1)  # 日事件数是奇数
# event_count['oddrand'] = event_count.odd.astype(int)
# event_count['oddadj'] = event_count.event_count // 2 + (event_count.oddrand.cumsum() % 2) * event_count.odd
# event_count['single'] = (event_count.event_count == 1)
# print(event_count.head())


def get_group2d(idx, panel):
    idx_rank = panel.groupby('tradingdate')[idx].rank()
    tmp = (idx_rank > panel.odd_adj).astype(int)  # rank > oddadj, 即 idx(CAR_8) 更高
    tmp = pd.concat((1 - tmp, tmp), axis=1)
    tmp[panel.single] = 1  # 当天只有1个事件，则H组和L组都持有

    group2d = panel[['id', 'tradingdate', 'stockcode']].copy()
    group2d[['L_' + idx, 'H_' + idx]] = tmp

    return group2d


panel = event_panel[mask_sum].copy().reset_index(drop=True)

idx = 'CAR_8'
group2d = get_group2d(idx=idx, panel=panel)
event_panel = event_panel.merge(group2d[['id', f'L_{idx}', f'H_{idx}']], on='id', how='left')

idx = 'CAR_6'
group2d = get_group2d(idx=idx, panel=panel)
event_panel = event_panel.merge(group2d[['id', f'L_{idx}', f'H_{idx}']], on='id', how='left')

idx = 'AR0'
group2d = get_group2d(idx=idx, panel=panel)
event_panel = event_panel.merge(group2d[['id', f'L_{idx}', f'H_{idx}']], on='id', how='left')

idx = 'AR1'
group2d = get_group2d(idx=idx, panel=panel)
event_panel = event_panel.merge(group2d[['id', f'L_{idx}', f'H_{idx}']], on='id', how='left')

idx = 'AR2'
group2d = get_group2d(idx=idx, panel=panel)
event_panel = event_panel.merge(group2d[['id', f'L_{idx}', f'H_{idx}']], on='id', how='left')


# %% 分组信号
def _get_signal(panel, idx, dur):
    signal = panel.pivot(index='tradingdate', columns='stockcode', values=idx)
    signal = signal.replace(0, np.nan).fillna(method='ffill', limit=dur - 1).replace(np.nan, 0)
    return signal


def get_signal(idx, panel, csv_path, dur=3, descrip=np.nan):
    fname = f'first_report_{idx}_dur{dur}'
    signal = _get_signal(panel, idx, dur)
    signal.to_csv(csv_path + fname + '.csv')
    print(fname.replace('.csv', ''), end='\n')

    irow = {'IF_TEST': 1,
            'F_NAME': fname,
            'F_BEGIN': str(signal.index[0].date()),
            'F_END': str(signal.index[-1].date()),
            'UPDATE': datetime.today().strftime('%Y-%m-%d'),
            'DESCRIP': descrip}
    return irow


panel = event_panel[event_panel.Tradeable].copy()
panel['baseline'] = 1
csv_path = conf['factorscsv_path']
dur = 3
f_info_list = [get_signal('baseline', panel, csv_path, dur), get_signal('L_CAR_6', panel, csv_path, dur),
               get_signal('H_CAR_6', panel, csv_path, dur), get_signal('L_CAR_8', panel, csv_path, dur),
               get_signal('H_CAR_8', panel, csv_path, dur), get_signal('L_AR0', panel, csv_path, dur),
               get_signal('H_AR0', panel, csv_path, dur)]


# %% 回测信息
def write_f_info(excel_path, info_list):
    df = pd.read_excel(excel_path)
    df['IF_TEST'] = 0
    for irow in info_list:
        df = df.append(info_list, ignore_index=True)
    # 同名因子只保留UPDATE日期最新的
    mask = df.sort_values(['UPDATE', 'IF_TEST'], ascending=False)[['F_NAME']].drop_duplicates().index
    df = df.loc[mask].sort_index()
    df.astype('str').to_excel(excel_path, encoding='gbk', index=None)


write_f_info(conf['factors_tested'], f_info_list)

# %%
event_panel.to_pickle(conf['event_first_report3'])
print('pickle event panel saved in', conf['event_first_report3'])
