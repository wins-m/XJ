"""
(created by swmao on Jan. 18th)
- 获取可否交易的标签，去除新上市、停牌、涨跌停
(updated Jan. 25th)
- 用行情数据，识别开盘涨停，只去除开盘涨停
- 多个csv：多个筛选条件
(modified Feb. 10th)
- 以收盘加交易，当日不可实现是因为昨日停牌/涨跌停，因此shift(1)
- 仅仅作用于 key= tradeable, tradeable_withupdown, tradeable_noupdown
- 将deprecate

TODO: 自动更新 stk_ipo_date, tdays_d, a_list_suspendsymbol, stk_maxupordown
"""
from datetime import timedelta
from tqdm import tqdm
import pandas as pd


def main():
    # %%
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    update_tradeable_label(conf)


def update_tradeable_label(conf):
    """
    获取可否交易的标签，格式为：
        |          | 000001.SZ | 000002.SZ | 000004.SZ |
        | -------- | --------- | --------- | --------- |
        | 2000/1/4 | TRUE      | TRUE      | TRUE      |
        | 2000/1/5 | TRUE      | TRUE      | TRUE      |
        | 2000/1/6 | TRUE      | TRUE      | TRUE      |
        | 2000/1/7 | TRUE      | TRUE      | TRUE      |

    可交易的股票设定如下
    - columns为sto_ipo_date中的stockcode.unique()
    - 上市满conf['ipo_delay']个自然日，未退市，根据`stk_ipo_date`下载到local
    - 去除停牌，根据`a_list_suspendsymbol`下载到local
    - 去除涨跌停，根据`stk_maxupordown` WHERE maxupordown!=0下载到local

    存到本地conf['tradeable_multiplier']，即tradeable_multiplier.csv
    - 最后更新：Jan. 18th    PANEL (5342, 4741)

    """
    # stockcode@tradingdate 填充以False
    stk_ipo_date = pd.read_csv(conf['stk_ipo_date'])
    col = stk_ipo_date['stockcode'].sort_values().unique()  # 有stk_ipo_date记录的个股
    print('模板内个股总数: %d' % len(col))
    tds = pd.read_csv(conf['tdays_d'], header=None, index_col=0, parse_dates=True)  # 交易日2000-2021手动数据库
    ind = tds.index.to_list()  # .loc['2010-01-01':'2021-12-31']
    df_template = pd.DataFrame(index=ind, columns=col).fillna(False)

    # 基础面板，根据tradingdates(tdays_d)和stk_ipo(stk_ipo_date)
    ipo = df_template.copy()
    for irow in tqdm(stk_ipo_date.iterrows()):
        stk = irow[1]['stockcode']
        d0 = str(irow[1]['ipo_date'])
        d1 = str(irow[1]['delist_date'])
        d1 = '2099-12-31' if (d1 == 'nan') else d1
        ipo.loc[d0: d1, stk: stk] = True
    print('上市后+退市前保存于 %s, key=%s' % (conf['a_list_tradeable'], 'ipo'))
    ipo.to_hdf(conf['a_list_tradeable'], key='ipo')

    # 新股上市{conf['ipo_delay']}内不交易
    ipo_delay = ipo.copy()
    for irow in tqdm(stk_ipo_date.iterrows()):
        stk = irow[1]['stockcode']
        d0 = pd.to_datetime(irow[1]['ipo_date'])
        d0id = len(tds)-1
        d1 = d0
        try:
            d0id = tds.index.get_loc(d0)
            d1 = tds.index[d0id + conf['ipo_delay']]
        except KeyError:
            d1 = d0 + timedelta(conf['ipo_delay'] * 1.5)  # ipo日在2000年以前，则加上1.5*ipo_delay个自然日
        except IndexError:
            d1 = tds.index[min(d0id + conf['ipo_delay'], len(ind)-1)]
        finally:
            ipo_delay.loc[d0:d1, stk:stk] = False  # 新股改False
    print('上市%s日后+退市前保存于 %s, key=%s' % (conf['ipo_delay'], conf['a_list_tradeable'], f"ipo{conf['ipo_delay']}"))
    ipo_delay.to_hdf(conf['a_list_tradeable'], key=f"ipo{conf['ipo_delay']}")  # key=ipo60

    # 停牌，设为False
    a_list_suspendsymbol = pd.read_csv(conf['a_list_suspendsymbol'], index_col=0, parse_dates=True)
    a_list_suspendsymbol['suspend'] = False
    suspend = a_list_suspendsymbol.reset_index().pivot(index='tradingdate', columns='stockcode', values='suspend')
    suspend = suspend.reindex_like(df_template).fillna(True)  # 空，非停牌，填充True
    print('停牌为False保存于 %s, key=%s' % (conf['a_list_tradeable'], 'suspend'))
    suspend.to_hdf(conf['a_list_tradeable'], key='suspend')

    # 涨跌停，当天为False
    stk_maxupordown = pd.read_csv(conf['stk_maxupordown'], index_col=0, parse_dates=True)
    stk_maxupordown = stk_maxupordown.reset_index().pivot(index='tradingdate', columns='stockcode', values='maxupordown')
    stk_maxupordown = stk_maxupordown.reindex_like(df_template)
    max_up = pd.DataFrame(~(stk_maxupordown == 1))
    max_down = pd.DataFrame(~(stk_maxupordown == -1))
    updown = ~stk_maxupordown.isna()
    print('涨跌停为False否则为True 存于 %s, key=%s' % (conf['a_list_tradeable'], 'updown & up & down'))
    updown.to_hdf(conf['a_list_tradeable'], key='updown')
    max_up.to_hdf(conf['a_list_tradeable'], key='up')
    max_down.to_hdf(conf['a_list_tradeable'], key='down')

    # 开盘涨跌停，开盘价=收盘价
    daily_open = pd.read_csv(conf['daily_open'], index_col=0, parse_dates=True)
    daily_close = pd.read_csv(conf['daily_close'], index_col=0, parse_dates=True)
    eq = (daily_open == daily_close).reindex_like(df_template)

    updown_open = ~(updown & eq)  # 非（涨跌停 且 开=收），即非“一字涨跌停”
    up_open = ~(~max_up & eq)  # 非 (涨停 且 开=收)，即一字涨停
    down_open = ~(~max_down & eq)  # 非 (跌停 且 开=收)，即一字跌停
    print('开盘一字涨跌停为False其余为True 存于 %s, key=%s' % (conf['a_list_tradeable'], 'updown_open & up_open & down_open'))
    updown_open.to_hdf(conf['a_list_tradeable'], key='updown_open')
    up_open.to_hdf(conf['a_list_tradeable'], key='up_open')
    down_open.to_hdf(conf['a_list_tradeable'], key='down_open')

    print('ipo%s & suspend_L1 & updown_open_L1 保存于 %s, key=%s' % (conf['ipo_delay'], conf['a_list_tradeable'], 'tradeable'))
    tradeable = ipo_delay & suspend.shift(1) & updown_open.shift(1)  # 以收盘加交易，当日不可实现是因为昨日停牌/涨跌停
    tradeable.to_hdf(conf['a_list_tradeable'], key='tradeable')

    print('ipo%s & suspend_L1 保存于 %s, key=%s' % (conf['ipo_delay'], conf['a_list_tradeable'], 'tradeable_withupdown'))
    tradeable1 = ipo_delay & suspend.shift(1)
    tradeable1.to_hdf(conf['a_list_tradeable'], key='tradeable_withupdown')

    print('ipo%s & suspend_L1 & updown_L1 保存于 %s, key=%s' % (conf['ipo_delay'], conf['a_list_tradeable'], 'tradeable_noupdown'))
    tradeable2 = ipo_delay & suspend.shift(1) & updown.shift(1)
    tradeable2.to_hdf(conf['a_list_tradeable'], key='tradeable_noupdown')


# %%
if __name__ == '__main__':
    main()
