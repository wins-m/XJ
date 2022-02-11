"""
(created by swmao on Jan. 18th)
- 获取可否交易的标签，去除新上市、停牌、涨跌停
(updated Jan. 25th)
- 用行情数据，识别开盘涨停，只去除开盘涨停
- 多个csv：多个筛选条件
(modified Feb. 10th)
- 以收盘加交易，当日不可实现是因为昨日停牌/涨跌停，因此shift(1)

"""
from datetime import timedelta
from tqdm import tqdm
import pandas as pd


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
    ind = pd.read_csv(conf['tdays_d'], header=None, index_col=0, parse_dates=True)  # 交易日2000-2021手动数据库
    # .loc['2010-01-01':'2021-12-31']
    df = pd.DataFrame(index=ind.index.rename(''), columns=col).fillna(False)
    for irow in tqdm(stk_ipo_date.iterrows()):
        stk = irow[1]['stockcode']
        d0 = str(irow[1]['ipo_date'])
        d1 = str(irow[1]['delist_date'])
        d1 = '2022-12-31' if (d1 == 'nan') else d1
        df.loc[d0:d1, stk:stk] = True
    print('上市后+退市前保存于 %s, key=%s' % (conf['a_list_tradeable'], 'ipo'))
    df.to_hdf(conf['a_list_tradeable'], key='ipo')
    ipo = df.copy()

    # 新股上市{conf['ipo_delay']}内不交易
    for irow in tqdm(stk_ipo_date.iterrows()):
        stk = irow[1]['stockcode']
        d0 = pd.to_datetime(irow[1]['ipo_date'])
        d0id = len(ind)-1
        d1 = d0
        try:
            d0id = ind.index.get_loc(d0)
            d1 = ind.index[d0id + conf['ipo_delay']]
        except KeyError:
            d1 = d0 + timedelta(conf['ipo_delay'] * 1.5)  # ipo日在2000年以前，则加上1.5*ipo_delay个自然日
        except IndexError:
            d1 = ind.index[min(d0id + conf['ipo_delay'], len(ind)-1)]
        finally:
            df.loc[d0:d1, stk:stk] = False  # 新股改False
    print('上市%s日后+退市前保存于 %s, key=%s' % (conf['ipo_delay'], conf['a_list_tradeable'], f"ipo{conf['ipo_delay']}"))
    df.to_hdf(conf['a_list_tradeable'], key=f"ipo{conf['ipo_delay']}")
    ipo_delay = df.copy()

    # 停牌，设为False
    a_list_suspendsymbol = pd.read_csv(conf['a_list_suspendsymbol'], index_col=0, parse_dates=True)
    a_list_suspendsymbol['suspend'] = False
    suspend = a_list_suspendsymbol.reset_index().pivot(index='tradingdate', columns='stockcode', values='suspend')
    suspend = suspend.reindex_like(df).fillna(True)  # 空，非停牌，填充True
    print('停牌为False保存于 %s, key=%s' % (conf['a_list_tradeable'], 'suspend'))
    suspend.to_hdf(conf['a_list_tradeable'], key='suspend')

    # 涨跌停，当天为False
    stk_maxupordown = pd.read_csv(conf['stk_maxupordown'], index_col=0, parse_dates=True)
    updown = stk_maxupordown.reset_index().pivot(index='tradingdate', columns='stockcode', values='maxupordown')
    updown = updown.reindex_like(df).isna()
    print('涨跌停为False否则为True 存于 %s, key=%s' % (conf['a_list_tradeable'], 'updown'))
    updown.to_hdf(conf['a_list_tradeable'], key='updown')
    is_maxupordown = ~updown
    # 开盘涨跌停，开盘价=收盘价
    daily_open = pd.read_csv(conf['daily_open'], index_col=0, parse_dates=True)
    daily_close = pd.read_csv(conf['daily_close'], index_col=0, parse_dates=True)
    eq = (daily_open == daily_close).reindex_like(is_maxupordown)
    updown_open = ~(is_maxupordown & eq)  # 非（涨跌停 且 开=收），即非“开盘即涨跌停”
    print('开盘一字涨跌停为False其余为True 存于 %s, key=%s' % (conf['a_list_tradeable'], 'updown_open'))
    updown_open.to_hdf(conf['a_list_tradeable'], key='updown_open')

    print('ipo%s & suspend & updown_open 保存于 %s, key=%s' % (conf['ipo_delay'], conf['a_list_tradeable'], 'updown_open'))
    tradeable = ipo_delay & suspend.shift(1) & updown_open.shift(1)  # 以收盘加交易，当日不可实现是因为昨日停牌/涨跌停
    tradeable.to_hdf(conf['a_list_tradeable'], key='tradeable')
    tradeable1 = ipo_delay & suspend.shift(1)
    tradeable1.to_hdf(conf['a_list_tradeable'], key='tradeable_withupdown')
    tradeable2 = ipo_delay & suspend.shift(1) & updown.shift(1)
    tradeable2.to_hdf(conf['a_list_tradeable'], key='tradeable_noupdown')
    print(f"""PANEL {tradeable.shape} SAVE IN {conf['a_list_tradeable']}""")


if __name__ == '__main__':
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    update_tradeable_label(conf)
