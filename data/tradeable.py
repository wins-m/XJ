"""
(created by swmao on Jan. 18th)
获取可否交易的标签，去除新上市、停牌、涨跌停

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
    col = stk_ipo_date['stockcode'].sort_values().unique()
    ind = pd.read_csv(conf['tdays_d'], header=None, index_col=0, parse_dates=True)
    # .loc[conf['begin_date']:conf['end_date']]
    df = pd.DataFrame(index=ind.index.rename(''), columns=col).fillna(False)
    print(f"""个股上市日期，（{conf['ipo_delay']}）天后开始设置为True""")
    begin_date = df.index[0]
    end_date = df.index[-1]
    for irow in tqdm(stk_ipo_date.iterrows()):
        stk = irow[1]['stockcode']
        d0 = pd.to_datetime(irow[1]['ipo_date'])
        d1 = pd.to_datetime(d0) + timedelta(days=conf['ipo_delay'])
        d1 = d1 if d1 > begin_date else begin_date
        d2 = str(irow[1]['delist_date'])
        d2 = end_date if (d2 == 'nan' or end_date < pd.to_datetime(d2)) else pd.to_datetime(d2)
        df.loc[d1:d2, stk:stk] = True

    print('停牌，设为False')
    a_list_suspendsymbol = pd.read_csv(conf['a_list_suspendsymbol'], index_col=0, parse_dates=True)
    # for irow in tqdm(a_list_suspendsymbol.iterrows()):
    #     td = irow[0]
    #     stk = irow[1]['stockcode']
    #     df.loc[td:td, stk:stk] = False
    a_list_suspendsymbol['suspend'] = False
    tmp = a_list_suspendsymbol.reset_index().pivot(index='tradingdate', columns='stockcode', values='suspend')
    tmp = tmp.reindex_like(df).fillna(True)  # 空，非停牌，填充True
    df = df * tmp

    print('涨跌停，当天为False')
    stk_maxupordown = pd.read_csv(conf['stk_maxupordown'], index_col=0, parse_dates=True)
    # for irow in tqdm(stk_maxupordown.iterrows()):
    #     if irow[1]['maxupordown'] in [1, -1]:
    #         td = irow[0]
    #         stk = irow[1]['stockcode']
    #         df.loc[td:td, stk:stk] = False
    tmp = stk_maxupordown.reset_index().pivot(index='tradingdate', columns='stockcode', values='maxupordown')
    tmp = tmp.reindex_like(df).isna()  # 空，意味着非涨跌停
    df = df * tmp

    print('存一份到本地（1-True, 0-False）')
    # df.astype(int).to_csv(conf['a_list_tradeable'].replace('hdf', 'csv'))
    df.to_hdf(conf['a_list_tradeable'], key='tradeable')
    print(f"""PANEL {df.shape} SAVE IN {conf['a_list_tradeable']}""")


if __name__ == '__main__':
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    update_tradeable_label(conf)
