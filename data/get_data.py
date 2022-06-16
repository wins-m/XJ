"""# `get_data.py`
(created by swmao on Jan. 11th)
下载单项指标（主要为因子）的面板数据，long转wide格式存入本地；
- 所用表格在[access_target](./data/access_target.xlsx)中指定
(update Jan. 25th)
xlsx中指定起止日期
(update Feb. 24th)
新加服务器alpha_001
(update May 21st)
下载指数成分股权重时用流通市值填充fv

"""
import pandas as pd
# import numpy as np
import sys

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.mysql import conn_mysql, mysql_query


def check_idx_constituent(conf):
    market_type, idx_code = 'CSI500', '000905.SH'
    idx_cons_path = conf['idx_constituent'].format(market_type)
    tmp = idx_cons_path.rsplit('/', maxsplit=1)
    idx_cons_path0 = tmp[0] + '/(depreciated)' + tmp[1]
    idx_close_path = conf['idx_marketdata_close']
    # stk_close_path = conf['closeAdj']
    stk_close_path = conf['daily_close']

    idx_cons = pd.read_csv(idx_cons_path, index_col=0, parse_dates=True)
    idx_cons0 = pd.read_csv(idx_cons_path0, index_col=0, parse_dates=True)
    idx_cons /= 100
    idx_cons0 /= 100

    idx_close = pd.read_csv(idx_close_path, index_col=0, parse_dates=True)  # adjusted ?
    csi500_rtn = idx_close[idx_code].pct_change().iloc[1:].rename('csi500')

    stk_close = pd.read_csv(stk_close_path, index_col=0, parse_dates=True)
    stk_rtn = stk_close.pct_change().iloc[1:]
    csi500_rtn0 = (idx_cons0 * stk_rtn.reindex_like(idx_cons0)).sum(axis=1).rename('Wind500')
    csi500_rtn1 = (idx_cons * stk_rtn.reindex_like(idx_cons)).sum(axis=1).rename('freeMV500')

    rtn_compare = pd.concat([csi500_rtn0, csi500_rtn1, csi500_rtn], axis=1).dropna()

    from matplotlib import pyplot as plt
    cumsum_compare = rtn_compare.cumsum().add(1); cumsum_compare.plot();  plt.show()
    (cumsum_compare[['Wind500', 'freeMV500']] - cumsum_compare[['csi500']].values).plot(); plt.show()
    (idx_cons - idx_cons0).abs().max(axis=1).plot(); plt.show()


def save_factor_panel(grid: pd.DataFrame, engine_list, data_path):
    """对name_cols中的表格（以IND,COL,VAL二维获取），取到本地"""
    print(f'\nDownload remote tables, grid size={len(grid)} ...')
    for i_row in grid.iterrows():
        tb = i_row[1]
        query = f"SELECT {tb['IND']},{tb['COL']},{tb['VAL']}" \
                f" FROM {tb['TABLE']}" \
                f" WHERE {tb['IND']}>='{tb['B_DATE']}' AND {tb['IND']}<='{tb['E_DATE']}'" \
                f"{' AND ' + tb['WHERE'] if isinstance(tb['WHERE'], str) else ''}" \
                f" ORDER BY {tb['IND']};"  # f" FROM {tb['BASE']}.{tb['TABLE']}" \
        print(query)  # query sentence
        engine = engine_list[tb['SERVER']]
        # request for 2D data
        try:
            df = mysql_query(query, engine)
        except:
            print('FAIL ACCESS', query)
            continue
        # reshape and save 2D data
        find_duplicated = False
        panel = pd.DataFrame()
        try:
            val_col = tb['VAL'].split('AS')[1].strip() if 'AS' in tb['VAL'] else tb['VAL']
            panel = df.pivot(index=tb['IND'], columns=tb['COL'], values=val_col)
        except ValueError:
            print('[Error] previous query failed, consider duplicated index')
            panel = df.groupby(
                [tb['IND'], tb['COL']])[tb['VAL']].mean().reset_index().pivot(
                index=tb['IND'], columns=tb['COL'], values=tb['VAL'])
            find_duplicated = True
        finally:
            panel.to_csv(data_path + tb['CSV'])
            if find_duplicated:
                cnt_panel = df.groupby(
                    ['update_date', tb['COL']]).count().reset_index().pivot(
                    index=tb['IND'], columns=tb['COL'], values=tb['VAL'])
                cnt_panel.to_csv(data_path + tb['CSV'].replace('.csv', '_count.csv'))
    else:
        print("Local Table Updated.")


def adjust_idx_constituent_weight(mv_path: str, idx_constituent_path: list):
    """Fill idx_constituent weight with free market share"""
    print(f'\nAdjust idx constituent, {len(idx_constituent_path)} files ...')
    mv = pd.read_csv(mv_path, index_col=0, parse_dates=True)
    for ind_cons_path in idx_constituent_path:
        idx_w = pd.read_csv(ind_cons_path, index_col=0, parse_dates=True)
        idx_w = idx_w.loc[idx_w.index.intersection(mv.index)]
        idx_w = idx_w * mv.reindex_like(idx_w)
        # mkt_freeshares could still be missing
        tmp = idx_w.count(axis=1).value_counts()
        print(f" {tmp.iloc[1:].sum() / tmp.sum() * 100:.2f}% insufficient days: {ind_cons_path.rsplit('/', maxsplit=1)[-1]}")
        idx_w = idx_w.apply(lambda s: s / s.sum(), axis=1)
        idx_w *= 100
        idx_w.to_csv(ind_cons_path)
    print('Finished.')


def transfer_data(mysql_engine, data_path, access_target, force_update=False):
    """准备访问服务器，获取数据到本地"""
    grid = pd.read_excel(access_target, dtype={
        'UPDATE': int, 'SERVER': int, 'BASE': str, 'TABLE': str, 'IND': str,
        'B_DATE': str, 'E_DATE': str, 'COL': str, 'VAL': str, 'WHERE': str, 'CSV': str
    })
    # grid['B_DATE'] = pd.to_datetime(grid['B_DATE'])
    # grid['E_DATE'] = pd.to_datetime(grid['E_DATE'])
    grid_mv = grid[grid['CSV'] == 'stk_marketvalue.csv']
    grid = grid if force_update else grid[grid['UPDATE'] == 1]
    if len(grid) == 0:
        print(f'set UPDATE=1 in `{access_target}`')
        return
    print(grid)

    # sql engines initialize here
    engine_list = []
    for e in mysql_engine.keys():  # ['engine0', 'engine1', 'engine2', 'engine4']:
        engine_info = mysql_engine[e]
        engine_list.append(conn_mysql(engine_info))

    # test connection
    cd = mysql_query("SELECT tradingdate FROM jeffdatabase.tdays_d ORDER BY tradingdate DESC LIMIT 1", engine_list[0])
    print('current_date', cd.values[0, 0])

    # update tables
    save_factor_panel(grid, engine_list, data_path)

    # force adjust idx_constituent table
    mask = grid['CSV'].apply(lambda x: 'idx_constituent' in x and 'depreciated' not in x)
    # mask = (grid['TABLE'] == 'idx_constituent') & (grid['VAL'] == '1 AS fv')
    if mask.sum() > 0:
        if 'stk_marketvalue' not in grid['TABLE'].values:  # force update free market share
            save_factor_panel(grid_mv, engine_list, data_path)
        mv_path = data_path + grid_mv['CSV'].iloc[0]
        idx_constituent_path = [data_path + _ for _ in grid[mask]['CSV']]
        adjust_idx_constituent_weight(mv_path, idx_constituent_path)


def get_data(conf):
    """main"""
    mysql_engine = conf['mysql_engine']
    force_update = conf['force_update']
    data_path = conf['data_path']
    access_target = conf['access_target']

    transfer_data(mysql_engine, data_path, access_target, force_update)


def main():
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    get_data(conf)
    # check_idx_constituent(conf)


if __name__ == '__main__':
    main()


"""depreciated
def save_marketdata(begin_date, end_date, engine, data_path):
    ""按日存市场行情数据 %Y-%m-%d.h5""
    pass
    import os
    save_path = data_path + 'stk_marketdata/'
    os.makedirs(save_path, exist_ok=True)
    filename_local = sorted(os.listdir(save_path)).pop().replace('.h5', '')
    sql0 = f"SELECT DISTINCT tradingdate FROM stk_marketdata " \
           f"WHERE tradingdate>={max(begin_date, filename_local)} " \
           f"AND tradingdate<={end_date};"
    tradedates = mysql_query(query=sql0, engine=engine)
    for td in tradedates:
        pass
"""