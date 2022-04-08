"""# `get_data.py`
(created by swmao on Jan. 11th)
下载单项指标（主要为因子）的面板数据，long转wide格式存入本地；
- 所用表格在[access_target](./data/access_target.xlsx)中指定
(update Jan. 25th)
xlsx中指定起止日期
(update Feb. 24th)
新加服务器alpha_001

"""
import pandas as pd
# import numpy as np
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.mysql import conn_mysql, mysql_query


def save_marketdata(begin_date, end_date, engine, data_path):
    """按日存市场行情数据 %Y-%m-%d.h5"""
    pass
    """
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


def save_factor_panel(grid: pd.DataFrame, engine_list, data_path):
    """对name_cols中的表格（以IND,COL,VAL二维获取），取到本地"""
    for i_row in grid.iterrows():
        tb = i_row[1]
        query = f"SELECT {tb['IND']},{tb['COL']},{tb['VAL']}" \
                f" FROM {tb['TABLE']}" \
                f" WHERE {tb['IND']}>='{tb['B_DATE']}' AND {tb['IND']}<='{tb['E_DATE']}'" \
                f"{' AND '+tb['WHERE'] if isinstance(tb['WHERE'], str) else ''}" \
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


def transfer_data(mysql_engine, data_path, access_target, force_update=False):
    """准备访问服务器，获取数据到本地"""
    grid = pd.read_excel(access_target)
    grid = grid if force_update else grid[grid['UPDATE'] == 1]

    engine_list = []
    for e in mysql_engine.keys():  # ['engine0', 'engine1', 'engine2', 'engine4']:
        engine_info = mysql_engine[e]
        engine_list.append(conn_mysql(engine_info))

    print(mysql_query("SELECT tradingdate FROM jeffdatabase.tdays_d ORDER BY tradingdate DESC LIMIT 1", engine_list[0]))

    save_factor_panel(grid, engine_list, data_path)
    # engine = engine_list[0]


def get_data(conf):
    """main"""
    mysql_engine = conf['mysql_engine']
    force_update = conf['force_update']
    # begin_date = conf['begin_date']
    # end_date = conf['end_date']
    data_path = conf['data_path']
    access_target = conf['access_target']
    # transfer_data(mysql_engine, begin_date, end_date, data_path, access_target, force_update)
    transfer_data(mysql_engine, data_path, access_target, force_update)


if __name__ == '__main__':
    import yaml
    
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    #
    get_data(conf)
