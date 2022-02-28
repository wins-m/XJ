"""
(created by swmao on Feb. 28th)
参考Wind-组合管理-调整持仓（EFR基准）-持仓文件导入-持仓权重模板

"""
import os

import pandas as pd
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")


def main():
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    csv_path = conf['factorscsv_path']
    data_path = conf['data_path']
    close_path = conf['daily_close']
    initial = False
    fbegin_date = None  # '2021-07-01'
    for src_file in ['first_report_baseline1.csv', 'first_report_H_AR0_L_CAR_8_dur3.csv']:
        weight_to_efr(src_file, csv_path, data_path, close_path, fbegin_date, initial)


def weight_to_efr(src_file: str, csv_path: str, data_path: str, close_path: str, fbegin_date=None, initial=True,):
    src_path = csv_path + src_file
    src = pd.read_csv(src_path, index_col=0, parse_dates=True)
    print(f'Read 2d: {src_path}')

    src = src.unstack().reset_index()
    src.columns = ['stockcode', 'tradingdate', 'weight']
    src = src[src.weight.abs() > 0]
    src = src.sort_values(['tradingdate', 'stockcode'])
    if fbegin_date is not None:
        src = src[src.tradingdate >= fbegin_date]
    close = pd.read_csv(close_path, index_col=0, parse_dates=True)
    src['price'] = src[['tradingdate', 'stockcode']].apply(lambda s: close.loc[s.tradingdate, s.stockcode], axis=1)
    src['wtf'] = ''

    if initial:
        print('Attention: Initial EFR weight 1,000,000')
        asset = src.iloc[0:1, :].copy()
        asset['stockcode'] = 'Asset'
        asset['weight'] = 1000000
        asset['price'] = 1
        src = pd.concat([asset, src])

    bd = src['tradingdate'].min().strftime('%Y_%m_%d')
    ed = src['tradingdate'].max().strftime('%Y_%m_%d')
    # src['tradingdate'] = src.tradingdate.apply(lambda x: x.strftime('%Y-%m-%d'))
    src = src[['tradingdate', 'stockcode', 'weight', 'price', 'wtf']]
    src = src.rename(columns={'tradingdate': '调整日期', 'stockcode': '证券代码', 'weight': '持仓权重', 'price': '成本价格', 'wtf': '是否融资融券'})

    tgt_file = 'EFR_' + src_file.replace('.csv', f'_{bd}_{ed}.xlsx')
    tgt_path = data_path + tgt_file
    tgt_path2 = csv_path + 'EFR_' + src_file
    src.to_excel(tgt_path, index=None)
    src.to_csv(tgt_path2, index=False)
    print(src)
    print(f'\n\nSaved in {tgt_path}, {tgt_path2}\n')


def get_weight_by_date(conf, begin_date , end_date):
    from supporter.mysql import conn_mysql, mysql_query
    from datetime import timedelta
    import numpy as np

    efr_tables = conf['tables']['efr_tables']
    update_target = [file.replace('.csv', '') for file in efr_tables[1:]]
    mysql_engine = conf['mysql_engine']
    engine_list = {engine_id: conn_mysql(engine_info) for engine_id, engine_info in mysql_engine.items()}

    target_table = update_target[0]
    # for target_table in update_target:
    query = f'SELECT 调整日期 FROM intern.{target_table} ORDER BY 调整日期 DESC LIMIT 1;'
    date_local = mysql_query(query, engine_list['engine2']).loc[0, '调整日期']

    query = f"""SELECT tradingdate,stockcode,fv FROM factordatabase.event_first_report WHERE tradingdate>'{date_local}' ORDER BY tradingdate;"""
    event_first_report = mysql_query(query, engine_list['engine2'])

    begin_date = event_first_report.tradingdate.min()
    end_date = event_first_report.tradingdate.max()
    print(date_local, begin_date, end_date)
    try:
        assert end_date > date_local  # 继续条件
    except AssertionError:
        raise AssertionError(f'数据库{target_table}最新日期与event_first_report一致，无需更新')

    # %% 准备所需数据
    prior_date = date_local - timedelta(30)
    query = f"""SELECT tradingdate,stockcode,close,close*adjfactor AS closeAdj FROM jeffdatabase.stk_marketdata WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    close_adj = mysql_query(query, engine_list['engine0'])

    prior_date = date_local
    query = f"""SELECT tradingdate,stockcode,west_instnum FROM jeffdatabase.stk_west_instnum_180 WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    instnum = mysql_query(query, engine_list['engine0'])

    prior_date = date_local - timedelta(95)
    query = f"""SELECT stockcode,ipo_date FROM jeffdatabase.stk_ipo_date WHERE ipo_date>'{prior_date}' ORDER BY ipo_date;"""
    ipo_date = mysql_query(query, engine_list['engine0'])

    prior_date = date_local
    query = f"""SELECT stockcode,tradingdate FROM jeffdatabase.a_list_suspendsymbol WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    a_list_suspend = mysql_query(query, engine_list['engine0'])

    prior_date = date_local
    query = f"""SELECT tradingdate,stockcode,maxupordown FROM jeffdatabase.stk_maxupordown WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    maxupordown = mysql_query(query, engine_list['engine0'])

    prior_date = date_local - timedelta(30)
    query = f"""SELECT tradingdate FROM jeffdatabase.tdays_d WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    tdays_d = mysql_query(query, engine_list['engine0'])

    # %% 数据入库
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

    event_panel = event_first_report.copy()

    close_2d = close_adj.pivot(index='tradingdate', columns='stockcode', values='close')
    event_panel = column_look_up(event_panel, close_2d, delay=0, kw='price', msg='未找对应收盘价（非交易日或今日）')

    close_adj_2d = close_adj.pivot(index='tradingdate', columns='stockcode', values='closeAdj')
    adjret = close_adj_2d.pct_change().iloc[1:]
    # adjret_mkt = pd.DataFrame(adjret.apply(lambda s: s.mean(), axis=1), columns=['mkt'])
    adjret_ab = adjret.apply(lambda s: s - s.mean(), axis=1)  # 异常大，特殊公司，此处保留异常
    adjret_car_8 = adjret.rolling(8).sum().shift(1)
    event_panel = column_look_up(event_panel, adjret_ab, delay=0, kw='AR0', msg='未找对应收盘价（非交易日或今日）')
    event_panel = column_look_up(event_panel, adjret_car_8, delay=0, kw='CAR_8', msg='未找对应收盘价（非交易日或今日）')

    instnum_2d = instnum.pivot(index='tradingdate', columns='stockcode', values='west_instnum')
    event_panel = column_look_up(event_panel, instnum_2d, delay=0, kw='instnum', msg='未找到对应instnum180')

    ipo_date_sr = ipo_date.set_index('stockcode')['ipo_date']
    event_ipo_date = event_panel['stockcode'].apply(lambda x: ipo_date_sr.loc[x] if x in ipo_date_sr.index else pd.to_datetime('2000-01-01'))
    event_panel['new_ipo'] = (pd.to_datetime(event_panel.tradingdate) - pd.to_datetime(event_ipo_date)).apply(lambda x: x.days <= 90)

    a_list_suspend_2d = a_list_suspend.copy()
    a_list_suspend_2d['suspend'] = 1
    a_list_suspend_2d = a_list_suspend_2d.pivot(index='tradingdate', columns='stockcode', values='suspend')
    a_list_suspend_2d = a_list_suspend_2d.reindex_like(adjret).fillna(0)
    event_panel = column_look_up(event_panel, a_list_suspend_2d, delay=0, kw='suspend', msg='未找到对应a_list_suspend')

    maxupordown['up'] = (maxupordown.maxupordown == 1).astype(int)
    maxupordown_2d = maxupordown.pivot(index='tradingdate', columns='stockcode', values='up')
    maxupordown_2d = maxupordown_2d.fillna(0)
    event_panel = column_look_up(event_panel, maxupordown_2d, delay=0, kw='up', msg='未找到对应 maxupordown')

    event_panel['isTradeday'] = event_panel.tradingdate.apply(lambda x: pd.to_datetime(x) in tdays_d.tradingdate.values).astype(int)

    # %%
    event_panel.to_pickle(conf['data_path'] + 'event_panel(update_tmp_.pkl')

    # %% oddadj, single, H_*, L_*

    # %% 2d signal -> dur3 -> 2d weight -> EFR


# %%
if __name__ == '__main__':
    main()
