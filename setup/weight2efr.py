"""
(created by swmao on Feb. 28th)
自动更新EFR
参考Wind-组合管理-调整持仓（EFR基准）-持仓文件导入-持仓权重模板 上传数据库 由数据库获取
0. 数据库中，由event_first_report维护event_first_report_selected，多溯2天(duration=3)
1. tradingdate, stockcode, fv, price, AR0, CAR_8, instnum, suspend, new_ipo, up, isTradeday
2. isTradeday = False (?) [by default, drop it]

"""
import pandas as pd
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.mysql import conn_mysql, mysql_query, sql_delete
from datetime import timedelta
from sqlalchemy.dialects.mysql import DATE, VARCHAR, DOUBLE


def main():
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    # transfer_holding_weights(conf)
    update_efr_weight(conf)
    # get_recent_efr_weight(conf)


def transfer_holding_weights(conf: dict):
    """(deprecated) Transfer 2D src_files to daily positions"""
    csv_path = conf['factorscsv_path']
    data_path = conf['data_path']
    close_path = conf['daily_close']
    initial = False
    fbegin_date = None  # '2021-07-01'

    close_price = pd.read_csv(close_path, index_col=0, parse_dates=True)

    # src_file = 'first_report_baseline1.csv'
    for src_file in ['first_report_baseline1.csv', 'first_report_H_AR0_L_CAR_8_dur3.csv']:
        src_path = csv_path + src_file
        src = pd.read_csv(src_path, index_col=0, parse_dates=True)
        if fbegin_date is not None:
            src = src[src.tradingdate >= fbegin_date]
        print(f'Read 2d: {src_path}')

        src = weight_to_efr(src, close_price, initial)

        bd = src['tradingdate'].min().strftime('%Y_%m_%d')
        ed = src['tradingdate'].max().strftime('%Y_%m_%d')
        tgt_file = 'EFR_' + src_file.replace('.csv', f'_{bd}_{ed}.xlsx')
        tgt_path = data_path + tgt_file
        tgt_path2 = csv_path + 'EFR_' + src_file
        src.to_excel(tgt_path, index=None)
        src.to_csv(tgt_path2, index=False)
        print(src)
        print(f'\n\nSaved in {tgt_path}, {tgt_path2}\n')


def weight_to_efr(src: pd.DataFrame, close_price: pd.DataFrame, initial=False) -> pd.DataFrame:
    """
    Transform 2D holding weight to stacked daily positions
    :param src: a 2D data frame of holding weight, row sum equals 1
    :param close_price: a 2D data frame of daily close weight (unadjusted) for purchase price
    :param initial: a bool adding first row as cash account if true
    :return: a stacked frame of daily positions infomation
    """
    src = src.unstack().reset_index()
    src.columns = ['stockcode', 'tradingdate', 'weight']
    src = src[src.weight.abs() > 0]
    src = src.sort_values(['tradingdate', 'stockcode'])
    src['price'] = src[['tradingdate', 'stockcode']].apply(lambda s: close_price.loc[s.tradingdate, s.stockcode], axis=1)
    src['wtf'] = ''

    if initial:
        print('Attention: Initial EFR weight 1,000,000')
        asset = src.iloc[0:1, :].copy()
        asset['stockcode'] = 'Asset'
        asset['weight'] = 1000000
        asset['price'] = 1
        src = pd.concat([asset, src])

    # src['tradingdate'] = src.tradingdate.apply(lambda x: x.strftime('%Y-%m-%d'))
    src = src[['tradingdate', 'stockcode', 'weight', 'price', 'wtf']]
    src = src.rename(columns={'tradingdate': '调整日期', 'stockcode': '证券代码', 'weight': '持仓权重', 'price': '成本价格', 'wtf': '是否融资融券'})

    return src


def update_efr_weight(conf):
    """Compare last date between event_first_report and efr_first_report* and update latter."""
    # efr_tables = conf['tables']['efr_tables']
    # update_target = [file.replace('.csv', '') for file in efr_tables[1:]]
    mysql_engine = conf['mysql_engine']
    engine_list = {engine_id: conn_mysql(engine_info) for engine_id, engine_info in mysql_engine.items()}

    # 确定需要更新的EFR范围：efr_baseline_dur3最后一个日期
    target_table = 'efr_baseline_dur3'  # update_target[-1]
    query = f'SELECT 调整日期 FROM intern.{target_table} ORDER BY 调整日期 DESC LIMIT 1;'
    date_local = mysql_query(query, engine_list['engine4'])
    if len(date_local) > 0:
        date_local = date_local.loc[0, '调整日期']
    else:
        import datetime
        date_local = datetime.date(2016, 1, 1)

    date_former = date_local - timedelta(5)
    query = f"""SELECT tradingdate,stockcode,fv FROM factordatabase.event_first_report WHERE tradingdate>'{date_former}' ORDER BY tradingdate;"""
    event_first_report = mysql_query(query, engine_list['engine2'])
    # if len(event_first_report) == 0:
    #     print(f'数据库{target_table}最新日期与event_first_report一致，无需更新')
    #     return
    begin_date = event_first_report.tradingdate.min()
    end_date = event_first_report.tradingdate.max()
    print(f'{target_table}已有最后日期:\t{date_local}')
    print(f'强制更新范围:\t{begin_date} ~ {end_date}')
    # try:
    #     assert end_date > date_local  # 继续条件
    # except AssertionError:
    #     raise AssertionError(f'数据库{target_table}最新日期与event_first_report一致，无需更新')

    # 准备所需数据
    prior_date = date_former - timedelta(30)
    query = f"""SELECT tradingdate,stockcode,close,close*adjfactor AS closeAdj FROM jeffdatabase.stk_marketdata WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    close_adj = mysql_query(query, engine_list['engine0'])

    prior_date = date_former
    query = f"""SELECT tradingdate,stockcode,west_instnum FROM jeffdatabase.stk_west_instnum_180 WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    instnum = mysql_query(query, engine_list['engine0'])

    prior_date = date_former - timedelta(95)
    query = f"""SELECT stockcode,ipo_date FROM jeffdatabase.stk_ipo_date WHERE ipo_date>'{prior_date}' ORDER BY ipo_date;"""
    ipo_date = mysql_query(query, engine_list['engine0'])

    prior_date = date_former
    query = f"""SELECT stockcode,tradingdate FROM jeffdatabase.a_list_suspendsymbol WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    a_list_suspend = mysql_query(query, engine_list['engine0'])

    prior_date = date_former
    query = f"""SELECT tradingdate,stockcode,maxupordown FROM jeffdatabase.stk_maxupordown WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    maxupordown = mysql_query(query, engine_list['engine0'])

    prior_date = date_former - timedelta(30)
    query = f"""SELECT tradingdate FROM jeffdatabase.tdays_d WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    tdays_d = mysql_query(query, engine_list['engine0'])

    # 数据入库
    from supporter.transformer import column_look_up

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
    a_list_suspend_2d = a_list_suspend_2d.reindex_like(adjret).fillna(0).astype(bool)
    event_panel = column_look_up(event_panel, a_list_suspend_2d, delay=0, kw='suspend', msg='未找到对应a_list_suspend')

    maxupordown['up'] = (maxupordown.maxupordown == 1).astype(int)
    maxupordown_2d = maxupordown.pivot(index='tradingdate', columns='stockcode', values='up')
    maxupordown_2d = maxupordown_2d.fillna(0).astype(bool)
    event_panel = column_look_up(event_panel, maxupordown_2d, delay=0, kw='up', msg='未找到对应 maxupordown')

    event_panel['isTradeday'] = event_panel.tradingdate.apply(lambda x: x in tdays_d.tradingdate.values).astype(bool)

    #
    def mask_efficiency(mask):
        _len = len(mask)
        l1 = mask.sum()
        print(f"{_len - l1} excluded from {_len} rows, left: {l1 / _len * 100:.2f} %")

    mask_l_instnum = event_panel['instnum'] < 6
    mask_efficiency(mask_l_instnum)
    mask_suspend = event_panel['suspend'] != 1
    mask_efficiency(mask_suspend)
    mask_maxup = event_panel['up'] != 1
    mask_efficiency(mask_maxup)
    mask_ipo = event_panel['new_ipo'] != 1
    mask_efficiency(mask_ipo)
    mask_tradeday = event_panel['isTradeday'] == 1
    mask_efficiency(mask_tradeday)

    mask_all = mask_l_instnum & mask_suspend & mask_maxup & mask_ipo & mask_tradeday
    mask_efficiency(mask_all)
    event_panel['tradeable'] = mask_all

    # oddadj, single, H_*, L_*
    tmp = event_panel.groupby('tradingdate')['tradeable'].sum().rename('td_cnt')
    tmp = tmp.reset_index()
    tmp['ex_cnt'] = tmp.td_cnt // 2
    panel = event_panel[event_panel.tradeable].merge(tmp, on='tradingdate', how='left')

    #
    event_panel.to_pickle(conf['data_path'] + f'event_panel(update_tmp)[{begin_date},{end_date}].pkl')

    # event_first_all_efr
    event_first_all = panel.pivot(index='tradingdate', columns='stockcode', values='fv')
    event_first_all = event_first_all.fillna(method='ffill', limit=2)
    event_first_all = event_first_all.fillna(0).apply(lambda s: s / s.abs().sum(), axis=1)
    event_first_all_efr = weight_to_efr(event_first_all, close_2d)

    # event_first_selected_efr
    mask_l_car_8 = panel.groupby('tradingdate').CAR_8.rank(ascending=False) > panel.ex_cnt
    mask_h_ar0 = panel.groupby('tradingdate').AR0.rank(ascending=True) > panel.ex_cnt
    mask_efficiency(mask_l_car_8 & mask_h_ar0)
    event_first_selected = panel[mask_l_car_8 & mask_h_ar0].pivot(index='tradingdate', columns='stockcode', values='fv')
    event_first_selected = event_first_selected.fillna(method='ffill', limit=2)
    event_first_selected = event_first_selected.fillna(0).apply(lambda s: s/s.abs().sum(), axis=1)
    # event_first_selected.to_csv(conf['factorscsv_path'] + 'efr_har0_lcar8_v1.csv')
    event_first_selected_efr = weight_to_efr(event_first_selected, close_2d)

    # event_first_hAR0_efr
    mask_h_ar0 = panel.groupby('tradingdate').AR0.rank(ascending=True) > panel.ex_cnt
    mask_efficiency(mask_h_ar0)
    event_first_selected = panel[mask_h_ar0].pivot(index='tradingdate', columns='stockcode', values='fv')
    event_first_selected = event_first_selected.fillna(method='ffill', limit=2)
    event_first_selected = event_first_selected.fillna(0).apply(lambda s: s/s.abs().sum(), axis=1)
    # event_first_selected.to_csv(conf['factorscsv_path'] + 'efr_har0_v2.csv')
    event_first_hAR0_efr = weight_to_efr(event_first_selected, close_2d)

    # event_first_lCAR8_efr
    mask_l_car_8 = panel.groupby('tradingdate').CAR_8.rank(ascending=False) > panel.ex_cnt
    mask_efficiency(mask_l_car_8)
    event_first_selected = panel[mask_l_car_8].pivot(index='tradingdate', columns='stockcode', values='fv')
    event_first_selected = event_first_selected.fillna(method='ffill', limit=2)
    event_first_selected = event_first_selected.fillna(0).apply(lambda s: s/s.abs().sum(), axis=1)
    # event_first_selected.to_csv(conf['factorscsv_path'] + 'efr_lcar8_v2.csv')
    event_first_lCAR8_efr = weight_to_efr(event_first_selected, close_2d)

    # 2d weight -> EFR

    dtypedict = {
        '调整日期': DATE(),
        '证券代码': VARCHAR(20),
        '持仓权重': DOUBLE(),
        '成本价格': DOUBLE(),
        '是否融资融券': VARCHAR(5),
    }
    engine_info = conf['mysql_engine']['engine4']

    def delete_and_upload(tname, efr_df, data_path, begin_date=begin_date, end_date=end_date):
        sql = f"DELETE FROM intern.{tname} WHERE 调整日期 >= '{begin_date}'"
        sql_delete(sql=sql, eg=engine_info)
        efr_df.to_sql(tname, con=engine_list['engine4'], if_exists='append', index=False, dtype=dtypedict)
        print(tname, begin_date, end_date, 'Uploaded.')
        filename = f'{tname}[{begin_date},{end_date}].xlsx'
        efr_df.to_excel(data_path + filename)
        print(f'{filename} Saved.')

    delete_and_upload('efr_baseline_dur3', event_first_all_efr, conf['data_path'])
    delete_and_upload('efr_har0_lcar8_dur3', event_first_selected_efr, conf['data_path'])
    delete_and_upload('efr_har0_dur3', event_first_hAR0_efr, conf['data_path'])
    delete_and_upload('efr_lcar8_dur3', event_first_lCAR8_efr, conf['data_path'])


def get_recent_efr_weight(conf):
    """Parse dates and tables to access daily PMS information"""
    mysql_engine = conf['mysql_engine']
    engine_list = {engine_id: conn_mysql(engine_info) for engine_id, engine_info in mysql_engine.items()}
    save_dir = conf['data_path']

    # begin_date = '2022-01-01'
    # end_date = '2022-12-31'
    # query = f"""SELECT tradingdate FROM jeffdatabase.tdays_d WHERE tradingdate>='{begin_date}' AND tradingdate<='{end_date}' ORDER BY tradingdate;"""
    # tdays_d = mysql_query(query, engine_list['engine0'])
    # date_last = tdays_d['tradingdate'].apply(lambda x: x.strftime('%Y-%m-%d')).to_list()
    date_last = None
    for tname in ['efr_baseline_dur3', 'efr_har0_lcar8_dur3', 'efr_har0_dur3', 'efr_lcar8_dur3']:  # ['efr_first_report_baseline1', 'efr_first_report_h_ar0_l_car_8_dur3']:
        _get_recent_efr_weight(tname, engine_list, save_dir, date_last=date_last)


def _get_recent_efr_weight(tname, engine_list, save_dir, date_last=None):
    """
    From updated MySQL database acquire PMS holding weights
    :param tname: a MySQL table name as source
    :param engine_list: a list of MySQL engines connected
    :param save_dir: a string of local save folder (mother)
    :param date_last: a list (or None) of tradedates to access
    :return:
    """
    if date_last is None:
        query = f'SELECT 调整日期 FROM intern.{tname} ORDER BY 调整日期 DESC LIMIT 1;'
        date_local = mysql_query(query, engine_list['engine2']).loc[0, '调整日期']
        date_last = date_local.strftime('%Y-%m-%d')

    if not isinstance(date_last, list):
        date_last = [date_last]

    df_efr = pd.DataFrame()
    for _date in date_last:
        query = f'SELECT * FROM intern.{tname} WHERE 调整日期="{_date}";'
        df_efr = pd.concat([df_efr, mysql_query(query, engine_list['engine2'])], axis=0)
    save_name = save_dir + tname + '[' + date_last[0] + ',' + date_last[-1] + '].xlsx'
    print(save_name)
    df_efr.to_excel(save_name, index=None)


# %%
if __name__ == '__main__':
    main()
