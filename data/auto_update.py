"""
(created by swmao on Feb. 16th)
1. 对比本地因子与当前时间，若需更新，
2. 从服务器获取所需截面切片，选择是否更新本地截面；
3. 计算需更新的因子值，存在本地；
4. 新的因子值，append到服务器。

"""
import pandas as pd
import numpy as np
import time, os
from datetime import timedelta
from sqlalchemy.dialects.mysql import BIGINT, DOUBLE, INTEGER, VARCHAR, DATE
from sqlalchemy import create_engine


# sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
# from supporter.mysql import conn_mysql, mysql_query
# from supporter.transformer import get_winsorize_sr
# from data.save_remote import transfer_pe_residual_table


def conn_mysql(eng: dict):
    """根据dict中的服务器信息，连接mysql"""
    user = eng['user']
    password = eng['password']
    host = eng['host']
    port = eng['port']
    dbname = eng['dbname']
    engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}?charset=UTF8MB4')
    return engine


def mysql_query(query, engine):
    """mysql接口，返回DataFrame"""
    return pd.read_sql_query(query, engine)


def get_winsorize_sr(sr: pd.Series, nsigma=3) -> pd.Series:
    """对series缩尾"""
    df = sr.copy()
    md = df.median()
    mad = 1.483 * df.sub(md).abs().median()
    up = df.apply(lambda k: k > md + mad * nsigma)
    down = df.apply(lambda k: k < md - mad * nsigma)
    df[up] = df[up].rank(pct=True).multiply(mad * 0.5).add(md + mad * (0.5 + nsigma))
    df[down] = df[down].rank(pct=True).multiply(mad * 0.5).add(md - mad * (0.5 + nsigma))
    return df


def add_id_column(df: pd.DataFrame, col0: str = 'tradingdate', col1: str = 'stockcode') -> pd.DataFrame:
    """由 col0: %Y-%m-%d, col1: 123456.XY 生成行id"""
    df['id'] = df[col0].apply(lambda s: s.replace('-', '')) + df[col1].apply(lambda s: s[:6])
    return df.sort_values('id').reset_index()


def transfer_pe_residual_table(df: pd.DataFrame) -> pd.DataFrame:
    """处理pe_residual*.csv"""
    df1 = df.T.unstack().reset_index().rename(columns={'level_0': 'industry', 'level_1': 'stockcode',
                                                       'level_2': 'tradingdate', 0: 'fv'})
    df1 = df1.dropna()
    df1['industry'] = df1['industry'].astype(int)
    df1 = add_id_column(df1)
    return df1[['tradingdate', 'stockcode', 'industry', 'fv', 'id']]


def pe_surprise_regress(trade_dates, begin_date, end_date,
                        factor_west_pe_180_2d, ci_sector_constituent_2d, factor_west_avgroe_180_2d,
                        factor_west_netprofit_growth_180_2d,
                        factor_west_netprofit_chg_180_6_1m_2d, factor_west_netprofit_chg_lid_2d, stk_west_surprise_2d,
                        instnum_class_2d, mv_class_2d,
                        group='ols_1', save_local_file=False, save_panel=False, factorscsv_path=None):
    """计算 pe_surprise_ols_?"""

    def ols_yhat(sub_df, fm, saying=False):
        """在DataFrame内依据公式回归返回预测值"""
        import statsmodels.formula.api as sm

        ols_res = sm.ols(formula=fm, data=sub_df).fit()
        if saying:
            print(ols_res.summary())
        return ols_res.predict(sub_df.iloc[:, 1:])

    save_filename = f"""pe_residual_{begin_date.replace('-', '')}_{end_date.replace('-', '')}.csv"""
    save_filename = save_filename.replace('pe_residual', f'pe_residual_{group}') if group != 'all' else save_filename
    factor_val = pd.DataFrame()
    panel_size = pd.DataFrame()
    lst_time = time_loop_start = time.time()
    td = trade_dates[0]
    for td in trade_dates:
        td_str = td.strftime('%Y-%m-%d')
        print('DATE:', td_str, end='\t')
        pe = factor_west_pe_180_2d.loc[td]
        # 查看pe的分布，确定取对数
        pe_log = pe.apply(np.log)
        panel = pe_log.rename('pe')
        sector = ci_sector_constituent_2d.loc[td]
        pe_log_winso = pe_log.groupby(sector).apply(lambda s: get_winsorize_sr(s))
        panel_winso = pe_log_winso.rename('pe')
        col_name = 'avgroe'
        df = factor_west_avgroe_180_2d.loc[td]
        df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
        panel = pd.concat([panel, df.rename(col_name)], axis=1)
        panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
        col_name = 'np_growth'
        df = factor_west_netprofit_growth_180_2d.loc[td]
        df1 = df  # 已经去过极值 df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
        panel = pd.concat([panel, df.rename(col_name)], axis=1)
        panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
        if group == 'all':
            # factor_west_netprofit_chg_180_6_1m
            col_name = 'np_chg_6m'
            df = factor_west_netprofit_chg_180_6_1m_2d.loc[td]
            df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
            panel = pd.concat([panel, df.rename(col_name)], axis=1)
            panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
            # factor_west_netprofit_chg_lid
            col_name = 'np_chg_lid'
            df = factor_west_netprofit_chg_lid_2d.loc[td]
            df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
            panel = pd.concat([panel, df.rename(col_name)], axis=1)
            panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
            # stk_west_surprise
            col_name = 'surprise'
            df = stk_west_surprise_2d.loc[td]
            df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
            panel = pd.concat([panel, df.rename(col_name)], axis=1)
            panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
        elif group == 'ols_1':
            # factor_west_netprofit_chg_180_6_1m
            col_name = 'np_chg_6m'
            df = factor_west_netprofit_chg_180_6_1m_2d.loc[td]
            df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
            panel = pd.concat([panel, df.rename(col_name)], axis=1)
            panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
        elif group == 'ols_2':
            # factor_west_netprofit_chg_lid
            col_name = 'np_chg_lid'
            df = factor_west_netprofit_chg_lid_2d.loc[td]
            df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
            panel = pd.concat([panel, df.rename(col_name)], axis=1)
            panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
        elif group == 'ols_3':
            # stk_west_surprise
            col_name = 'np_chg_6m'
            df = stk_west_surprise_2d.loc[td]
            df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
            panel = pd.concat([panel, df.rename(col_name)], axis=1)
            panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
        # instnum_class
        lhs = pd.get_dummies(instnum_class_2d.loc[td]).rename(columns={1: 'instnum1', 2: 'instnum2', 3: 'instnum3'})
        panel = pd.concat([panel, lhs], axis=1)
        panel_winso = pd.concat([panel_winso, lhs], axis=1)
        # mv_class
        lhs = pd.get_dummies(mv_class_2d.loc[td]).rename(columns={1: 'mv1', 2: 'mv2', 3: 'mv3'})
        panel = pd.concat([panel, lhs], axis=1)
        panel_winso = pd.concat([panel_winso, lhs], axis=1)
        # ci_sector_constituent
        lhs = sector.rename('sector')
        panel = pd.concat([panel, lhs], axis=1)
        panel_winso = pd.concat([panel_winso, lhs], axis=1)
        if save_local_file and save_panel:
            os.makedirs(factorscsv_path + save_filename.replace('.csv', ''), exist_ok=True)
            panel.to_csv(factorscsv_path + save_filename.replace('.csv', f'/{td_str}_raw.csv'))
            panel_winso.to_csv(factorscsv_path + save_filename.replace('.csv', f'/{td_str}_reg.csv'))
        # 面板去空值
        panel_ind_cnt = panel_winso.count()
        panel_nonna = panel_winso[panel_ind_cnt[(panel_ind_cnt >= 400)].index]  # 解释变量覆盖个股数量少于400则忽略该变量
        # panel_nonna = panel.dropna(how='all', axis=1)  # 若解释变量全部缺失，则剔除该解释变量（因此因子值时间段前后不可比较）
        panel_nonna = panel_nonna.dropna(how='any', axis=0)
        print('PANEL:', panel_nonna.shape, end='\t')
        panel_size[td_str] = panel_nonna.shape
        # 回归
        if len(panel_nonna) > 0:
            var_list = panel_nonna.columns.to_list()
            fm = var_list[0] + ' ~ ' + ' + '.join(var_list[1:-1])  # 回归公式
            fv2 = panel.groupby('sector').apply(lambda s: s['pe']) - panel_nonna.groupby('sector').apply(
                lambda s: ols_yhat(s, fm))
            fv2 = fv2.dropna().rename(td_str)
            factor_val = pd.concat((factor_val, fv2), axis=1)
        else:
            factor_val[td_str] = np.nan
        # notice
        cur_time = time.time()
        print(f'loop time {(cur_time - lst_time):.3f} s')
        lst_time = cur_time

    print(f'LOOP FINISHED, cost time {(time.time() - time_loop_start):.3f} s\n')
    if save_local_file:
        factor_val.to_csv(factorscsv_path + save_filename)
        panel_size.T.to_csv(factorscsv_path + save_filename.replace('pe_residual', 'panel_size'))
        print('newest date:', end_date, "save in", factorscsv_path + save_filename)

    return factor_val, save_filename


# %%
if __name__ == '__main__':
    # %% config
    # import yaml
    # conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    # conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    csv_path = '/mnt/c/Users/Winst/Documents/factors_csv/'  # conf['factorscsv_path']
    save_panel = False  # conf['save_panel']
    group = 'loop3'  # conf['pe_ols_group']
    pe_tables = [True,
                 'pe_residual_ols_3.csv',
                 'pe_residual_ols_2.csv',
                 'pe_residual_ols_1.csv']  # conf['tables']['pe_tables']

    mysql_engine = {'engine0': {'user': 'intern01',
                                'password': 'rh35th',
                                'host': '192.168.1.104',
                                'port': '3306',
                                'dbname': 'jeffdatabase'},
                    'engine1': {'user': 'intern01',
                                'password': 'rh35th',
                                'host': '192.168.1.104',
                                'port': '3306',
                                'dbname': 'factordatabase'},
                    'engine2': {'user': 'intern02',
                                'password': 'fh840t',
                                'host': '192.168.1.104',
                                'port': '3306',
                                'dbname': 'factordatabase'},
                    'engine3': {'user': 'intern01',
                                'password': 'rh35th',
                                'host': '192.168.1.104',
                                'port': '3306',
                                'dbname': 'intern'}}  # conf['mysql_engine']

    engine_list = {engine_id: conn_mysql(engine_info) for engine_id, engine_info in mysql_engine.items()}
    # update_target = [file for file_group in conf['tables'] for file in conf['tables'][file_group][1:] if conf['tables'][file_group][0]]

    # %% PE RESIDUAL 获取需要更新的时间范围
    update_target = [file.replace('.csv', '') for file in pe_tables[1:]]
    target_table = update_target[0]  # ols_1 ols_2 ols_3 同时更新，因此获取一个时间戳
    # for file in update_target:
    # fval_local = pd.read_csv(csv_path + file, index_col=0, parse_dates=True)

    query = f'SELECT tradingdate FROM intern.{target_table} ORDER BY tradingdate DESC LIMIT 1;'
    date_local = mysql_query(query, engine_list['engine2']).loc[0, 'tradingdate']
    prior_date = date_local - timedelta(120)  # stk_west_surprise 一季度一次需要填充空值！

    query = f"""SELECT tradingdate,stockcode,fv FROM factordatabase.factor_west_pe_180 WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    factor_west_pe_180 = mysql_query(query, engine_list['engine2'])

    begin_date = factor_west_pe_180['tradingdate'][factor_west_pe_180['tradingdate'] > date_local].min()
    end_date = factor_west_pe_180['tradingdate'].max()
    print(prior_date, date_local, begin_date, end_date)
    assert end_date > date_local  # 继续条件

    # %% 获取所需数据
    factor_west_pe_180_2d = factor_west_pe_180.pivot(index='tradingdate', columns='stockcode', values='fv')

    query = f"""SELECT tradingdate,stockcode,industry FROM jeffdatabase.ci_sector_constituent WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    ci_sector_constituent = mysql_query(query, engine_list['engine0'])
    ci_sector_constituent_2d = ci_sector_constituent.pivot(index='tradingdate', columns='stockcode', values='industry')
    ci_sector_constituent_2d = ci_sector_constituent_2d.reindex_like(factor_west_pe_180_2d).fillna(method='ffill').loc[
                               begin_date:]

    query = f"""SELECT tradingdate,stockcode,class FROM factordatabase.instnum_class WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    instnum_class = mysql_query(query, engine_list['engine1'])
    instnum_class_2d = instnum_class.pivot(index='tradingdate', columns='stockcode', values='class')
    instnum_class_2d = instnum_class_2d.reindex_like(factor_west_pe_180_2d).fillna(method='ffill').loc[begin_date:]

    query = f"""SELECT tradingdate,stockcode,class FROM factordatabase.mv_class WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    mv_class = mysql_query(query, engine_list['engine1'])
    mv_class_2d = mv_class.pivot(index='tradingdate', columns='stockcode', values='class')
    mv_class_2d = mv_class_2d.reindex_like(factor_west_pe_180_2d).fillna(method='ffill').fillna(method='backfill').loc[
                  begin_date:]

    query = f"""SELECT tradingdate,stockcode,fv FROM factordatabase.factor_west_netprofit_chg_180_6_1m WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    factor_west_netprofit_chg_180_6_1m = mysql_query(query, engine_list['engine2'])
    factor_west_netprofit_chg_180_6_1m_2d = factor_west_netprofit_chg_180_6_1m.pivot(index='tradingdate',
                                                                                     columns='stockcode', values='fv')
    factor_west_netprofit_chg_180_6_1m_2d = factor_west_netprofit_chg_180_6_1m_2d.reindex_like(
        factor_west_pe_180_2d).loc[
                                            begin_date:]

    query = f"""SELECT tradingdate,stockcode,fv FROM factordatabase.factor_west_netprofit_chg_lid WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    factor_west_netprofit_chg_lid = mysql_query(query, engine_list['engine2'])
    factor_west_netprofit_chg_lid_2d = factor_west_netprofit_chg_lid.pivot(index='tradingdate', columns='stockcode',
                                                                           values='fv')
    factor_west_netprofit_chg_lid_2d = factor_west_netprofit_chg_lid_2d.reindex_like(factor_west_pe_180_2d).loc[
                                       begin_date:]

    query = f"""SELECT update_date,stockcode,surprise FROM factordatabase.stk_west_surprise WHERE update_date>'{prior_date - timedelta(120)}' ORDER BY update_date;"""
    stk_west_surprise = mysql_query(query, engine_list['engine2']).groupby(['update_date', 'stockcode'])[
        'surprise'].mean().reset_index()
    stk_west_surprise_2d = stk_west_surprise.pivot(index='update_date', columns='stockcode', values='surprise')
    stk_west_surprise_2d = stk_west_surprise_2d.reindex_like(factor_west_pe_180_2d).fillna(method='ffill',
                                                                                           limit=120).loc[
                           begin_date:]

    query = f"""SELECT tradingdate,stockcode,fv FROM factordatabase.factor_west_avgroe_180 WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    factor_west_avgroe_180 = mysql_query(query, engine_list['engine2'])
    factor_west_avgroe_180_2d = factor_west_avgroe_180.pivot(index='tradingdate', columns='stockcode', values='fv')
    factor_west_avgroe_180_2d = factor_west_avgroe_180_2d.reindex_like(factor_west_pe_180_2d).loc[begin_date:]

    query = f"""SELECT tradingdate,stockcode,fv FROM factordatabase.factor_west_netprofit_growth_180 WHERE tradingdate>'{prior_date}' ORDER BY tradingdate;"""
    factor_west_netprofit_growth_180 = mysql_query(query, engine_list['engine2'])
    factor_west_netprofit_growth_180_2d = factor_west_netprofit_growth_180.pivot(index='tradingdate',
                                                                                 columns='stockcode',
                                                                                 values='fv')
    factor_west_netprofit_growth_180_2d = factor_west_netprofit_growth_180_2d.reindex_like(factor_west_pe_180_2d).loc[
                                          begin_date:]

    factor_west_pe_180_2d = factor_west_pe_180_2d.loc[begin_date:]

    # %% Loop & Upload
    trade_dates = factor_west_pe_180_2d.index.to_list()
    dtypedict = {
        'id': BIGINT(20, unsigned=True),
        'fv': DOUBLE(),
        'industry': INTEGER(),
        'stockcode': VARCHAR(20),
        'tradingdate': DATE()
    }

    # if group == 'loop3':
    for _group in ['ols_1', 'ols_2', 'ols_3']:
        factor_val, save_filename = pe_surprise_regress(trade_dates, begin_date.__str__(), end_date.__str__(),
                                                        factor_west_pe_180_2d, ci_sector_constituent_2d,
                                                        factor_west_avgroe_180_2d,
                                                        factor_west_netprofit_growth_180_2d,
                                                        factor_west_netprofit_chg_180_6_1m_2d,
                                                        factor_west_netprofit_chg_lid_2d,
                                                        stk_west_surprise_2d,
                                                        instnum_class_2d, mv_class_2d,
                                                        group=_group, save_local_file=True, save_panel=save_panel,
                                                        factorscsv_path=csv_path)  # 是否存在本地
        tname = save_filename.rsplit('_', 2)[0]
        print(tname, begin_date, end_date, 'Calculated.')
        df = transfer_pe_residual_table(factor_val)
        df.to_sql(tname, con=engine_list['engine3'], if_exists='append', index=False, dtype=dtypedict)
        print(tname, begin_date, end_date, 'Uploaded.')
