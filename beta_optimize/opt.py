"""
(created by swmao on April 22nd; rewrite on August 1st)

"""
import os
import time
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple
import cvxpy as cp
from multiprocessing import Pool, RLock, freeze_support
import matplotlib.pyplot as plt
import seaborn
import warnings
from sqlalchemy import create_engine

np.random.seed(9)
warnings.simplefilter("ignore")
seaborn.set_style("darkgrid")
plt.rc("figure", figsize=(9, 5))
plt.rc("font", size=12)
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rc("savefig", dpi=90)
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"

_PATH = '/home/swmao/'
_BASE = 'intern'  # 注意 main() - conf - mysql_engine - engine4 同步修改


def main():
    # Configs:
    conf = {
        'optimize_target': f'{_PATH}/optimize_target_v2.xlsx',  # 优化参数设置
        'fetch_alpha_first': True,  # 从服务器获取最新alpha，转化成 tradingdate列 stockcode行
        'factorscsv_path': f"{_PATH}/factors_csv/",  # alpha csv file 存放地址
        'factorsres_path': f"{_PATH}/factors_res/",  # 结果生成的地址
        'telling': False,  # 是否提示历次优化情况
        'alpha_from_database': {  # alpha来源的服务器信息，键值与下服务器序号对应
            'factor_apm': 'engine4',
        },
        # 服务器配置
        'mysql_engine': {
            'engine0': {'user': 'intern01',
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
                        'dbname': 'alphas101'},
            'engine4': {'user': 'intern01',
                        'password': 'rh35th',
                        'host': '192.168.1.104',
                        'port': '3306',
                        'dbname': 'intern'},
            'engine5': {'user': 'intern01',
                        'password': 'rh35th',
                        'host': '192.168.1.104',
                        'port': '3306',
                        'dbname': 'jqdata'},
            'engine6': {'user': 'intern01',
                        'password': 'rh35th',
                        'host': '192.168.1.104',
                        'port': '3306',
                        'dbname': 'alphas_jqdata'}
        },
        # 'data_path': None,  # "/mnt/c/Users/Winst/Documents/data_local/",
        # 'idx_constituent': None,  # "/mnt/c/Users/Winst/Documents/data_local/idx_constituent_{}.csv",
        # 'closeAdj': None,  # "/mnt/c/Users/Winst/Documents/data_local/stk_marketdata_closeAdj.csv",
        # 'tc': None,  # 2e-3,  # 手续费（双边）
        # 'barra_panel': None,  # '/mnt/c/Users/Winst/Documents/data_local/BARRA/barra_panel.h5',  # key= 'y2022'  正交化后的暴露面板
        # 'fct_cov_path': None,  # '/mnt/c/Users/Winst/Documents/data_local/BARRA/F_NW_Eigen_VRA[2014-02-10,2022-03-30].csv',
        # 'specific_risk': None,  # '/mnt/c/Users/Winst/Documents/data_local/BARRA/D_NW_SM_SH_VRA[2014-02-10,2022-03-30].csv',
        # 'dat_path_pca': None,  # '/mnt/c/Users/Winst/Documents/data_local/PCA/d120,pc60,ipo60/',
    }
    os.makedirs(conf['factorscsv_path'], exist_ok=True)
    os.makedirs(conf['factorsres_path'], exist_ok=True)
    optimize(conf, mkdir_force=True, process_num=1)  # process_num > 1, 多进程


def cal_result_stat(df: pd.DataFrame, save_path: str = None, kind='cumsum', freq='D', lang='EN') -> pd.DataFrame:
    """
    对日度收益序列df计算相关结果
    :param lang:
    :param df: 值为日收益率小r，列index为日期DateTime
    :param save_path: 存储名（若有）
    :param kind: 累加/累乘
    :param freq: M D W Y
    :return: 结果面板
    """
    if kind == 'cumsum':
        df1 = df.cumsum() + 1
    elif kind == 'cumprod':
        df1 = df.add(1).cumprod()
    else:
        raise ValueError(f"""Invalid kind={kind}, only support('cumsum', 'cumprod')""")

    if freq == 'D':
        freq_year_adj = 242
    elif freq == 'W':
        freq_year_adj = 48
    elif freq == 'M':
        freq_year_adj = 12
    elif freq == 'Y':
        freq_year_adj = 1
    else:
        raise ValueError(f"""Invalid freq={freq}, only support('D', 'W', 'M', 'Y')""")

    data = df.copy()
    data['Date'] = data.index
    data['SemiYear'] = data['Date'].apply(lambda s: f'{s.year}-H{s.month // 7 + 1}')
    res: pd.DataFrame = data.groupby('SemiYear')[['Date']].last().reset_index()
    res.index = res['Date']
    res['Cash'] = (2e7 * df1.loc[res.index]).round(1)
    res['UnitVal'] = df1.loc[res.index]
    res['TRet'] = res['UnitVal'] - 1
    res['PRet'] = res['UnitVal'].pct_change()
    res.iloc[0, -1] = res['UnitVal'].iloc[0] - 1
    res['PSharpe'] = df.groupby(data.SemiYear).apply(lambda s: s.mean() / s.std() * np.sqrt(freq_year_adj)).values
    mdd = df1 / df1.cummax() - 1 if kind == 'cumprod' else df1 - df1.cummax()
    res['PMaxDD'] = mdd.groupby(data.SemiYear).min().values
    res['PCalmar'] = res['PRet'] / res['PMaxDD'].abs()
    res['PWinR'] = df.groupby(data['SemiYear']).apply(lambda s: (s > 0).mean()).values
    res['TMaxDD'] = mdd.min().values[0]
    res['TSharpe'] = (df.mean() / df.std() * np.sqrt(freq_year_adj)).values[0]
    res['TCalmar'] = res['TSharpe'] / res['TMaxDD'].abs()
    res['TWinR'] = (df > 0).mean().values[0]
    res['TAnnRet'] = (df1.iloc[-1] ** (freq_year_adj / len(df1)) - 1).values[0]

    res['Date'] = res['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    if lang == 'CH':
        res1 = pd.DataFrame(columns=res.columns, index=['CH'])
        res1.loc['CH', :] = ['日期', '年度', '资金', '净值', '累计收益',
                             '收益', '夏普', '回撤', '卡玛', '胜率',
                             '总回撤', '总夏普', '总卡玛', '总胜率',
                             '年化收益', ]
        res = pd.concat([res, res1], ignore_index=True)

    res = res.set_index('SemiYear')
    if save_path is not None:
        table_save_safe(res, save_path)
    return res


def cal_sr_max_drawdown(df: pd.Series, ishow=False, title=None, save_path=None, kind='cumprod') -> pd.DataFrame:
    """计算序列回撤"""
    cm = df.cummax()
    mdd = pd.DataFrame(index=df.index)
    mdd[f'{df.name}_maxdd'] = (df / cm - 1) if kind == 'cumprod' else (df - cm)

    if save_path is not None:
        try:
            mdd.plot(kind='area', figsize=(9, 5), grid=True, color='y', alpha=.5, title=title)
        except ValueError:
            mdd[mdd > 0] = 0
            mdd.plot(kind='area', figsize=(9, 5), grid=True, color='y', alpha=.5, title=title)
        finally:
            plt.tight_layout()
            plt.savefig(save_path)
            if ishow:
                plt.show()
            plt.close()

    return mdd


def conn_mysql(eng: dict):
    """根据dict中的服务器信息，连接mysql"""
    user = eng['user']
    password = eng['password']
    host = eng['host']
    port = eng['port']
    dbname = eng['dbname']
    engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}?charset=UTF8MB4')
    return engine


def df_union_sub(df0: pd.DataFrame, df1: pd.DataFrame) -> pd.DataFrame:
    """Align index and columns and calculate df0 - df1"""
    assets0 = df0.columns
    assets1 = df1.columns
    views0 = df0.index
    views1 = df1.index
    assets = assets0.union(assets1)
    views = views0.union(views1)
    df0 = pd.DataFrame(df0, columns=assets, index=views).fillna(0)
    df1 = pd.DataFrame(df1, columns=assets, index=views).fillna(0)
    res = df0.sub(df1)
    return res


def get_accessible_stk(i: set, a: set, b: set, s: set = None) -> Tuple[list, list, dict]:
    bs = b.intersection(s) if s else b  # beta (and sigma)
    i_a = i.difference(a)  # idx cons w/o alpha
    i_b = i.difference(bs)  # idx cons w/o beta (and sigma)
    info = {'#i_a': len(i_a), 'i_a': list(i_a),
            '#i_b(s)': len(i_b), 'i_b(s)': list(i_b)}
    pool = list(a.intersection(bs))  # accessible asset pool
    base = list(i.intersection(bs))  # index component with beta (and sigma)
    return pool, base, info


def get_alpha_dat(alpha_name, csv_path, bd, ed, fw=0) -> pd.DataFrame:
    """"""
    file_path = csv_path + alpha_name + '.csv'
    if not os.path.exists(file_path):
        raise Exception(f'Alpha CSV file not found in `{file_path}')
    dat = pd.read_csv(file_path, index_col=0, parse_dates=True)

    if fw:
        dat = dat.shift(fw)

    if dat.index.values[-1] < pd.to_datetime(ed):
        raise Exception(f'Update local CSV file `{file_path}` to newest date {ed} FIRST!')
    dat = dat.loc[bd: ed]

    # Alpha调整：截面标准差调整到2.25%
    dat = dat.apply(lambda s: (s - np.nanmean(s)) / np.nanstd(s) * 2.25e-2, axis=1)

    return dat


def get_beta_expo_cnstr(beta_kind, conf, bd, ed, H0, H1, beta_args, l_cvg_fill=True):
    def cvg_f_fill(fr: pd.DataFrame, w=10, q=.75, ishow=False, notify=False) -> pd.DataFrame:
        """F-Fill if Low Coverage: 日覆盖率低于过去w日均值的q倍时填充"""
        beta_covered_stk_num = fr.index.get_level_values(0).value_counts().sort_index()
        mask_l_cvg = beta_covered_stk_num < (beta_covered_stk_num.shift(1).rolling(w).mean() * q)
        rep_tds = beta_covered_stk_num[mask_l_cvg]
        # print(rep_tds)
        tds = fr.index.get_level_values(0).unique()
        tds = pd.Series(tds, index=tds)
        # td = rep_tds.index[0]
        for td in rep_tds.index:
            td_1 = tds[:td].iloc[-2]
            td1 = tds[td:].iloc[1]
            if notify:
                print(td.strftime('%Y-%m-%d'), '->', td_1.strftime('%Y-%m-%d'), len(fr.loc[td]), '->',
                      len(fr.loc[td_1]))
            fr = pd.concat([fr.loc[:td_1], fr.loc[td_1:td_1].rename(index={td_1: td}), fr.loc[td1:]])
        if ishow:
            from matplotlib import pyplot as plt
            plt.plot(beta_covered_stk_num)
            plt.plot(fr.index.get_level_values(0).value_counts().sort_index())
            plt.tight_layout()
            plt.title(f'F-Fill Coverage Lower than {q} * (past {w}d mean)')
            plt.show()
            plt.close()
        return fr

    def get_barra_exposure(from_remote=True) -> pd.DataFrame:
        """
        Get Barra Exposure DataFrame.
        :return: frame like
                                  beta  ...  ind_CI005030.WI
        2016-02-01 000001.SZ  0.485531  ...              NaN
                   000004.SZ  0.593129  ...              NaN
                   000005.SZ  0.884217  ...              NaN
                   000006.SZ  0.845717  ...              NaN
                   000008.SZ -0.496430  ...              NaN
                                ...  ...              ...
        2022-03-31 688799.SH -0.437337  ...              0.0
                   688800.SH  1.768729  ...              0.0
                   688819.SH -0.312409  ...              0.0
                   688981.SH -2.325878  ...              0.0
                   689009.SH -1.174486  ...              0.0
        """

        if from_remote:
            print('Load Barra exposure panel orthogonal...')

            # Style exposure
            expo_style = mysql_query(
                query=f"SELECT * FROM {_BASE}.barra_exposure_orthogonal"
                      f" WHERE tradingdate>='{bd}' AND tradingdate<='{ed}'",
                engine=conn_mysql(eng=conf['mysql_engine']['engine4'])
            )
            expo_style['tradingdate'] = pd.to_datetime(expo_style['tradingdate'])
            expo_style = expo_style.set_index(['tradingdate', 'stockcode'])
            expo_style = expo_style.dropna()
            mask = expo_style.index.get_level_values(0)
            expo_style: pd.DataFrame = expo_style.groupby(mask).apply(lambda s: (s - s.mean()) / s.std())

            # Industry exposure
            td_1 = mysql_query(
                query=f"SELECT tradingdate"
                      f" FROM jeffdatabase.ind_citic_constituent"
                      f" WHERE tradingdate<='{bd}' ORDER BY tradingdate DESC LIMIT 1",
                engine=conn_mysql(eng=conf['mysql_engine']['engine0'])
            )
            td_1 = td_1.loc[0, 'tradingdate'].strftime('%Y-%m-%d')
            tds = mysql_query(
                query=f"SELECT tradingdate FROM jeffdatabase.tdays_d"
                      f" WHERE tradingdate>='{td_1}' AND tradingdate<='{ed}'",
                engine=conn_mysql(eng=conf['mysql_engine']['engine0'])
            )
            tds['tradingdate'] = pd.to_datetime(tds['tradingdate'])
            tds = tds.set_index('tradingdate')
            expo_indus = mysql_query(
                query=f"SELECT tradingdate,stockcode,industry_l1"
                      f" FROM jeffdatabase.ind_citic_constituent"
                      f" WHERE tradingdate>='{td_1}'"
                      f" AND tradingdate<='{ed}'",
                engine=conn_mysql(eng=conf['mysql_engine']['engine0'])
            )
            expo_indus['tradingdate'] = pd.to_datetime(expo_indus['tradingdate'])
            expo_indus = expo_indus.pivot('tradingdate', 'stockcode', 'industry_l1')
            expo_indus = pd.concat([tds, expo_indus], axis=1).fillna(method='ffill')
            expo_indus = expo_indus.loc[bd:]
            expo_indus = pd.get_dummies(expo_indus.stack())
            expo_indus.columns = ['ind_' + _.rsplit('_', maxsplit=1)[-1].split('.')[0] for _ in expo_indus.columns]

            res = pd.concat([expo_style, expo_indus], axis=1).sort_index()

            return res

        else:
            expo = pd.DataFrame()
            begin_year = int(bd.split('-')[0])
            end_year = int(ed.split('-')[0])
            for _ in range(begin_year, end_year + 1):
                expo_style = pd.DataFrame(pd.read_hdf(conf['barra_panel'], key=f'y{_}'))
                expo = pd.concat([expo, expo_style])
            expo: pd.DataFrame = expo.loc[bd: ed]

            cols_style = [c for c in expo.columns if 'rtn' not in c and 'ind' not in c and 'country' != c]
            mask = expo[cols_style].index.get_level_values(0)
            expo_style: pd.DataFrame = expo[cols_style].groupby(mask).apply(lambda s: (s - s.mean()) / s.std())

            cols_indus = [c for c in expo.columns if 'ind_' in c]
            expo_indus: pd.DataFrame = expo[cols_indus]

            res = pd.concat([expo_style, expo_indus], axis=1)

            return res

    def get_pca_exposure(PN=20) -> pd.DataFrame:
        """
        Exposure on Zscore(PCA) factors top PN=20
        Return:
                                 pc000     pc001  ...     pc018     pc019
        2016-02-01 000001.SZ -1.010773  0.218575  ... -0.513719  0.187009
                   000002.SZ -2.117420 -2.055099  ... -0.098845 -0.792386
                   000004.SZ -0.590739  0.005483  ... -1.191663  0.056746
                   000006.SZ  1.130240 -0.238240  ... -0.997222 -0.269062
                   000007.SZ -0.462391  0.111827  ...  0.028676 -0.766955
                                ...       ...  ...       ...       ...
        2022-03-31 301017.SZ -1.086786  1.361293  ... -0.910399 -1.174649
                   688239.SH -1.546764  0.589980  ... -1.662582 -0.858083
                   301020.SZ -0.183162  0.070591  ...  0.217342 -1.335232
                   301021.SZ  0.016632 -0.603018  ... -0.476811 -0.581117
                   605287.SH  0.869714 -1.449705  ... -0.251319 -1.681979
        """

        def combine_pca_exposure(src, tgt, suf):
            """Run it ONCE to COMBINE principal exposures, when PCA is updated"""
            expo = pd.DataFrame()
            pass;  # print(f'\nMerge PCA exposure PN={PN} {suf}')
            for pn in tqdm(range(PN)):
                kw = f'pc{pn:03d}'
                df = pd.read_csv(src + f'{kw}.csv', index_col=0, parse_dates=True)
                df = df.loc[bd: ed]
                expo = pd.concat([expo, df.stack().rename(kw)], axis=1)
            #
            expo.to_pickle(tgt)
            return expo

        def get_suffix(s):
            s1 = s.split('-')
            return s1[0][-2:] + s1[1]

        suffix = f"{get_suffix(bd)}_{get_suffix(ed)}"
        pkl_path = conf['data_path'] + f"exposure_pca{PN}_{suffix}.pkl"
        if os.path.exists(pkl_path):
            expo_pca: pd.DataFrame = pd.read_pickle(pkl_path)
        else:
            expo_pca = combine_pca_exposure(conf['dat_path_pca'], pkl_path, suffix)

        ind_date = expo_pca.index.get_level_values(0)
        expo_pca = expo_pca.groupby(ind_date).apply(lambda s: (s - s.mean()) / s.std())
        return expo_pca

    def get_beta_constraint(all_c, info) -> pd.DataFrame:
        res = pd.DataFrame(index=all_c, columns=['L', 'H'])
        res.loc[:, 'L'] = np.nan  # -np.inf
        res.loc[:, 'H'] = np.nan  # np.inf
        for item in info:
            res.loc[item[0], 'L'] = item[1]
            res.loc[item[0], 'H'] = item[2]
        return res

    if beta_kind == 'Barra':
        expo_beta = get_barra_exposure(from_remote=True)
        sty_c = beta_args[0]  # ['size', 'beta', 'momentum']
        ind_c = [c for c in expo_beta.columns if 'ind' == c[:3]]
        # cnstr_info = [(sty_c, expoL, expoH), (ind_c, expoL, expoH)]
        cnstr_info = [(sty_c, -H0, H0), (ind_c, -H1, H1)]
        cnstr_beta = get_beta_constraint(all_c=expo_beta.columns, info=cnstr_info)

    elif beta_kind == 'PCA':
        raise Exception('Only support Barra exposure!')
        # principal_number = beta_args[0]  # 20
        # expo_beta = get_pca_exposure(PN=principal_number)
        # cnstr_info = [(list(expo_beta.columns), -H0, H0)]
        # cnstr_beta = get_beta_constraint(all_c=expo_beta.columns, info=cnstr_info)

    else:
        raise Exception('beta_kind {Barra, PCA}')

    if l_cvg_fill:  # beta覆盖不足，用上一日的beta暴露填充
        expo_beta = cvg_f_fill(expo_beta, w=10, q=.75, ishow=False)

    return expo_beta, cnstr_beta


def get_factor_covariance(path_F, bd=None, ed=None, from_remote=True) -> pd.DataFrame:
    """
    mat F: Sigma = X F X - D
    :param path_F: path of csv file or config dict
    :param bd: begin date
    :param ed: end date
    :param from_remote: True then load from remote database
    :return: factor covariance matrix or matrices, like
    """
    print('Load covariance of pure factor returns (Barra)')
    if from_remote:

        # 传入的config配置记录服务器信息
        assert isinstance(path_F, dict)
        conf = path_F

        # 从数据库获取
        df = mysql_query(
            query=f"SELECT *"
                  f" FROM {_BASE}.barra_factor_cov_nw_eigen_vra"
                  f" WHERE tradingdate>='{bd}'"
                  f" AND tradingdate<='{ed}'",
            engine=conn_mysql(conf['mysql_engine']['engine4'])
        )
        df['tradingdate'] = pd.to_datetime(df['tradingdate'])

        #
        if ('fname' in df.columns) and ('names' not in df.columns):
            df = df.rename(columns={'fname': 'names'})
        df = df.set_index(['tradingdate', 'names'])

    else:
        assert isinstance(path_F, str)
        df = pd.read_csv(path_F, index_col=[0, 1], parse_dates=[0])

    # if fw > 0:  # shift fw tradedates in index.level0
    #     df = df.groupby(['names']).shift(fw)
    #     df = df.loc[df.index.get_level_values(0).unique()[fw]:]
    df = df.loc[bd:] if bd is not None else df
    df = df.loc[:ed] if ed is not None else df

    return df


def get_index_constitution(conf, bd, ed, mkt_type='CSI500', from_remote=True) -> pd.DataFrame:
    """
    Read csv file - index constituent, return cons stock weight, sum 1
    :param conf:
    :param bd: return begin date
    :param ed: return end date
    :param mkt_type:
    :param from_remote:
    :return: DataFrame of shape (n_views, n_assets)
    """

    if from_remote:
        print('Load market index constitution weight...')

        # Decide market index code
        a_dict = {
            'CSI500': '000905.SH',
            'CSI300': '000300.SH',
        }
        if mkt_type in a_dict:
            a_code = a_dict[mkt_type]
        else:
            raise Exception(f'Unsupported market index: {mkt_type}')

        # Get index constituent
        df = mysql_query(
            query=f"SELECT tradingdate,stockcode"
                  f" FROM jeffdatabase.idx_constituent"
                  f" WHERE indexcode='{a_code}'"
                  f" AND tradingdate>='{bd}'"
                  f" AND tradingdate<='{ed}'",
            engine=conn_mysql(eng=conf['mysql_engine']['engine0'])
        )
        df['fv'] = 1
        df['tradingdate'] = pd.to_datetime(df['tradingdate'])
        df = df.pivot('tradingdate', 'stockcode', 'fv')

        df = df.fillna(0)

        # Decide constituent weight from market freeshares
        df1 = mysql_query(
            query=f"SELECT tradingdate,stockcode,mkt_freeshares"
                  f" FROM jeffdatabase.stk_marketvalue"
                  f" WHERE tradingdate>='{bd}'"
                  f" AND tradingdate<='{ed}'",
            engine=conn_mysql(eng=conf['mysql_engine']['engine0'])
        )
        df1['tradingdate'] = pd.to_datetime(df1['tradingdate'])
        df1 = df1.pivot('tradingdate', 'stockcode', 'mkt_freeshares')

        df = df1.reindex_like(df) * df
        df = df.apply(lambda s: s / s.sum(), axis=1)

        return df

    else:
        csv = conf['idx_constituent'].format(mkt_type)
        ind_cons = pd.read_csv(csv, index_col=0, parse_dates=True)
        ind_cons = ind_cons.loc[bd: ed]
        ind_cons = ind_cons.dropna(how='all', axis=1)
        # ind_cons = ind_cons.fillna(0)
        ind_cons /= 100
        return ind_cons


def get_save_path(res_path, mkt_type, alpha_name):
    save_suffix = f'OptResWeekly[{mkt_type}]{alpha_name}'
    save_path = f"{res_path}{save_suffix}/"
    os.makedirs(save_path, exist_ok=True)
    return save_path


def get_specific_risk(path_D, bd=None, ed=None, from_remote=True) -> pd.DataFrame:
    """
    mat D: Sigma = X F X - D
    :param path_D: .../D_NW_SM_SH_VRA[yyyy-mm-dd,yyyy-mm-dd].csv
    :param bd:
    :param ed:
    :param from_remote:
    :return: dataframe of diag item of D

    """
    if from_remote:
        assert isinstance(path_D, dict)
        conf = path_D
        df = mysql_query(
            query=f"SELECT * FROM {_BASE}.barra_specific_risk_nw_sm_sh_vra"
                  f" WHERE tradingdate>='{bd}' AND tradingdate<='{ed}'",
            engine=conn_mysql(eng=conf['mysql_engine']['engine4'])
        )
        df['tradingdate'] = pd.to_datetime(df['tradingdate'])
        df = df.pivot('tradingdate', 'stockcode', 'fv')
        return df

    else:
        assert isinstance(path_D, str)
        df = pd.read_csv(path_D, index_col=0, parse_dates=True)
        df = df.loc[bd:] if bd is not None else df
        df = df.loc[:ed] if ed is not None else df
        return df


def get_tradedates(conf, begin_date, end_date, kind='tdays_w', from_remote=True) -> pd.Series:
    if from_remote or (kind not in conf):
        print(f'Load {kind}...')
        df = mysql_query(
            query=f"SELECT tradingdate FROM jeffdatabase.{kind}"
                  f" WHERE tradingdate>='{begin_date}' AND tradingdate<='{end_date}'",
            engine=conn_mysql(eng=conf['mysql_engine']['engine0'])
        )
        df['tradingdate'] = pd.to_datetime(df['tradingdate'])
        df = df.set_index(df['tradingdate'])
        tradedates = df['tradingdate']
        return tradedates
    else:
        tdays_d = pd.read_csv(conf[kind], header=None, index_col=0, parse_dates=True)
        tdays_d = tdays_d.loc[begin_date: end_date]
        tdays_d['tdays_d'] = tdays_d.index
        tradedates = tdays_d.tdays_d
        return tradedates


def info2suffix(ir1: pd.Series) -> str:
    """suffix for all result file"""
    return f"{ir1['beta_suffix']}(B={ir1['B']},E={ir1['E']},D={ir1['D']},H0={ir1['H0']}" + \
           ('' if np.isnan(float(ir1['H1'])) else f",H1={ir1['H1']}") + \
           (f",G={ir1['G']}" if float(ir1['G']) > 0 else '') + \
           (f",S={ir1['S']}" if float(ir1['S']) < np.inf else '') + ')'


def io_make_sub_dir(path, force=False, inp=False):
    if force:
        os.makedirs(path, exist_ok=True)
    else:
        if os.path.exists(path):
            if os.path.isdir(path) and len(os.listdir(path)) == 0:
                return 1
            else:
                if inp:
                    return 0
                cmd = input(f"Write in non-empty dir '{path}' ?(y/N)")
                if cmd != 'y' and cmd != 'Y':
                    raise FileExistsError(path)
        else:
            os.makedirs(path, exist_ok=False)
    return 1


def load_optimize_target(opt_tgt: str) -> pd.DataFrame:
    df = pd.read_excel(opt_tgt, index_col=0, dtype=object)
    df = df[df.index == 1]
    # df['N'] = df['N'].apply(lambda x: float(x))
    # df['H0'] = df['H0'].apply(lambda x: float(x))
    # df['H1'] = df['H1'].apply(lambda x: float(x))
    # df['B'] = df['B'].apply(lambda x: float(x) / 100)
    # df['E'] = df['E'].apply(lambda x: float(x) / 100)
    # df['D'] = df['D'].apply(lambda x: float(x))
    # df['G'] = df['G'].apply(lambda x: float(x) * 1e4)
    # df['S'] = df['S'].apply(lambda x: float(x))
    # df['wei_tole'] = df['wei_tole'].apply(lambda x: float(x))
    # df['opt_verbose'] = df['opt_verbose'].apply(lambda x: x == 'TRUE')

    return df


def mysql_query(query, engine, telling=True):
    """mysql接口，返回DataFrame"""
    if telling:
        print(query)
    return pd.read_sql_query(query, engine)


def optimize(conf: dict, mkdir_force: bool, process_num: int):
    t0 = time.time()

    # Optimize Target:
    opt_tgt_path = conf['optimize_target']
    optimize_target = load_optimize_target(opt_tgt=opt_tgt_path)
    print(optimize_target)

    telling = conf['telling']
    fetch_alpha_first = conf['fetch_alpha_first']
    alpha_from_database = conf['alpha_from_database']  # {'factor_apm': 'engine4'}

    # Run optimize:
    f_pid = os.getpid()
    if process_num > 1:
        print(f'father process {f_pid}')
        freeze_support()
        p = Pool(process_num, initializer=tqdm.set_lock, initargs=(RLock(),))
        cnt = 0
        for ir in optimize_target.iterrows():
            ir1 = ir[1]
            p.apply_async(
                optimize1,
                args=[
                    (conf, ir1, mkdir_force, cnt % process_num, f_pid),
                    telling,
                    fetch_alpha_first,
                    alpha_from_database
                ]
            )
            cnt += 1
        p.close()
        p.join()
    else:
        for ir in optimize_target.iterrows():
            ir1 = ir[1]

            args = (conf, ir1, mkdir_force, 0, f_pid)

            optimize1(args, telling, fetch_alpha_first, alpha_from_database)

    # Exit:
    print(f'\nTime used: {second2clock(round(time.time() - t0))}')


def optimize1(args, telling=False, fetch_alpha_first=True, alpha_from_database={'factor_apm': 'engine4'}):
    """"""
    # Decode setting and parameters  # TODO: de config & track args in dict
    conf: dict = args[0]
    ir1: pd.Series = args[1]
    dir_force: bool = args[2]
    pos: int = args[3]
    f_pid: int = args[4]

    csv_path: str = conf['factorscsv_path']
    res_path: str = conf['factorsres_path']

    # idx_cons_path: str = conf['idx_constituent']
    # fct_cov_path: str = conf['fct_cov_path']
    # stk_rsk_path: str = conf['specific_risk']
    # close_adj_path: str = conf['closeAdj']
    # trade_cost: float = float(conf['tc'])

    mkt_type = ir1['mkt_type']
    begin_date = ir1['begin_date']
    end_date = ir1['end_date']
    if not isinstance(end_date, str) or end_date == 'NA':
        df = mysql_query(
            query=f"SELECT tradingdate"
                  f" FROM {_BASE}.barra_factor_cov_nw_eigen_vra"
                  f" WHERE tradingdate>={begin_date}"
                  f" ORDER BY tradingdate DESC LIMIT 1",
            engine=conn_mysql(eng=conf['mysql_engine']['engine4'])
        )
        end_date = df.loc[0, 'tradingdate'].strftime('%Y-%m-%d')
        del df
    N = float(ir1['N'])
    opt_verbose = (ir1['opt_verbose'] == 'TRUE')
    B = float(ir1['B']) / 100
    E = float(ir1['E']) / 100
    H0 = float(ir1['H0'])
    H1 = float(ir1['H1'])
    D = float(ir1['D'])
    G = float(ir1['G']) * 1e4
    S = float(ir1['S'])
    wei_tole = float(ir1['wei_tole'])
    alpha_name = ir1['alpha_name']
    beta_kind = ir1['beta_kind']
    suffix = info2suffix(ir1)
    script_info = {
        'opt_verbose': opt_verbose, 'begin_date': begin_date, 'end_date': end_date, 'mkt_type': mkt_type,
        'N': N, 'H0': H0, 'H1': H1, 'B': B, 'E': E, 'D': D, 'G': G, 'S': S, 'wei_tole': wei_tole,
        'alpha_name': alpha_name, 'beta_kind': beta_kind, 'alpha_5d_rank_ic': 'NA', 'suffix': suffix,
    }

    beta_args = eval(ir1['beta_args'])
    save_path = get_save_path(
        res_path=res_path,
        mkt_type=mkt_type,
        alpha_name=alpha_name)
    save_path_sub = f'{save_path}{suffix}/'
    if io_make_sub_dir(save_path_sub, force=dir_force, inp=(os.getpid() != f_pid)) == 0:
        return

    # Download Alpha Value
    if fetch_alpha_first and alpha_name in alpha_from_database.keys():
        engine_info: dict = conf['mysql_engine'][alpha_from_database[alpha_name]]
        file_path = csv_path + alpha_name + '.csv'
        if os.path.exists(file_path):
            df0 = pd.read_csv(file_path, index_col=0, parse_dates=True)
            bd1 = df0.index[-1].strftime('%Y-%m-%d')
        else:
            df0 = pd.DataFrame()
            bd1 = begin_date

        df = mysql_query(
            query=f"SELECT tradingdate,stockcode,fv"
                  f" FROM {engine_info['dbname']}.{alpha_name}"
                  f" WHERE tradingdate>='{bd1}'"
                  f" AND tradingdate<='{end_date}'",
            engine=conn_mysql(eng=engine_info)
        )
        df['tradingdate'] = pd.to_datetime(df['tradingdate'])
        df = df.pivot('tradingdate', 'stockcode', 'fv')

        df = pd.concat([df0, df])
        df = df.drop_duplicates().sort_index()
        df.to_csv(file_path)

    # Load DataFrames
    alpha: pd.DataFrame = get_alpha_dat(
        alpha_name=alpha_name,
        csv_path=csv_path,
        bd=begin_date,
        ed=end_date,
        fw=0  # Alpha 日期对应当天收盘更新（次日开盘前更新）
    )
    tradedates = get_tradedates(
        conf=conf,
        begin_date=begin_date,
        end_date=end_date,
        kind='tdays_w',  # TODO: daily optimize
        from_remote=True
    )
    beta_expo, beta_cnstr = get_beta_expo_cnstr(
        beta_kind=beta_kind,
        conf=conf,
        bd=begin_date,
        ed=end_date,
        H0=H0,
        H1=H1,
        beta_args=beta_args
    )
    ind_cons = get_index_constitution(
        conf=conf,
        bd=begin_date,
        ed=end_date,
        mkt_type=mkt_type,
        from_remote=True
    )
    fct_cov = get_factor_covariance(
        path_F=conf,
        bd=begin_date,
        ed=end_date,
        from_remote=True
    )
    stk_rsk = get_specific_risk(
        path_D=conf,
        bd=begin_date,
        ed=end_date,
        from_remote=True
    )

    desc = suffix
    all_args = (
        tradedates,
        beta_expo,
        beta_cnstr,
        ind_cons,
        fct_cov,
        stk_rsk,
        alpha,
        (
            mkt_type,
            N,
            D,
            B,
            E,
            G,
            S,
            wei_tole,
            opt_verbose,
            desc,
            pos
        )
    )
    #  Optimize:
    portfolio_weight, optimize_iter_info = portfolio_optimize(all_args=all_args,
                                                              telling=telling)

    #  Save Historical Optimize Results:
    with open(save_path_sub + 'config_optimize.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(script_info, f)
    optimize_iter_info.T.to_excel(
        save_path_sub + f'opt_info_{suffix}.xlsx')
    portfolio_weight.to_csv(
        save_path_sub + 'portfolio_weight_{}.csv'.format(suffix))
    print(f'Optimize Information & Portfolio Weight saved in {save_path_sub}')

    # # Graphs & Tables:
    # opt_res = OptRes(
    #     conf=conf,
    #     ir1=ir1,
    #     close_adj=close_adj_path,
    #     idx_cons=idx_cons_path,
    #     res_path=res_path,
    #     tc=trade_cost)
    # if opt_res.load_dat() == 0:
    #     print(f"Dir not found `{opt_res.path}`")
    # else:
    #     opt_res.figure_historical_result()
    #     opt_res.figure_portfolio_weight()
    #     opt_res.figure_turnover()
    #     opt_res.figure_opt_time()
    #     opt_res.figure_risk_a_result()

    return


def portfolio_optimize(all_args, telling=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optimize
    :param all_args:
        tradedates: tradedates when you optimize your portfolio
        beta_expo: beta exposure, columns={betas}, index={(tradedate, stockcode)}
        beta_cnstr: beta constraint, columns=[H, L], index={betas}
        ind_cons: index constituent weight
        fct_cov: adjusted covariance matrix of pure factor return, columns={betas}, index={(tradedate, betas)}
        stk_rsk: stock specific risk, sqrt(variance), columns={stockcode}, index={tradedates}
        alpha: alpha to maximize, columns={stockcode}, index={tradedates}
        args:
            mkt_type: market index type (CSI500, or CSI300)
            N: maximum pool number, e.g., select X stocks from 1000 candidate, with the largest alpha value
            D: maximum turnover rate, less than 200(%)
            B: minimum index-constituent weight-sum (%)
            E: maximum excess holding weight (part of)
            G: gamma, risk aversion coefficient
            S: maximum risk exposure matrix
            wei_tole: weight tolerance
            opt_verbose: show solver process
            desc: msg in progress bar
            pos: position of progress bar
    :param telling: show pool stock number and determent process for each day iteration
    :return:
        holding_weight: holding weight, columns={stockcode}, index={tradedate}
        optimize_iter_info: optimize information, columns={infos}, index={tradedate}

    """

    #
    tradedates, beta_expo, beta_cnstr, ind_cons, fct_cov, stk_rsk, alpha, args = all_args
    mkt_type, N, D, B, E, G, S, wei_tole, opt_verbose, desc, pos = args

    def get_stk_alpha(dat_td) -> set:
        if N < np.inf:
            res = set(dat_td.rank(ascending=False).sort_values().index[:N])
        else:
            res = set(dat_td.dropna().index)
        return res

    holding_weight: pd.DataFrame = pd.DataFrame()
    df_lst_w: pd.DataFrame = pd.DataFrame()
    optimize_iter_info: pd.DataFrame = pd.DataFrame()
    w_lst = None

    # start_time = time.time()
    # td = '2021-12-31'
    loop_bar = tqdm(range(len(tradedates)), ncols=99, desc=desc, delay=0.01, position=pos, ascii=False)
    #
    for cur_td in loop_bar:
        td = tradedates.iloc[cur_td].strftime('%Y-%m-%d')
        #
        use_sigma = (G > 0) or (S < np.inf)  # Specific Risk
        # sigma: pd.DataFrame = get_risk_matrix(path_sigma, td, max_backward=5, notify=False)

        # Asset pool accessible: beta exposure + risk matrix, ^ alpha, ^ ind_cons
        stk_alpha = get_stk_alpha(alpha.loc[td])
        stk_index = set(ind_cons.loc[td].dropna().index)
        stk_beta = set(beta_expo.loc[td].index)
        stk_sigma = set(stk_rsk.loc[td].dropna().index) if use_sigma else None
        ls_pool, ls_base, sp_info = get_accessible_stk(i=stk_index, a=stk_alpha, b=stk_beta, s=stk_sigma)
        ls_clear = list(set(df_lst_w.index).difference(ls_pool))  # 上期持有 组合资产未覆盖
        if telling:
            print(f"\n\t{mkt_type} - alpha({len(stk_alpha)}) = {sp_info['#i_a']} [{','.join(sp_info['i_a'])}]")
            print(f"\t{mkt_type} - beta({len(stk_beta)}) ^ sigma({len(stk_sigma)})"
                  f" = {sp_info['#i_b(s)']} [{','.join(sp_info['i_b(s)'])}]" if use_sigma else
                  f"\t{mkt_type} - beta({len(stk_beta)}) = {sp_info['#i_b(s)']} [{','.join(sp_info['i_b(s)'])}]")
            print(f'\talpha exposed ({len(ls_pool)}/{len(stk_alpha)})')
            print(f'\t{mkt_type.lower()} exposed ({len(ls_base)}/{len(stk_index)})')
            print(f'\tformer holdings not exposed ({len(ls_clear)}/{len(df_lst_w)}) [{",".join(ls_clear)}]')

        wb = ind_cons.loc[td, ls_base]
        wb /= wb.sum()  # part of index-constituent are not exposed to beta factors; (not) treat them as zero-exposure.
        wb_ls_pool = pd.Series(ind_cons.loc[td, ls_base], index=ls_pool).fillna(0)  # cons w, index broadcast as pool
        a = alpha.loc[td, ls_pool]  # alpha
        mat_F = fct_cov.loc[td]
        mat_F = mat_F.dropna(how='all').dropna(axis=1, how='all')
        mat_F = mat_F.loc[[x for x in mat_F.index if x != 'country'], [x for x in mat_F.columns if x != 'country']]
        srs_D = stk_rsk.loc[td, ls_pool]
        xf = beta_expo.loc[td].loc[ls_pool].dropna(axis=1)
        xf = xf[xf.columns.intersection(mat_F.index)]

        def wtf(a, mat_F, srs_D, xf, df_lst_w, w_lst, G, ishow=False):  # TODO: out
            a1 = a.values.reshape(1, -1)
            mf = np.matrix(mat_F)
            sd = np.matrix(srs_D ** 2)
            # f_del = beta_expo.loc[td].loc[ls_base].dropna(axis=1).T @ wb  # - f_del_overflow
            f_del = beta_expo.loc[td].loc[ls_base].dropna(axis=1).T @ wb  # - f_del_overflow
            f_del = f_del.loc[xf.columns]
            fl = (f_del + beta_cnstr.loc[f_del.index, 'L']).dropna()
            fh = (f_del + beta_cnstr.loc[f_del.index, 'H']).dropna()

            D_offset = df_lst_w.loc[ls_clear].abs().sum().values[0] if len(ls_clear) > 0 else 0

            # Constraints
            wN = len(ls_pool)
            ww = cp.Variable((wN, 1), nonneg=True)
            opt_cnstr = OptCnstr()

            # (1) sum 1
            opt_cnstr.sum_bound(ww, e=np.ones([1, wN]), down=None, up=1)

            # (2) cons component percentage
            opt_cnstr.sum_bound(ww, e=(1 - pd.Series(wb, index=ls_pool).isna()).values.reshape(1, -1), down=B, up=None)

            # (3) cons weight deviation
            offset = wb_ls_pool.apply(lambda _: max(E, _ / 2))  # max(E, 0.5w) as offset
            down = (wb_ls_pool - offset).values.reshape(-1, 1)
            up = (wb_ls_pool + offset).values.reshape(-1, 1)
            opt_cnstr.uni_bound(ww, down=down, up=up)
            del offset, down, up

            # (4)(5) beta exposure
            opt_cnstr.sum_bound(
                ww,
                e=xf[fl.index].values.T,
                down=fl.values.reshape(-1, 1),
                up=fh.values.reshape(-1, 1)
            )

            # (6) turnover constraint
            if len(df_lst_w) > 0:  # not first optimization
                w_lst = df_lst_w.reindex_like(pd.DataFrame(index=ls_pool, columns=df_lst_w.columns)).fillna(0)
                w_lst = w_lst.values
                d = D - D_offset
                opt_cnstr.norm_bound(ww, w0=w_lst, d=d, L=1)
            else:  # first iteration, holding 0
                w_lst = np.zeros([len(ls_pool), 1]) if w_lst is None else w_lst  # former holding

            # (7) specific risk
            # if use_sigma:
            wbp = wb_ls_pool.values.reshape(-1, 1)
            x = ww - wbp
            risk = cp.quad_form(xf.values.T @ x, mf) + sd @ (x ** 2)
            if S < np.inf:
                opt_cnstr.add_constraints(risk <= S)
            # G = 20000
            objective = cp.Maximize(a1 @ ww - G * risk) if G > 0 else cp.Maximize(a1 @ ww)
            # S = 1e-7
            # opt_cnstr.add_constraints(risk <= S)
            # objective = cp.Maximize(a @ w)
            #
            # else:
            #     objective = cp.Maximize(a @ w)

            # Solve
            constraints = opt_cnstr.get_constraints()
            prob = cp.Problem(objective, constraints)
            if ishow:
                result = prob.solve(verbose=True, solver='ECOS', max_iters=1000)
                # result = prob.solve(verbose=True, solver='SCS', max_iters=1000)
            else:
                result = prob.solve(verbose=opt_verbose, solver='ECOS', max_iters=1000)
            if prob.status == 'optimal_inaccurate':
                result = prob.solve(verbose=opt_verbose, solver='ECOS', max_iters=10000)
                #
            if prob.status == 'optimal_inaccurate':
                result = prob.solve(verbose=opt_verbose, solver='SCS', max_iters=1000)
                #
            if prob.status == 'optimal':
                w1 = ww.value.copy()
                w1[w1 < wei_tole] = 0
                w1 /= w1.sum()
                df_w = pd.DataFrame(w1, index=ls_pool, columns=[td])
                turnover = np.abs(w_lst - df_w.values).sum() + D_offset
                hdn = (df_w.values > 0).sum()
                df_lst_w = df_w.replace(0, np.nan).dropna()
                if ishow:  # Graph:
                    df_lst_w.sort_values(td, ascending=False).reset_index(drop=True).plot(
                        title=f'{td}, $\gamma={G}$, res=' + f'{result:.3f} - {risk.value[0, 0] * G:.3f}')
                    # title=f'{td}, $\S={S}$, res=' + f'{result:.3f} - {risk.value[0,0] * G:.3f}')
                    plt.tight_layout()
                    plt.show()
            else:
                raise Exception(f'{prob.status} problem')
                # turnover = 0
                # print(f'.{prob.status} problem, portfolio ingredient unchanged')
                # if len(lst_w) > 0:
                #     lst_w.columns = [td]
            iter_info = {
                '#alpha^beta': len(ls_pool), '#index^beta': len(ls_base), '#index': len(stk_index),
                'turnover': turnover, 'holding': hdn,
                'risk': risk.value[0, 0] * G, 'opt0': result, 'opt1': (a1 @ w1)[0, 0],
                'solver': prob.solver_stats.solver_name, 'status': prob.status, 'stime': prob.solver_stats.solve_time,
            }

            #
            return iter_info, f_del, df_lst_w, w_lst

        #
        iter_info, f_del, df_lst_w, w_lst = wtf(a, mat_F, srs_D, xf, df_lst_w, w_lst, G=G, ishow=False)

        #  Update optimize iteration information
        holding_weight = pd.concat([holding_weight, df_lst_w.T])
        iter_info = iter_info | {'# cons w/o alpha': sp_info['#i_a'],
                                 '# cons w/o beta(sigma)': sp_info['#i_b(s)'],
                                 'cons w/o alpha': ', '.join(sp_info['i_a']),
                                 'cons w/o beta(sigma)': ', '.join(sp_info['i_b(s)'])}
        iter_info = iter_info | f_del.to_dict()
        optimize_iter_info[td] = pd.Series(iter_info)
        # progressbar(cur_td + 1, len(tradedates), msg=f' {td} turnover={turnover:.3f} #stk={hdn}', stt=start_time)
    print()

    return holding_weight, optimize_iter_info


def portfolio_statistics_from_weight(weight, cost_rate, all_ret, save_path=None):
    """对持仓计算结果"""
    res = pd.DataFrame(index=weight.index)
    res['NStocks'] = (weight.abs() > 0).sum(axis=1)
    res['Turnover'] = weight.diff().abs().sum(axis=1)
    res['Return'] = (all_ret.reindex_like(weight) * weight).sum(axis=1)
    res['Return_wc'] = res['Return'] - res['Turnover'] * cost_rate
    res['Wealth(cumsum)'] = res['Return'].cumsum().add(1)
    res['Wealth_wc(cumsum)'] = res['Return_wc'].cumsum().add(1)
    res['Wealth(cumprod)'] = res['Return'].add(1).cumprod()
    res['Wealth_wc(cumprod)'] = res['Return_wc'].add(1).cumprod()
    if save_path:
        res.to_csv(save_path)
    return res


def second2clock(x: int):
    if x < 3600:
        return f"{(x // 60):02d}:{x % 60:02d}"
    elif x < 3600 * 24:
        return f"{(x // 3600):02d}:{(x // 60) % 60:02d}:{x % 60:02d}"
    else:
        d = x // (24 * 3600)
        x = x % (24 * 3600)
        return f"{d}d {(x // 3600):02d}:{(x // 60) % 60:02d}:{x % 60:02d}"


def table_save_safe(df: pd.DataFrame, tgt: str, kind=None, notify=False, **kwargs):
    """
    安全更新已有表格（当tgt在磁盘中被打开，5秒后再次尝试存储）
    :param df: 表格
    :param tgt: 目标地址
    :param kind: 文件类型，暂时仅支持csv
    :param notify: 是否
    :return:
    """
    kind = tgt.rsplit(sep='.', maxsplit=1)[-1] if kind is None else kind

    if kind == 'csv':
        func = df.to_csv
    elif kind == 'xlsx':
        func = df.to_excel
    elif kind == 'pkl':
        func = df.to_pickle
    elif kind == 'h5':
        if 'key' in kwargs:
            hdf_k = kwargs['key']
        elif 'k' in kwargs:
            hdf_k = kwargs['k']
        else:
            raise Exception('Save FileType hdf but key not given in table_save_safe')

        def func():
            df.to_hdf(tgt, key=hdf_k)
    else:
        raise ValueError(f'Save table filetype `{kind}` not supported.')

    try:
        func(tgt)
    except PermissionError:
        print(f'Permission Error: saving `{tgt}`, retry in 5 seconds...')
        time.sleep(5)
        table_save_safe(df, tgt, kind)
    finally:
        if notify:
            print(f'{df.shape} saved in `{tgt}`.')


class OptCnstr(object):
    """Manage constraints in optimization problem"""

    def __init__(self):
        self.cnstr = []

    def add_constraints(self, o):
        self.cnstr.append(o)

    def get_constraints(self) -> list:
        return self.cnstr

    def risk_bound(self, w, w0, Sigma, up):
        """
        组合跟踪误差
        :param w: N * 1 cvxpy variable
        :param w0: N * 1 numpy array
        :param Sigma: N * N numpy array
        :param up: 1 * 1 numpy array
        """
        wd = w - w0
        self.cnstr.append(wd.T @ Sigma @ wd - up <= 0)

    def norm_bound(self, w, w0, d, L=1):
        """
        范数不等式(default L1)
        :param w: N * 1 cvxpy variable
        :param w0: N * 1 numpy array
        :param d: float
        :param L: L()-norm
        """
        self.cnstr.append(cp.norm(w - w0, L) - d <= 0)

    def uni_bound(self, w, down=None, up=None):
        """
        每个变量的上下界
        :param w: N * 1 cvxpy variable
        :param down: N * 1 numpy array
        :param up: N * 1 numpy array
        """
        if down is not None:
            self._uni_down_bound(w, down)
        if up is not None:
            self._uni_up_bound(w, up)

    def sum_bound(self, w, e, down=None, up=None):
        """
        股权重加权和在上下界内
        :param w: N * 1 cvxpy variable
        :param e: X * N numpy array
        :param down: X * 1 numpy array
        :param up: X * 1 numpy array
        """
        if down is not None:
            self._sum_down_bound(w, e, down)
        if up is not None:
            self._sum_up_bound(w, e, up)

    def _uni_down_bound(self, w, bar):
        self.cnstr.append(bar - w <= 0)

    def _uni_up_bound(self, w, bar):
        self.cnstr.append(w - bar <= 0)

    def _sum_down_bound(self, w, e, bar):
        self.cnstr.append(bar - e @ w <= 0)

    def _sum_up_bound(self, w, e, bar):
        self.cnstr.append(e @ w - bar <= 0)


class OptRes(object):

    def __init__(self, conf: dict, ir1: pd.Series, close_adj, idx_cons, res_path, tc=None):

        self.conf: dict = conf
        self.close_adj_path: str = close_adj  # conf['closeAdj']
        self.idx_cons_path: str = idx_cons  # conf['idx_constituent']
        self.res_path: str = res_path  # conf['factorsres_path']
        self.trade_cost: float = float(tc) if isinstance(tc, str) else tc  # conf['cr']

        self.alpha_name: str = ir1['alpha_name']
        self.mkt_type: str = ir1['mkt_type']
        self.suffix: str = info2suffix(ir1)
        self.path: str = f"{self.res_path}OptResWeekly[{self.mkt_type}]{self.alpha_name}/{self.suffix}/"

        self.W: pd.DataFrame = pd.DataFrame()
        self.bd: pd.Timestamp = pd.Timestamp(2000, 1, 1)
        self.ed: pd.Timestamp = pd.Timestamp(2099, 12, 31)
        self.views: list = list()
        self.rtn_next: pd.DataFrame = pd.DataFrame()
        self.ind_cons_w: pd.DataFrame = pd.DataFrame()
        self.port: Portfolio = Portfolio()
        self.e_port: Portfolio = Portfolio()
        self.opt_info: pd.DataFrame = pd.DataFrame()
        self.Return: pd.DataFrame = pd.DataFrame()  # returns next week
        self.Wealth: pd.DataFrame = pd.DataFrame()

    def load_dat(self) -> int:
        if (not os.path.exists(self.path)) or (len(os.listdir(self.path)) == 0):
            return 0
        self.W = pd.read_csv(self.path + f"portfolio_weight_{self.suffix}.csv", index_col=0, parse_dates=True)
        self.bd = self.W.index[0].strftime('%Y-%m-%d')
        self.ed = self.W.index[-1].strftime('%Y-%m-%d')
        self.views = list(self.W.index)
        self.port: Portfolio = Portfolio(w=self.W)
        self.opt_info = pd.read_excel(self.path + f'opt_info_{self.suffix}.xlsx', index_col=0, parse_dates=True)
        # Weekly Close Return
        close_adj = pd.read_csv(self.close_adj_path, index_col=0, parse_dates=True)
        close_adj_w = close_adj.loc[self.views]
        self.rtn_next = close_adj_w.pct_change().shift(-1)
        # Index Constitution Weight
        self.ind_cons_w = get_index_constitution(self.conf, self.bd, self.ed, self.mkt_type, True)
        self.ind_cons_w = self.ind_cons_w.loc[self.views]
        # Excess Result
        self.e_port = Portfolio(w=df_union_sub(df0=self.W, df1=self.ind_cons_w))
        return 1

    def figure_portfolio_weight(self):
        self.port.plot_weight_hist(
            path=self.path + 'figure_weight_hist_' + self.suffix + '.png')
        self.e_port.plot_weight_hist(
            path=self.path + 'figure_weight_hist_' + self.suffix + '[E].png')
        self.port.plot_port_asset_num(
            path=self.path + 'figure_portfolio_size_' + self.suffix + '.png',
            rw={'W': 1, '4W': 4, '52W': 52})
        self.port.plot_asset_weight(
            path=self.path + 'figure_portfolio_weight_' + self.suffix + '.png')

    def figure_historical_result(self):

        # Half Year Statistics (without cost)
        self.port.cal_panel_result(cr=0, ret=self.rtn_next)
        self.port.cal_half_year_stat(wc=False, freq='W', lang='CH')
        self.port.get_half_year_stat(wc=False, path=self.path + 'table_half_year_stat_' + self.suffix + '.xlsx')
        self.e_port.cal_panel_result(cr=0, ret=self.rtn_next)
        self.e_port.cal_half_year_stat(wc=False, freq='W', lang='CH')
        self.e_port.get_half_year_stat(wc=False, path=self.path + 'table_half_year_stat_' + self.suffix + '[E].xlsx')
        if isinstance(self.trade_cost, float):
            self.port.cal_panel_result(cr=self.trade_cost, ret=self.rtn_next)
            self.port.cal_half_year_stat(wc=True, freq='W', lang='CH')
            _path = self.path + 'table_half_year_stat_' + self.suffix + f'(cr={self.trade_cost}).xlsx'
            self.port.get_half_year_stat(wc=True, path=_path)
            # TODO: turnover cost from long side - in Portfolio, add baseline
            self.e_port.cal_panel_result(cr=self.trade_cost, ret=self.rtn_next)
            self.e_port.cal_half_year_stat(wc=True, freq='W', lang='CH')
            _path = self.path + 'table_half_year_stat_' + self.suffix + f'(cr={self.trade_cost})[E].xlsx'
            self.e_port.get_half_year_stat(wc=True, path=_path)

        # Table Return
        self.Return['portfolio'] = (self.rtn_next.reindex_like(self.W) * self.W).sum(axis=1)
        self.Return[self.mkt_type] = (self.rtn_next.reindex_like(self.ind_cons_w) * self.ind_cons_w).sum(axis=1)
        self.Return.to_excel(self.path + 'table_return_' + self.suffix + '.xlsx')

        # # Table & Graph Holding Weight (2021-12-31)
        # tmp = pd.concat([self.W.loc['2021-12-31'].dropna().rename('port'),
        #                  self.ind_cons_w.loc['2021-12-31'].dropna().rename('cons')],
        #                 axis=1)
        # tmp = tmp.sort_values(['port', 'cons'], ascending=False)
        # tmp['diff'] = tmp.iloc[:, 0] - tmp.iloc[:, 1]
        # tmp.to_excel(self.path + 'table_holding_diff_20211231' + self.suffix + '.xlsx')
        # tmp['diff'].dropna().reset_index(drop=True).plot(style='o', title=f'Weight Difference to {self.mkt_type}')
        # plt.tight_layout()
        # plt.savefig(self.path + 'figure_holding_diff_20211231_' + self.suffix + '.png')
        # plt.close()
        # tmp['port'].dropna().reset_index(drop=True).plot(title='Holding Weight')
        # plt.tight_layout()
        # plt.savefig(self.path + 'figure_holding_weight_20211231_' + self.suffix + '.png')
        # plt.close()
        # del tmp

        # Table & Graph Wealth
        self.Wealth = self.Return.cumsum()
        self.Wealth['Excess'] = self.Wealth.iloc[:, 0] - self.Wealth.iloc[:, 1]
        self.Wealth = self.Wealth.add(1)
        self.Wealth.to_excel(self.path + 'table_result_wealth_' + self.suffix + '.xlsx')
        self.Wealth.plot(title='Wealth (1 week ahead)')
        plt.tight_layout()
        plt.savefig(self.path + 'figure_result_wealth_' + self.suffix + '.png')
        plt.close()

    def figure_turnover(self):
        self.port.plot_turnover(
            path=self.path + 'figure_turnover_' + self.suffix + '.png',
            ishow=False)

    def figure_opt_time(self):
        sr = self.opt_info['stime']
        sr.plot(title=f'Solve Time, M={sr.mean():.3f}s')
        plt.tight_layout()
        plt.savefig(self.path + 'figure_solve_time_' + self.suffix + '.png')
        plt.close()

    def figure_risk_a_result(self):
        df = self.opt_info[['risk', 'opt0']]
        # df['risk'] = df['risk'].apply(lambda x: x if isinstance(x, float) else eval(x)[0][0])
        df['alpha'] = df['risk'] + df['opt0']
        _title = f"Result({df['opt0'].mean():.3f}): Alpha({df['alpha'].mean():.3f}) - Risk({df['risk'].mean():.3f})"
        df.rename(columns={'opt0': 'alpha - risk'}).plot(title=_title)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_alpha_risk_' + self.suffix + '.png')
        plt.close()


class Portfolio(object):
    r"""
    w: DataFrame of shape (n_views, n_assets), holding weight row sum 1
    cr: float, cost rate

    ------
    port = Portfolio(w,[ cr, ret])
    port.cal_panel_result(cr: float, ret: pd.DataFrame)
    port.cal_half_year_stat(wc: bool)

    """

    def __init__(self, w: pd.DataFrame = None, **kwargs):
        self.w_2d = w
        self.views = w.index.to_list() if isinstance(w, pd.DataFrame) else list()

        self.panel: pd.DataFrame = pd.DataFrame(
            index=self.views, columns=['NStocks', 'Turnover', 'Return', 'Return_wc',
                                       'Wealth(cumsum)', 'Wealth_wc(cumsum)',
                                       'Wealth(cumprod)', 'Wealth_wc(cumprod)'])
        self.cost_rate = None
        if ('cr' in kwargs) and ('ret' in kwargs):
            self.cal_panel_result(cr=kwargs['cr'], ret=kwargs['ret'])

        self.stat = {True: pd.DataFrame(), False: pd.DataFrame()}  # half year statistics
        self.mdd = {}

    def cal_panel_result(self, cr: float, ret: pd.DataFrame):
        """From hist ret and hold weight, cal panel: NStocks, Turnover, Return(), Wealth()"""
        self.cost_rate = cr
        self.panel = portfolio_statistics_from_weight(weight=self.w_2d, cost_rate=cr, all_ret=ret)

    def cal_half_year_stat(self, wc=False, freq='D', lang='EN'):
        """cal half year statistics from `panel`"""
        if self.panel.dropna().__len__() == 0:
            raise Exception('Empty self.panel')
        col = 'Return_wc' if wc else 'Return'
        self.stat[wc] = cal_result_stat(self.panel[[col]], freq=freq, lang=lang)

    def get_position_weight(self, path=None) -> pd.DataFrame:
        if path is not None:
            table_save_safe(df=self.w_2d, tgt=path)
        return self.w_2d.copy()

    def get_panel_result(self, path=None) -> pd.DataFrame:
        if path is not None:
            table_save_safe(df=self.panel, tgt=path)
        return self.panel.copy()

    def get_half_year_stat(self, wc=False, path=None) -> pd.DataFrame:
        if wc not in self.stat.keys():
            print('Calculate half-year statistics before get_half_year_stat...')
            self.cal_half_year_stat(wc=wc)
        if path is not None:
            table_save_safe(df=self.stat[wc], tgt=path)
        return self.stat[wc]

    def plot_turnover(self, ishow, path, title='Turnover'):
        if self.panel is None:
            raise Exception('Calculate panel result before plot turnover!')

        sr = self.panel['Turnover']
        sr.plot(figsize=(9, 5), grid=True, title=title + f', M={sr.mean() * 100:.2f}%')
        plt.tight_layout()
        plt.savefig(path)
        if ishow:
            plt.show()
        plt.close()

    def plot_cumulative_returns(self, ishow, path, kind='cumsum', title=None):
        title = f'Portfolio Absolute Result ({kind})' if title is None else title
        self.panel[[f'Wealth({kind})', f'Wealth_wc({kind})']].plot(figsize=(9, 5), grid=True, title=title)
        plt.tight_layout()
        plt.savefig(path)
        if ishow:
            plt.plot()
        plt.close()

    def plot_max_drawdown(self, ishow, path, wc=False, kind='cumsum', title=None):
        col = f'Wealth_wc({kind})' if wc else f'Wealth({kind})'
        title = f'MaxDrawdown {col}' if title is None else title
        df = self.panel[col].copy()
        df = df + 1 if df.iloc[0] < .6 else df
        cal_sr_max_drawdown(df=df, ishow=ishow, title=title, save_path=path, kind=kind)

    def plot_weight_hist(self, path, ishow=False, title='Weight Distribution'):
        plt.hist(self.w_2d.values.flatten(), bins=100)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path)
        if ishow:
            plt.show()
        plt.close()

    def plot_port_asset_num(self, path, ishow=False, rw=None):
        if rw is None:
            rw = {'D': 1, '20D': 20, '60D': 60}
        n_stk = self.panel['NStocks']
        tmp = pd.DataFrame()
        for k, v in rw.items():
            tmp[k] = n_stk.rolling(v).mean()
        tmp.plot(title=f'Portfolio Asset Numbers (rolling mean), M={n_stk.mean():.2f}', linewidth=2)
        plt.tight_layout()
        plt.savefig(path)
        if ishow:
            plt.show()
        plt.close()

    def plot_asset_weight(self, path, ishow=False, title='Asset Weight'):
        tmp = pd.DataFrame()
        tmp['w-MAX'] = self.w_2d.max(axis=1)
        tmp['w-MEDIAN'] = self.w_2d.median(axis=1)
        tmp['w-AVERAGE'] = self.w_2d.mean(axis=1)
        tmp.plot(title=title, linewidth=2)
        plt.tight_layout()
        plt.savefig(path)
        if ishow:
            plt.show()
        plt.close()

    def get_stock_number(self) -> pd.Series:
        return self.panel['NStocks'].copy()

    def get_turnover(self) -> pd.Series:
        return self.panel['Turnover'].copy()

    def get_daily_ret(self, wc=False) -> pd.Series:
        return self.panel['Return_wc' if wc else 'Return'].copy()

    def get_wealth(self, wc=False, kind='cumsum') -> pd.Series:
        return self.panel[f'Wealth_wc({kind})' if wc else f'Wealth({kind})'].copy()


if __name__ == '__main__':
    main()
