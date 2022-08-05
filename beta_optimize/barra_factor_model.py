"""
(created by swmao on July 4th)

目前每次更新时会计算504+X的纯因子收益
1）从远程数据库，获取已有最新日期
2）在最新日期 ，往前推2h=504天，获取504+X天 调整后风格暴露，纯因子收益；资产收益，资产市值
3）对待更新的日期，获取 资产收益、资产市值、原始风格暴露，计算 调整过的风格暴露，纯因子收益
4）合并 纯因子收益，输入MFM，进行>504长度的调整，得到NW+Eigen+VRA的因子协方差
5）合并 纯因子收益，调整过的风格暴露，自查年收益，资产市值，输入SRR，进行>504长度的调整，
得到NW+SM+SH+VRA的资产特异性波动（标准差）

Config:
- begin_date
- end_date
- data_path
- ipo_await
- access_target
- mysql_engine

Step1: Calculate pure factor return
------
Cache table in local disk is optional;
Once cached, shall be manually removed;

`Y (T+1 asset return) ~ X (T+1 beta return) @ B (T0 beta exposure)`
解释：上传的计算结果中，日期标记为T+1. 例如7月29日，指的是，
    7月28日收盘-7月29日收盘的资产收益、纯因子收益，对
    7月28日（7月29日10:00更新）的因子暴露（即因子暴露后移1天）回归。
那么在周末（7月30日、31日）进行组合优化决定
    8月1日持仓时，使用
    7月29的个股特异方差、因子协方差，以及
    7月29日(何时更新?)的beta暴露和alpha值
    （后两者不参与纯因子收益率和风险估计）。

Input:
- beta exposure (country + style + industry)
- close, daily close return

Output:
- barra_panel, `barra_exposure_orthogonal`
- barra_fval, `barra_pure_factor_return`
- barra_omega


Step2: Calculate adjusted risk matrix
------
Try to access input from local cache;
Then amend unfulfilled views from remote;
Then compute unfulfilled views left (Step1);

Input:
- pure factor return
- orthogonal beta exposure
- (asset specific return) or (asset return)
- asset market value

Output:
- factor return covariance
    shape (n_views * n_features, n_features + 2)
- asset specific risk
    shape (n_views, n_assets) -> (n_views * n_assets, 3)


Result: Table Updated in Database
------
- intern.barra_pure_factor_return
- intern.barra_exposure_orthogonal
- intern.barra_factor_cov_nw_eigen_vra
- intern.barra_specific_risk_nw_sm_sh_vra

"""
import time
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sqlalchemy import create_engine
from typing import List
from tqdm import tqdm
from typing import Dict, Tuple
from sqlalchemy.dialects.mysql import DOUBLE, VARCHAR, DATE
import warnings
import matplotlib.pyplot as plt
import seaborn

warnings.simplefilter("ignore")

seaborn.set_style("darkgrid")
plt.rc("figure", figsize=(9, 5))
plt.rc("savefig", dpi=90)
plt.rc("font", size=12)
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"
# plt.rc("font", family="sans-serif")

_PATH = '/home/swmao/'  # 修改这个地址！


def main():

    # 配置信息（修改这里）
    conf = {
        # 开始日期, 'NA'若由数据库已存在的最新日期推测
        'begin_date': 'NA',
        # 结束日期
        'end_date': '2099-12-31',
        # 频率 d/w/m, TODO 不能再改，多种频率会存在同一个数据表中
        'freq': 'd',
        # 排除上市不足（）个自然日的新股
        # 'ipo_await': 90,
        # 选用的Barra因子名（与数据库一致）
        'ls_style': [
            'size', 'beta', 'momentum',
            'residual_volatility', 'non_linear_size',
            'book_to_price_ratio', 'liquidity',
            'earnings_yield', 'growth', 'leverage',
        ],
        # 本地缓存目录
        'data_path': f'{_PATH}/cache0705/',  # '/home/swmao/cache0705/',
        # 需要获取的数据库表格信息
        'access_target': f"{_PATH}/access_target.xlsx",
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
    }

    # 本地缓存目录
    os.makedirs(conf['data_path'], exist_ok=True)
    os.makedirs(conf['data_path'] + 'barra_panel/', exist_ok=True)
    os.makedirs(conf['data_path'] + 'barra_omega/', exist_ok=True)
    os.makedirs(conf['data_path'] + 'barra_fval/', exist_ok=True)

    # 查询数据库表格最新日期的对象ND；conf中begin_date指定为NA时由最后一个表推断
    ND = NewestDate(conf['mysql_engine'])
    if conf['begin_date'] == 'NA':
        conf['begin_date'] = ND.infer_begin_date()

    # (1) Barra多因子模型
    bfm = BarraFM(conf=conf, CR=CsvReader(conf), load_remote=False)
    # 一次性将begin_date到end_date的style暴露读到本地；可不执行。
    bfm.cache_all_style_exposure(continuous_date=(conf['freq'] == 'd'))
    # 计算纯因子收益cache=False时不保留本地缓存，但会尝试从本地读取缓存。
    bfm.cal_pure_return_by_time(cache=True)
    # 上传到服务器：a)纯因子收益b)正交的风格暴露
    bfm.upload_results(
        eng=conf['mysql_engine']['engine4'],
        how='append',
        td_1=ND.newest_date(kw='barra_pure_factor_return')
    )
    # 内存清理，delete多因子模型，保留下一步所需
    factor_ret = bfm.get_factor_return()
    asset_ret = bfm.get_asset_return()
    barra_exposure = bfm.get_expo_panel()
    views = bfm.get_views()
    del bfm

    # (2) 纯因子收益 协方差估计与调整
    mfm = MFM(fr=factor_ret)
    mfm.newey_west_adj_by_time()
    mfm.save_factor_covariance(conf['data_path'], level='NW')
    mfm.eigen_risk_adj_by_time()
    mfm.save_factor_covariance(conf['data_path'], level='Eigen')
    mfm.vol_regime_adj_by_time()
    mfm.save_factor_covariance(conf['data_path'], level='VRA')
    # mfm.upload_adjusted_covariance(eng=conf['mysql_engine']['engine4'], level='NW', how='append')
    # mfm.upload_adjusted_covariance(eng=conf['mysql_engine']['engine4'], level='Eigen', how='append')
    mfm.upload_adjusted_covariance(
        eng=conf['mysql_engine']['engine4'],
        level='VRA',
        how='append',
        td_1=ND.newest_date(kw='barra_factor_cov_nw_eigen_vra')
    )
    del mfm

    # (3) 个股特异性波动 估计与调整
    srr = SRR(
        sr=asset_ret,
        fr=factor_ret,
        expo=barra_exposure,
        mv=load_asset_marketvalue(
            CR=CsvReader(conf),
            bd=views[0],
            ed=views[-1],
            views=None)  # TODO: not daily
    )
    srr.specific_return_by_time()  # Specific Risk
    srr.newey_west_adj_by_time()  # New-West Adjustment
    srr.save_vol_regime_adj_risk(conf['data_path'], level='Raw')
    srr.save_vol_regime_adj_risk(conf['data_path'], level='NW')
    srr.struct_mod_adj_by_time()  # Structural Model Adjustment
    srr.save_vol_regime_adj_risk(conf['data_path'], level='SM')
    srr.bayesian_shrink_by_time()  # Bayesian Shrinkage Adjustment
    srr.save_vol_regime_adj_risk(conf['data_path'], level='SH')
    srr.vol_regime_adj_by_time()  # Volatility Regime Adjustment
    srr.save_vol_regime_adj_risk(conf['data_path'], level='VRA')
    # srr.upload_asset_specific_risk(eng=conf['mysql_engine']['engine4'], level='Raw', how='append')
    # srr.upload_asset_specific_risk(eng=conf['mysql_engine']['engine4'], level='NW', how='append')
    # srr.upload_asset_specific_risk(eng=conf['mysql_engine']['engine4'], level='SM', how='append')
    # srr.upload_asset_specific_risk(eng=conf['mysql_engine']['engine4'], level='SH', how='append')
    srr.upload_asset_specific_risk(
        eng=conf['mysql_engine']['engine4'],
        level='VRA',
        how='append',
        td_1=ND.newest_date(kw='barra_specific_risk_nw_sm_sh_vra')
    )
    del srr

    return


class NewestDate(object):

    def __init__(self, mysql_eng: dict):
        self.mysql_engine = mysql_eng
        self.eng_info = self.mysql_engine['engine4']

    def newest_date(self, kw='barra_pure_factor_return', eng: dict = None, col='tradingdate') -> str:
        if eng is None:
            eng = self.eng_info
        query = f"SELECT {col} FROM {eng['dbname']}.{kw} ORDER BY {col} DESC LIMIT 1"
        engine = conn_mysql(eng)
        try:
            df = mysql_query(query, engine)
        except:
            return None
        if len(df) != 1:
            return None
            # raise Exception(f"Database {eng['dbname']}.{kw} not found")
        last_date: str = df.loc[0, col].strftime('%Y-%m-%d')
        return last_date

    def infer_begin_date(self, kw='barra_specific_risk_nw_sm_sh_vra') -> str:
        td_1 = self.newest_date(kw=kw)
        if td_1 is None:
            raise Exception(
                f"A begin date (now NA) must be given in config"
                f" since table {kw} doesn't exist")
        df = mysql_query(
            query=f"SELECT tradingdate FROM tdays_d"
                  f" WHERE tradingdate<='{td_1}'"
                  f" ORDER BY tradingdate DESC"
                  f" LIMIT 505",
            engine=conn_mysql(eng=self.mysql_engine['engine0'])
        )
        if len(df) != 505:
            raise Exception('')
        return df.loc[504, 'tradingdate'].strftime('%Y-%m-%d')


class CsvReader(object):

    def __init__(self, conf):
        self.data_path: str = conf['data_path']
        self.mysql_engine: dict = conf['mysql_engine']
        self.bd: str = conf['begin_date']
        self.ed: str = conf['end_date']
        self.acc_tgt = pd.read_excel(conf['access_target']).set_index('KEY')

    def get_engine(self, server_id: int) -> dict:
        """Get server engine info: dict."""
        server_id = str(server_id).replace('engine', '')
        return self.mysql_engine[f"engine{server_id}"]

    def get_info_sr(self, key: str) -> pd.Series:
        """Get target info matching key in acc_tgt `KEY` columns."""
        try:
            info = self.acc_tgt.loc[key]
        except KeyError:
            raise Exception(f'`{key}` not find in access target KEYs')
        return info

    def infer_info_sr(self, local_path: str) -> pd.Series:
        """Get target info from local cache path."""
        mask = (self.acc_tgt['CSV'] == local_path.rsplit('/', maxsplit=1)[-1])
        if mask.sum() == 0:
            raise Exception(f'Cannot infer target info from local path `{local_path}`')
        elif mask.sum() > 1:
            raise Exception(f'Inferred target info not unique for local path `{local_path}`, \n'
                            f'{self.acc_tgt[mask]}')
        info = self.acc_tgt[mask].iloc[0]
        return info

    def read_csv(self,
                 views=None,
                 info=None,
                 bd: str = None,
                 ed: str = None,
                 local_path: str = None,
                 d_type=object,
                 cache=True
                 ) -> pd.DataFrame:
        """
        Read one 2d csv table
            If bd~ed not fully covered,
            access remote & update local.
        :param views:
            List[str], tradingdate
        :param d_type:
            str, claim type of frame
        :param info:
            str, or Series of shape (11,), or None
        :param bd:
            str, data begin date
        :param ed:
            str, data end date
        :param local_path:
            str, where data csv saved
        :param cache:
            bool, True than update local cache
        :return:
            required data frame
        """

        # Remote table information
        if info is None:
            if local_path is None:
                raise Exception('read_csv no info or local path')
            if self.data_path in local_path:
                info = self.infer_info_sr(local_path)
        elif isinstance(info, str):
            info = self.get_info_sr(info)
        elif not isinstance(info, pd.Series):
            raise Exception(f'Invalid read csv info `{info}`')

        # Local csv cache path
        if local_path is None:
            local_path = self.data_path + info['CSV']

        # Index: bd ~ ed, type str
        if views is None:
            if bd is None:
                bd = self.bd
            if ed is None:
                ed = self.ed
        else:
            bd = views[0]
            ed = views[-1]
        if isinstance(bd, pd.Timestamp):
            bd = bd.strftime('%Y-%m-%d')
        if isinstance(ed, pd.Timestamp):
            ed = ed.strftime('%Y-%m-%d')

        # Connect remote engine
        eng = self.mysql_engine[f"engine{info['SERVER']}"]

        # Get csv table (local merge remote)
        if os.path.exists(local_path):
            df = pd.read_csv(local_path, index_col=0, parse_dates=True, dtype=d_type)
            T0 = len(df)
            if len(df) == 0:
                raise Exception(f'Empty file `{local_path}`')
            bd0, ed0 = df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d')

            if views is None:
                if bd < bd0:
                    # load remote, bd < bd0
                    df10 = load_remote_table(info=info, eng=eng, bd=bd, ed=bd0)
                    if len(df10) > 0:
                        if pd.to_datetime(df10.index[-1]) == pd.to_datetime(bd0):
                            df10 = df10.iloc[:-1]
                        if len(df10) > 0:
                            df = pd.concat([df10, df])
                        del df10
                if ed0 < ed:
                    # load remote, ed0 < ed
                    df01 = load_remote_table(info=info, eng=eng, bd=ed0, ed=ed)
                    if len(df01) > 0:
                        if pd.to_datetime(df01.index[0]) == pd.to_datetime(ed0):
                            df01 = df01.iloc[1:]
                        if len(df01) > 0:
                            df = pd.concat([df, df01])
                        del df01
            else:
                views_absent = [_ for _ in views if (pd.to_datetime(_) not in df.index)]
                if len(views_absent) > 0:
                    df_absent = load_remote_table(info=info, eng=eng, views=views_absent)
                    if len(df_absent) > 0:
                        df = pd.concat([df, df_absent]).sort_index()

            if cache and (T0 < len(df)):
                table_save_safe(df=df, tgt=local_path, kind='csv')

        else:  # entirely from remote
            if views is None:
                df = load_remote_table(info=info, eng=eng, bd=bd, ed=ed)
            else:
                df = load_remote_table(info=info, eng=eng, views=views)
            if cache:
                table_save_safe(df=df, tgt=local_path, kind='csv')

        return df.loc[pd.to_datetime(bd): pd.to_datetime(ed)]


class BarraFM(object):
    """Barra Factor Model"""

    def __init__(self, conf: dict, CR: CsvReader = None, load_remote=True):

        self.begin_date: str = conf['begin_date']
        self.end_date: str = conf['end_date']
        self.freq: str = conf['freq']
        self.data_path: str = conf['data_path']
        # self.ipo_await: int = conf['ipo_await']
        self.ls_style: List[str] = conf['ls_style']
        self.ls_indus: List[str] = []
        if CR is None:
            self.CR = CsvReader(conf)
        else:
            self.CR = CR

        # Load views: tradingdate with freq from begin_date to end_date
        self.views: List[str] = load_tradedate_view(eng=conf['mysql_engine']['engine0'],
                                                    bd=self.begin_date,
                                                    ed=self.end_date,
                                                    freq=self.freq)

        # Load asset returns, (close return, next period)
        if self.freq == 'd':
            self.asset_ret: pd.DataFrame = load_asset_return(
                CR=self.CR, bd=self.views[0], ed=self.views[-1], shifting=False)
        else:
            self.asset_ret: pd.DataFrame = load_asset_return(
                CR=self.CR, views=self.views, shifting=False)

        # Infer views (daily tradedate) & periods length
        self.views = self.asset_ret.index.to_series().apply(lambda x: x.strftime("%Y-%m-%d")).to_list()
        self.T: int = len(self.views)

        # Result: pure factor return from begin_date to end_date
        self.expo_panel: pd.DataFrame = pd.DataFrame()
        self.factor_ret: pd.DataFrame = pd.DataFrame()

        if load_remote:
            # Decide date range of Barra factor return & orthogonal exposure existed in database
            ND = NewestDate(conf['mysql_engine'])
            td_2 = self.views[0]  # 请求数据的开始日期
            td_1 = ND.newest_date(kw='barra_pure_factor_return')  # 服务器中最新日期
            td1 = min(td_1, self.views[-1])  # 请求数据的结束日期

            # Barra pure factor return, existed in remote database
            query = f"SELECT * FROM intern.barra_pure_factor_return" \
                    f" WHERE tradingdate>='{td_2}' AND tradingdate<='{td1}'"
            df = mysql_query(query, conn_mysql(eng=conf['mysql_engine']['engine4']))
            df['tradingdate'] = pd.to_datetime(df['tradingdate'])
            self.factor_ret = df.set_index('tradingdate')

            # Barra orthogonal factor exposure, existed in remote database
            query = f"""SELECT * FROM intern.barra_exposure_orthogonal WHERE tradingdate>='{td_2}' AND tradingdate<='{td1}'"""
            df = mysql_query(query, conn_mysql(eng=conf['mysql_engine']['engine4']))
            df['tradingdate'] = pd.to_datetime(df['tradingdate'])
            self.expo_panel = df.set_index(['tradingdate', 'stockcode'])

            # Industry dummies
            query = f"SELECT tradingdate,stockcode,industry_l1" \
                    f" FROM jeffdatabase.ind_citic_constituent" \
                    f" WHERE tradingdate>='{td_2}' AND tradingdate<='{td1}'"
            df = mysql_query(query, conn_mysql(eng=conf['mysql_engine']['engine0']))
            df['tradingdate'] = pd.to_datetime(df['tradingdate'])
            df = df.set_index(['tradingdate', 'stockcode'])
            df = pd.get_dummies(df)
            df.columns = pd.Index([f"ind_{_.rsplit('_', maxsplit=1)[-1].split('.')[0]}" for _ in
                                   df.columns])  # rename ind columns: ind_000001

            self.expo_panel = pd.concat([self.expo_panel, df], axis=1).dropna()

    def upload_results(self, eng: dict, how='insert', td_1=None):
        """"""

        # Upload pure factor return
        df = self.factor_ret
        if td_1 is not None:
            df = df[df.index > pd.to_datetime(td_1)]
        d_type_dict = {'tradingdate': DATE()} | {_: DOUBLE() for _ in df.columns}
        table_name = 'barra_pure_factor_return'
        print('\nUpload pure factor return of shape', df.shape, '...')
        if how == 'replace' or how == 'r':
            df.reset_index().to_sql(
                name=table_name,
                con=conn_mysql(eng),
                if_exists='replace',
                index=False,
                dtype=d_type_dict,
            )
        elif how == 'append' or how == 'a':
            df.reset_index().to_sql(
                name=table_name,
                con=conn_mysql(eng),
                if_exists='append',
                index=False,
                dtype=d_type_dict,
            )
        elif how == 'insert' or how == 'i':
            bd_ed = mysql_query(
                query=f"SELECT MIN(tradingdate) AS bd, MAX(tradingdate) as ed"
                      f" FROM intern.barra_pure_factor_return",
                engine=conn_mysql(eng)
            )
            bd = bd_ed['bd'][0]
            ed = bd_ed['ed'][0]
            if bd is not None:
                bd = bd.strftime('%Y-%m-%d')
                df.loc[df.index < bd].reset_index().to_sql(
                    name=table_name,
                    con=conn_mysql(eng),
                    if_exists='append',
                    index=False,
                    dtype=d_type_dict,
                )
            if ed is not None:
                ed = ed.strftime('%Y-%m-%d')
                df.loc[df.index > ed].reset_index().to_sql(
                    name=table_name,
                    con=conn_mysql(eng),
                    if_exists='append',
                    index=False,
                    dtype=d_type_dict,
                )
            del bd_ed, bd, ed
        else:
            raise Exception(f'Invalid how={how}, [replace(r), append(a), insert(i)]')
        del df, table_name

        # Upload exposure panel
        df = self.expo_panel[self.ls_style].dropna(how='all')
        if td_1 is not None:
            df = df[df.index.get_level_values(0) > td_1]
        table_name = 'barra_exposure_orthogonal'
        d_type_dict = {'tradingdate': DATE(), 'stockcode': VARCHAR(20)} | {_: DOUBLE() for _ in df.columns}
        print('\nUpload orthogonal exposure of shape', df.shape, '...')
        if how == 'replace' or how == 'r':
            df.reset_index().to_sql(
                name=table_name,
                con=conn_mysql(eng),
                if_exists='replace',
                index=False,
                dtype=d_type_dict,
            )
        elif how == 'append' or how == 'a':
            df.reset_index().to_sql(
                name=table_name,
                con=conn_mysql(eng),
                if_exists='append',
                index=False,
                dtype=d_type_dict,
            )
        elif how == 'insert':
            bd_ed = mysql_query(
                query=f"SELECT MIN(tradingdate) AS bd, MAX(tradingdate) AS ed"
                      f" FROM intern.barra_exposure_orthogonal",
                engine=conn_mysql(eng)
            )
            bd = bd_ed['bd'][0]
            ed = bd_ed['ed'][0]
            if bd is not None:
                bd = bd.strftime('%Y-%m-%d')
                df.loc[df.index.get_level_values(0) < bd].reset_index().to_sql(
                    name=table_name,
                    con=conn_mysql(eng),
                    if_exists='append',
                    index=False,
                    dtype=d_type_dict,
                )
            if ed is not None:
                ed = ed.strftime('%Y-%m-%d')
                df.loc[df.index.get_level_values(0) > ed].reset_index().to_sql(
                    name=table_name,
                    con=conn_mysql(eng),
                    if_exists='append',
                    index=False,
                    dtype=d_type_dict,
                )
            del bd_ed, bd, ed
        else:
            raise Exception(f'Invalid how={how}, [replace(r), append(a), insert(i)]')
        del df

    def cal_pure_return_by_time(self, cache=True):
        """
        For every date in views[1:], calculate pure factor return.
        :param cache: bool, True则将单日的纯因子收益/正交面板/因子成分权重 存到本地
        :return:
        """
        tmp_panel = []
        tmp_fval = []
        print('\nCalculate pure factor returns ...')

        # 每日，准备因子暴露面板（正交），计算纯因子收益
        td_1 = None  # 当前日期 前一个交易日 用于查询因子暴露
        for td in tqdm(self.views):
            if td_1 is None:
                td_1 = td
                continue

            # 日期已存在（从服务器获得），跳过
            if len(self.factor_ret) > 0 and pd.to_datetime(td) <= self.factor_ret.index[-1]:
                continue

            # 单日纯因子收益、因子暴露、因子构成的 缓存地址
            barra_fval_path = self.data_path + 'barra_fval/' + td + '.csv'
            barra_expo_path = self.data_path + 'barra_panel/' + td_1 + '.csv'
            barra_omega_path = self.data_path + 'barra_omega/' + td + '.csv'

            # Get pure factor return & orthogonal factor exposure
            if os.path.exists(barra_fval_path) and os.path.exists(barra_expo_path):  # from local cache
                fv_1d = pd.read_csv(barra_fval_path, index_col=0, parse_dates=True)
                expo1do = pd.read_csv(barra_expo_path, index_col=0)
                fv_1d.index.name = 'tradingdate'

            else:  # from JQ original style exposure

                # Load exposure (z-score, orthogonal)
                if os.path.exists(barra_expo_path):  # orthogonal exposure from local cache
                    expo1do = pd.read_csv(barra_expo_path, index_col=0)
                    self.ls_indus = [_ for _ in expo1do.columns if _[:4] == 'ind_']
                else:  # compose exposure panel from several remote tables
                    # Load 1-period exposure
                    expo1d = self.get_expo1d(td=td_1, cache=True)
                    # Orthogonalize
                    expo1do = exposure_orthogonal(beta_exposure_csi=expo1d,
                                                  ls_style=self.ls_style,
                                                  ls_indus=self.ls_indus,
                                                  s_mv_raw='size')
                    # Save 1d orthogonal exposure in local path
                    if cache:
                        table_save_safe(df=expo1do, tgt=barra_expo_path, kind='csv')

                # orthogonal_exposure + asset_return --(WLS)--> factor_return
                pf_w, fv_1d = wls_1d(
                    beta_exposure_csi=expo1do,
                    ls_style=self.ls_style,
                    ls_indus=self.ls_indus,
                    rtn_next_period=self.asset_ret.loc[td],
                    s_mv_raw='size',
                )
                fv_1d = pd.DataFrame(fv_1d.rename(pd.to_datetime(td))).T

                # Save in disk
                if cache:  # 本地缓存单日纯因子收益/个股在因子的权重（回归结果）
                    table_save_safe(df=pf_w, tgt=barra_omega_path, kind='csv')
                    table_save_safe(df=fv_1d, tgt=barra_fval_path, kind='csv')

            # Save in memory
            expo1do.index = pd.MultiIndex.from_arrays(
                arrays=[[pd.to_datetime(td)] * len(expo1do),
                        expo1do.index.to_series()],
                names=['tradingdate', 'stockcode']
            )
            tmp_panel.append(expo1do)
            fv_1d.index.name = 'tradingdate'
            tmp_fval.append(fv_1d)

            td_1 = td

        # Concat List[daily exposure & factor return]
        if len(tmp_panel) > 0:
            self.expo_panel = pd.concat(
                [self.expo_panel,
                 pd.concat(tmp_panel)]
            )
        if len(tmp_fval) > 0:
            self.factor_ret = pd.concat(
                [self.factor_ret,
                 pd.concat(tmp_fval)]
            )

    def cache_all_style_exposure(self, views: list = None, continuous_date=True):
        """Save all views' raw exposure in disk cache path"""
        if views is None:
            views = self.views
        ls_style = self.ls_style
        for info in ls_style:
            if continuous_date:
                self.CR.read_csv(info=info,
                                 bd=views[0],
                                 ed=views[-1],
                                 local_path=None,
                                 d_type=float,
                                 cache=True)
            else:
                self.CR.read_csv(
                    info=info,
                    views=views,
                    local_path=None,
                    d_type=float,
                    cache=True
                )

    def get_expo1d(self, td: str, cache: bool = True) -> pd.DataFrame:
        """
        Access 1 day exposure
        :param td: str
        :param cache: bool, cache in local disk
        :return: DataFrame of shape (n_assets, n_features)
        """
        if isinstance(td, pd.Timestamp):
            td = td.strftime('%Y-%m-%d')

        # Result panel
        panel = pd.DataFrame()

        # Style exposure
        ls_style = self.ls_style
        # info = ls_style[0]
        for info in ls_style:
            df = self.CR.read_csv(info=info, bd=td, ed=td, local_path=None, d_type=float, cache=cache)
            if len(df) == 0:
                raise Exception(f"{info} {td} value missing")
            df = df.T
            df.columns = [info]
            panel = pd.concat([panel, df], axis=1)
            del df
        del info

        # Industry exposure
        info = self.CR.acc_tgt.loc['ind_citic']
        eng = self.CR.get_engine(info['SERVER'])

        engine = conn_mysql(eng)
        query = f"SELECT {info['IND']},{info['COL']},{info['VAL']}" \
                f" FROM {info['TABLE']}" \
                f" WHERE {info['IND']}=(SELECT MAX({info['IND']}) FROM {info['TABLE']} WHERE {info['IND']} <= '{td}')"
        df = mysql_query(query, engine, telling=False)
        df = df.pivot(index=info['IND'], columns=info['COL'], values=info['VAL'])
        if len(df) != 1:
            raise Exception(df)
        df = pd.get_dummies(df.iloc[0, :])
        df.columns = pd.Index([f"ind_{_.split('.')[0]}" for _ in df.columns])  # rename ind columns: ind_000001
        panel = pd.concat([panel, df], axis=1)
        self.ls_indus = df.columns.to_list()
        del df

        # Country exposure
        panel['country'] = 1

        return panel

    def get_factor_return(self) -> pd.DataFrame:
        return self.factor_ret.copy()

    def get_expo_panel(self) -> pd.DataFrame:
        return self.expo_panel.copy()

    def get_asset_return(self) -> pd.DataFrame:
        return self.asset_ret.copy()

    def get_views(self) -> List[str]:
        return self.views.copy()


class MFM(object):
    """
    Adjust covariance of factor pure return.
    Input: DataFrame like
                     country      size  ...  ind_CI005028.WI  ind_CI005029.WI
        2022-03-25 -0.000808 -0.000332  ...         0.014670        -0.001120
        2022-03-28 -0.006781  0.001878  ...        -0.003114        -0.001607
        2022-03-29  0.013088  0.005239  ...        -0.002646         0.002034
        2022-03-30  0.001433 -0.007067  ...        -0.000453         0.006430

    需要提前给505天（h=252），即给出505+X天的纯因子收益，能够计算最后X天的因子协方差
    """

    def __init__(self, fr: pd.DataFrame = None):
        """
        Adjust factor
        :param fr: factor pure return
        """
        self.factor_ret = fr
        if fr is None:
            self.views = pd.Series(dtype=object)
            self.T = 0
        else:
            self.views = pd.to_datetime(fr.index.to_series())
            self.T = len(fr)

        # self.Newey_West_adj_cov: Dict[str, pd.DataFrame] = dict()
        self.Newey_West_adj_cov: pd.DataFrame = pd.DataFrame()  # T - h, h=252
        self.eigen_risk_adj_cov: pd.DataFrame = pd.DataFrame()  # T - h, h=252
        self.vol_regime_adj_cov: pd.DataFrame = pd.DataFrame()  # T - 2 * h, h=252

    def newey_west_adj_by_time(self, h=252, tau=90, q=2) -> pd.DataFrame:
        """
        纯因子收益率全历史计算协方差进行 Newey West 调整
        :param h: 计算协方差回看的长度 T-h, T-1
        :param tau: 协方差半衰期
        :param q: 假设因子收益为q阶MA过程
        :return: dict, key=日期, val=协方差F_NW
        """
        if self.factor_ret is None:
            raise Exception('No factor return value')

        # Newey_West_cov = {}
        tmp_cov = []
        print('\nNewey West Adjust...')
        for t in range(h, self.T):
            td = self.views[t].strftime('%Y-%m-%d')
            try:
                ret = self.factor_ret[t - h:t]
                # ret.count()  TODO: check factor return missing
                cov = cov_newey_west_adj(ret=ret, tau=tau, q=q)
                # self.Newey_West_adj_cov[td] = cov

                cov.index = pd.MultiIndex.from_arrays(
                    arrays=[[pd.to_datetime(td)] * len(cov),
                            cov.index.to_series()],
                    names=['tradingdate', 'fname']
                )
                tmp_cov.append(cov)

            except:
                tmp_cov.append(
                    pd.DataFrame(index=pd.MultiIndex.from_arrays([[pd.to_datetime(td)], ['NA']],
                                                                 names=['tradingdate', 'fname']))
                )
                # self.Newey_West_adj_cov[td] = (pd.DataFrame())

            progressbar(cur=t - h + 1, total=self.T - h, msg=f'\tdate: {self.views[t - 1].strftime("%Y-%m-%d")}')
        print()
        self.Newey_West_adj_cov = pd.concat(tmp_cov)

        return self.Newey_West_adj_cov

    def eigen_risk_adj_by_time(self, T=1000, M=100, scal=1.4) -> pd.DataFrame:
        """
        逐个F_NW进行 Eigenfactor Rist Adjustment
        :param T: 模拟序列长度
        :param M: 模拟次数
        :param scal: scale coefficient for bias
        :return: dict, key=日期, val=协方差 F_Eigen
        """
        if len(self.Newey_West_adj_cov) == 0:
            raise Exception('run newey_west_adj_by_time first for F_NW')

        print('\nEigen-value Risk Adjust...')
        cnt = 0
        tmp_cov = []
        # td = '2019-03-19'

        # for td in self.Newey_West_adj_cov.keys():
        tradedates = self.Newey_West_adj_cov.index.get_level_values(0).unique().to_list()
        for td0 in tradedates:
            td = td0.strftime('%Y-%m-%d')
            try:
                # cov = self.Newey_West_adj_cov[td]
                # self.eigen_risk_adj_cov[td] = cov_eigen_risk_adj(cov=cov, T=T, M=M, scal=scal)
                cov = cov_eigen_risk_adj(cov=self.Newey_West_adj_cov.loc[td], T=T, M=M, scal=scal)
                cov.index = pd.MultiIndex.from_arrays(
                    arrays=[[pd.to_datetime(td)] * len(cov),
                            cov.index.to_series()],
                    names=['tradingdate', 'fname']
                )
                tmp_cov.append(cov)
            except:
                # self.eigen_risk_adj_cov[td] = pd.DataFrame()
                tmp_cov.append(
                    pd.DataFrame(index=pd.MultiIndex.from_arrays([[pd.to_datetime(td)], ['NA']],
                                                                 names=['tradingdate', 'fname']))
                )

            cnt += 1
            progressbar(cnt, len(tradedates), f'\tdate: {td}')
        print()

        self.eigen_risk_adj_cov = pd.concat(tmp_cov)
        return self.eigen_risk_adj_cov

    def vol_regime_adj_by_time(self, h=252, tau=42) -> pd.DataFrame:
        """
        Volatility Regime Adjustment
        :param h: 波动率乘数的回看时长
        :param tau: 波动率乘数半衰期
        :return: 波动率调整后的相关矩阵，历史缩短 h 日
        """
        if len(self.eigen_risk_adj_cov) == 0:
            raise Exception('run eigen_risk_adj_by_time first for F_Eigen')

        T, K = self.factor_ret.shape

        factor_var = list()
        # tradedates = list(self.eigen_risk_adj_cov.keys())
        tradedates = self.Newey_West_adj_cov.index.get_level_values(0).unique().to_list()
        for td in tradedates:
            f_var_i = np.diag(self.eigen_risk_adj_cov.loc[td])
            if len(f_var_i) == 0:
                f_var_i = np.array(K * [np.nan])
            factor_var.append(f_var_i)
        print()

        factor_var = np.array(factor_var)
        B2 = (self.factor_ret.loc[tradedates] ** 2 / factor_var).mean(axis=1)

        weights = .5 ** (np.arange(h - 1, -1, -1) / tau)
        weights / weights.sum()
        # lamb2 = {}
        print('\nVolatility Regime Adjustment...')
        cnt = 0
        tmp_cov = []
        len(tradedates)
        for td0, td1 in zip(tradedates[:-h + 1], tradedates[h - 1:]):
            lamb2 = B2.loc[td0: td1] @ weights
            # self.vol_regime_adj_cov[td1] = self.eigen_risk_adj_cov[td1] * lamb2
            cov = self.eigen_risk_adj_cov.loc[td1] * lamb2
            cov.index = pd.MultiIndex.from_arrays(
                arrays=[[pd.to_datetime(td1)] * len(cov),
                        cov.index.to_series()],
                names=['tradingdate', 'fname']
            )
            tmp_cov.append(cov)
            cnt += 1
            progressbar(cnt, len(tradedates) - h, f"\tdate: {td1.strftime('%Y-%m-%d')}")
        print()
        self.vol_regime_adj_cov = pd.concat(tmp_cov)
        return self.vol_regime_adj_cov

    def upload_adjusted_covariance(self, eng: dict, level='VRA', how='append', td_1: str = None):
        """"""
        if level == 'NW':
            cov_d = self.Newey_West_adj_cov
            table_name = 'barra_factor_cov_nw'
        elif level == 'Eigen':
            cov_d = self.eigen_risk_adj_cov
            table_name = 'barra_factor_cov_nw_eigen'
        elif level == 'VRA':
            cov_d = self.vol_regime_adj_cov
            table_name = "barra_factor_cov_nw_eigen_vra"
        else:
            raise Exception('save_factor_covariance arg: level not in {`NW`, `Eigen`, `VRA`}')

        if len(cov_d) == 0:
            raise Exception('run *_adj_by_time first for *_adj_cov')

        if td_1 is not None:
            cov_d = cov_d[cov_d.index.get_level_values(0) > td_1]

        d_type_dict = {'tradingdate': DATE(),
                       'fname': VARCHAR(30)} | {_: DOUBLE() for _ in cov_d.columns}
        cov_d.reset_index().to_sql(
            name=table_name,
            con=conn_mysql(eng),
            if_exists=how,
            index=False,
            dtype=d_type_dict,
        )

    def save_factor_covariance(self, path, level='VRA'):
        """Save as ${path}/F_NW_Eigen_VRA[yyyy-mm-dd,yyyy-mm-dd].csv"""

        def frame1d_2d(df: pd.DataFrame, td: str) -> pd.DataFrame:
            df['names'] = df.index
            df['tradingdate'] = td
            df = df.set_index(['tradingdate', 'names'])
            return df

        def dict2frame(_cov: Dict[str, pd.DataFrame]) -> pd.DataFrame:
            """字典形式存储的换到Frame"""
            res = pd.DataFrame()
            for td in _cov.keys():
                res = res.append(frame1d_2d(_cov[td], td))
            return res

        if level == 'NW':
            cov_d = self.Newey_West_adj_cov
            file_name = "F_NW[{}].csv"
        elif level == 'Eigen':
            cov_d = self.eigen_risk_adj_cov
            file_name = "F_NW_Eigen[{}].csv"
        elif level == 'VRA':
            cov_d = self.vol_regime_adj_cov
            file_name = "F_NW_Eigen_VRA[{}].csv"
        else:
            raise Exception('save_factor_covariance arg: level not in {`NW`, `Eigen`, `VRA`}')

        if len(cov_d) == 0:
            raise Exception('run *_adj_by_time first for factor_covariance_*')
        # cov = dict2frame(cov_d)
        cov = cov_d
        file_name = file_name.format(
            ','.join(
                [_.strftime('%Y-%m-%d') for _ in cov.index.get_level_values(0)[[0, -1]]]
            )
        )
        print(f'\nSave as `{file_name}`')
        cov.to_csv(path + '/' + file_name)


class SRR(object):
    """Specific Return Risk"""

    def __init__(self, sr, fr, expo, mv):
        """
        Asset specific return adjustment, input length (T) + 2*h, h=252
        :param sr: asset returns
        :param fr: factor returns
        :param expo: factor exposure
        :param mv: market value
        """
        self.asset_ret: pd.DataFrame = sr
        self.factor_ret: pd.DataFrame = fr
        self.expo_panel: pd.DataFrame = expo
        self.mkt_val: pd.DataFrame = mv

        self.views: pd.Series = pd.to_datetime(fr.index.to_series())
        self.T: int = len(fr)

        self.u: pd.DataFrame = pd.DataFrame()  # T
        self.SigmaRaw: pd.DataFrame = pd.DataFrame()  # T - h=252
        self.SigmaNW: pd.DataFrame = pd.DataFrame()  # T - h=252
        self.GammaSM: pd.DataFrame = pd.DataFrame()
        self.SigmaSM: pd.DataFrame = pd.DataFrame()  # T - h=252
        self.SigmaSH: pd.DataFrame = pd.DataFrame()  # T - h=252
        self.LambdaVRA: pd.DataFrame = pd.DataFrame()
        self.SigmaVRA: pd.DataFrame = pd.DataFrame()  # T - 2*h=504

    def specific_return_by_time(self):
        print('\nSpecific Return...')
        self.u = specific_return_yxf(Y=self.asset_ret, X=self.expo_panel, F=self.factor_ret)
        return self.u

    def newey_west_adj_by_time(self, h=252, NA_bar=.75, tau=90, q=5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        特异收益率全历史计算协方差进行 Newey West 调整
        :param NA_bar:
        :param h: 计算协方差回看的长度 T-h, T-1
        :param tau: 协方差半衰期
        :param q: 假设因子收益为q阶MA过程
        :return: dict, key=日期, val=协方差Sigma_NW
        """
        if self.u is None:
            raise Exception('No specific return value')

        # Newey_West_cov = {}
        print('\nNewey West Adjust...')
        SigmaRaw = []
        SigmaNW = []
        # t = h
        for t in range(h, self.T):
            td = self.views[t].strftime('%Y-%m-%d')
            try:
                u0 = self.u.iloc[t - h:t]
                u1 = u0[u0.columns[u0.count() > h * NA_bar]]
                sigma_raw, sigma_nw = var_newey_west_adj(ret=u1, tau=tau, q=q)
                SigmaRaw.append(sigma_raw.rename(td))
                SigmaNW.append(sigma_nw.rename(td))
            except:
                SigmaRaw.append(pd.Series([]).rename(td))
                SigmaNW.append(pd.Series([]).rename(td))

            progressbar(cur=t - h + 1, total=self.T - h, msg=f'\tdate: {td}')
        print()

        self.SigmaRaw = pd.DataFrame(SigmaRaw)
        self.SigmaNW = pd.DataFrame(SigmaNW)
        self.SigmaRaw.index = pd.to_datetime(self.SigmaRaw.index.to_series())
        self.SigmaNW.index = pd.to_datetime(self.SigmaNW.index.to_series())
        return self.SigmaRaw, self.SigmaNW

    def struct_mod_adj_by_time(self, h=252, NA_bar=.75, E=1.05):
        if len(self.SigmaNW) == 0:
            raise Exception('No Newey-West Adjusted Sigma SigmaNW')

        print('\nStructural Model Adjust...')
        GammaSM = []
        SigmaSM = []
        cnt = 0
        td = self.views[h]
        for td in self.views[h: self.T]:
            # %
            sigNW = self.SigmaNW.loc[td].dropna()
            U = self.u.loc[:td].iloc[-h:]
            U = U.loc[:, U.count() > h * NA_bar]
            expo = self.expo_panel.loc[td].dropna(axis=1, how='all').dropna(axis=0, how='any')  # 全空的风格暴露&有空的个股
            MV = self.mkt_val.loc[td].dropna()
            # %
            try:
                res = var_struct_mod_adj(U=U, sigNW=sigNW, expo=expo, MV=MV, E=E)
                GammaSM.append(res[0].rename(td))
                SigmaSM.append(res[1].rename(td))
            except:  # TODO: WLS回归可能出现奇异阵
                GammaSM.append(pd.Series([]).rename(td))
                SigmaSM.append(pd.Series([]).rename(td))
            cnt += 1
            progressbar(cur=cnt, total=self.T - h, msg=f'\tdate: {td.strftime("%Y-%m-%d")}')
        print()

        self.SigmaSM = pd.DataFrame(SigmaSM)
        self.GammaSM = pd.DataFrame(GammaSM)

        return self.GammaSM, self.SigmaSM

    def bayesian_shrink_by_time(self, q=1, gn=10):
        T = len(self.SigmaSM)
        if T == 0:
            raise Exception('No Newey-West Adjusted Sigma SigmaNW')

        print('\nBayesian Shrink Adjust...')
        cnt = 0
        SigmaSH = []
        # td = self.SigmaSM.index[-1]
        for td in self.SigmaSM.index:
            MV = self.mkt_val.loc[td]
            sigSM = self.SigmaSM.loc[td].rename('sig_hat')
            mv: pd.Series = MV[sigSM.index]  # TODO: why MV=NA?
            # print(mv.isna().sum())
            try:
                SigmaSH.append(var_bayesian_shrink(sigSM=sigSM, mv=mv, q=q, gn=gn).rename(td))
            except:
                SigmaSH.append(pd.Series([]).rename(td))
            cnt += 1
            progressbar(cur=cnt, total=T, msg=f'\tdate: {td.strftime("%Y-%m-%d")}')
        print()

        self.SigmaSH = pd.DataFrame(SigmaSH)
        return self.SigmaSH

    def vol_regime_adj_by_time(self, h=252, tau=42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        B2 = (self.u.reindex_like(self.SigmaSH) / self.SigmaSH) ** 2
        w = self.mkt_val.reindex_like(B2) * (1 - B2.isna())
        w = w.apply(lambda s: s / s.sum(), axis=1)
        B2 = (B2 * w).sum(axis=1)

        weights = .5 ** (np.arange(h - 1, -1, -1) / tau)
        weights /= weights.sum()
        tradedates = B2.index
        Lambda = pd.DataFrame()
        SigmaVRA = pd.DataFrame()
        cnt = 0
        print('\nVolatility Regime Adjustment...')
        for td0, td1 in zip(tradedates[:-h + 1], tradedates[h - 1:]):
            # lamb2[td1] = B2.loc[td0: td1] @ weights
            lamb = np.sqrt(B2.loc[td0: td1] @ weights)
            Lambda[td1] = lamb
            SigmaVRA[td1] = self.SigmaSH.loc[td1] * lamb
            cnt += 1
            progressbar(cnt, len(tradedates) - h, f'\tdate: {td1.strftime("%Y-%m-%d")}')
        print()

        self.LambdaVRA = Lambda.T
        self.SigmaVRA = SigmaVRA.T
        return self.LambdaVRA, self.SigmaVRA

    def upload_asset_specific_risk(self, eng: dict, level='VRA', how='append', td_1: str = None):
        """"""
        if level == 'Raw':
            sig = self.SigmaRaw
            table_name = "barra_specific_risk_raw"
        elif level == 'NW':
            sig = self.SigmaNW
            table_name = "barra_specific_risk_nw"
        elif level == 'SM':
            sig = self.SigmaSM
            table_name = "barra_specific_risk_nw_sm"
        elif level == 'SH':
            sig = self.SigmaSH
            table_name = "barra_specific_risk_nw_sm_sh"
        elif level == 'VRA':
            sig = self.SigmaVRA
            table_name = "barra_specific_risk_nw_sm_sh_vra"
        else:
            raise Exception('level not in {`Raw`, `NW`, `SM`, `SH`, `VRA`}')

        if len(sig) == 0:
            raise Exception('run *_adj_by_time first for specific_risk_*')

        if td_1 is not None:
            sig = sig[sig.index.get_level_values(0) > td_1]

        sig = sig.stack().reset_index()
        sig.columns = ['tradingdate', 'stockcode', 'fv']
        d_type_dict = {'tradingdate': DATE(),
                       'stockcode': VARCHAR(20),
                       'fv': DOUBLE()}
        sig.to_sql(
            name=table_name,
            con=conn_mysql(eng),
            if_exists=how,
            index=False,
            dtype=d_type_dict,
        )

    def save_vol_regime_adj_risk(self, path, level='VRA'):
        """Save as ${path}/D_NW_SM_SH_VRA[yyyy-mm-dd,yyyy-mm-dd].csv"""
        if level == 'Raw':
            sig = self.SigmaRaw
            file_name = "D_Raw[{}].csv"
        elif level == 'NW':
            sig = self.SigmaNW
            file_name = "D_NW[{}].csv"
        elif level == 'SM':
            sig = self.SigmaSM
            file_name = "D_NW_SM[{}].csv"
        elif level == 'SH':
            sig = self.SigmaSH
            file_name = "D_NW_SM_SH[{}].csv"
        elif level == 'VRA':
            sig = self.SigmaVRA
            file_name = "D_NW_SM_SH_VRA[{}].csv"
        else:
            raise Exception('level not in {`Raw`, `NW`, `SM`, `SH`, `VRA`}')

        if len(sig) == 0:
            raise Exception('run *_adj_by_time first for specific_risk_*')
        file_name = file_name.format(f"{sig.index[0].strftime('%Y-%m-%d')},{sig.index[-1].strftime('%Y-%m-%d')}")
        print(f'Save as {file_name}')
        sig.to_csv(path + '/' + file_name)

    def cal_volatility_cross_section(self):
        """TODO"""
        sr_mv = (self.mkt_val.reindex_like(self.u) * (1 - self.u.isna())).apply(lambda s: s / s.sum(), axis=1)
        sr_mv @ self.u ** 2
        pass

    def plot_structural_model_gamma(self):
        g = self.GammaSM
        if len(g) == 0:
            raise Exception('run *_adj_by_time first for specific_risk_*')
        ratio = (g == 1).sum(axis=1) / g.count(axis=1)
        ratio.plot(title='ratio of good-quality specific return (with $\gamma=1$)')
        plt.tight_layout()
        plt.show()


def load_asset_return(
        CR: CsvReader,
        views: List[str] = None,
        bd: str = None,
        ed: str = None,
        cache: bool = True,
        shifting: int = 0
) -> pd.DataFrame:
    """

    :param CR: CsvReader, cache parser
    :param views: None or list of views
    :param bd: None or begin date
    :param ed: None or end date
    :param cache: update local cache if True
    :param shifting: shift backward
    :return:
    """
    df = CR.read_csv(info='close_adj',
                     views=views,
                     bd=bd,
                     ed=ed,
                     local_path=None,
                     d_type=float,
                     cache=cache)  # adjusted close price
    # TODO: exclude new IPO
    df = df.pct_change()
    if shifting:
        df = df.shift(-shifting).iloc[:-shifting]
    # TODO: other winsorize methods
    df[df > 0.11] = 0.11
    df[df < -0.11] = -0.11
    return df


def load_asset_marketvalue(
        CR: CsvReader,
        views: List[str] = None,
        bd: str = None,
        ed: str = None,
        cache: bool = True,
) -> pd.DataFrame:
    df = CR.read_csv(info='marketvalue',
                     views=views,
                     bd=bd,
                     ed=ed,
                     local_path=None,
                     d_type=float,
                     cache=cache)  # adjusted close price
    return df


def exposure_orthogonal(beta_exposure_csi: pd.DataFrame,
                        ls_style: list,
                        ls_indus: list,
                        s_mv_raw: str = 'size'
                        ) -> pd.DataFrame:
    """
    风格因子对行业、市值正交. 注意此后的size已经调整！
    :param beta_exposure_csi:
        DataFrame of shape (n_assets, n_features),
        Barra factor exposure - Country, Style, Industry
    :param ls_style:
        list,
        names of style factors
    :param ls_indus:
        list,
        names of industry
    :param s_mv_raw:
        str,
        name of market value in columns
    :return:
        DataFrame of shape (n_assets, n_features),
        Orthogonal exposure
    """
    res = beta_exposure_csi.copy()
    # col = ls_style[1]
    for col in ls_style:
        # print(col)
        if col == s_mv_raw:
            x = beta_exposure_csi[ls_indus]
        else:
            x = beta_exposure_csi[[s_mv_raw] + ls_indus]
        y = beta_exposure_csi[col]
        est = sm.OLS(y, x, missing='drop').fit()
        sr = est.resid
        res.loc[:, col] = (sr - np.nanmean(sr)) / np.nanstd(sr)
        del sr
    return res


def wls_1d(beta_exposure_csi: pd.DataFrame,
           ls_style: list,
           ls_indus: list,
           rtn_next_period: pd.Series,
           s_mv_raw: str = 'size',
           cross_check: bool = False,
           ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    WLS, calculate pure factor return (1 period),
    Return: pure factor components & returns (1 period)
    :param beta_exposure_csi:
        DataFrame of shape (n_assets, n_features),
        Barra exposure - Country, Style, Industry
    :param ls_style:
        list,
        name of industry in expo columns
    :param ls_indus:
        list,
        name of style in expo columns
    :param rtn_next_period:
        Series of shape (n_assets,),
        asset returns next period
    :param s_mv_raw:
        str,
        column name 'size' asset market value (non-neutralized) this period
    :param cross_check:
        bool,
        check WLS result with statsmodels.api.WLS
    :return Tuple[pf_w, fv_1d]:
        pf_w:
            DataFrame of shape (n_assets, n_features),
            pure factor institution, weight sum = 1
        fv_1d:
            Series of shape (n_features,),
            pure factor returns next period

    """

    # Check missing value
    mask = (~beta_exposure_csi.isna().any(axis=1)) & (~rtn_next_period.isna())

    # WLS 权重：市值对数
    market_value_raw = beta_exposure_csi.loc[mask, s_mv_raw]
    mv = market_value_raw.loc[mask].apply(lambda _: np.exp(_))
    w_mv = mv.apply(lambda _: np.sqrt(_))
    w_mv = w_mv / w_mv.sum()
    mat_v = np.diag(w_mv)

    # 最终进入回归的因子, 19年前行业分类只有30个
    if beta_exposure_csi.loc[mask, ls_indus].isna().any().sum() > 1:
        f_cols = ['country'] + ls_style + ls_indus[:-1]
    else:
        f_cols = ['country'] + ls_style + ls_indus
    mat_x = beta_exposure_csi.loc[mask, f_cols].values

    # 行业因子约束条件
    mv_indus = mv.values.T @ beta_exposure_csi.loc[mask, ls_indus].values
    assert mv_indus.prod() != 0
    k = len(f_cols)
    mat_r = np.diag([1.] * k)[:, :-1]
    mat_r[-1:, -len(ls_indus) + 1:] = - mv_indus[:-1] / mv_indus[-1]

    # WLS求解（Menchero & Lee, 2015)
    mat_omega = mat_r @ np.linalg.inv(mat_r.T @ mat_x.T @ mat_v @ mat_x @ mat_r) @ mat_r.T @ mat_x.T @ mat_v
    pf_w = pd.DataFrame(mat_omega.T, index=beta_exposure_csi.loc[mask].index, columns=f_cols)

    mat_y = rtn_next_period.loc[mask].values
    fv_1d = pd.Series(mat_omega @ mat_y, index=f_cols)

    # 等效计算，条件处理后的WLS
    if cross_check:
        mod = sm.WLS(mat_y, mat_x @ mat_r, weights=w_mv)
        res = mod.fit()
        fv = pd.Series(mat_r @ res.params, index=f_cols)
        assert (fv - fv_1d).abs().sum() < 1e-12

    return pf_w, fv_1d


def load_tradedate_view(eng: dict, bd: str, ed: str, freq='d') -> List[str]:
    """
    Load views - tradingdate list from remote
    :param eng: dict, engine0
    :param bd: str, begin date
    :param ed: str, end date
    :param freq: str, frequency - 'd' 'w' or 'm'
    :return: Series[Timestamp], tradingdate in the range
    """
    engine = conn_mysql(eng=eng)
    col = 'tradingdate'
    tb = f'tdays_{freq}'
    query = f"SELECT {col} FROM {tb}" \
            f" WHERE {col} >= '{bd}'" \
            f" AND {col} <= '{ed}'"
    df = mysql_query(query, engine)
    if len(df) < 1:
        raise Exception(f"No {col} from {bd}"
                        f" to {ed}, freq='{freq}'")
    views = [_.strftime('%Y-%m-%d') for _ in df[col]]
    return views


def load_remote_table(info, eng, bd=None, ed=None, notify=True, views=None) -> pd.DataFrame:
    """
    ..
    :param notify:
    :param info: series of shape (11,)
    :param views: List[str]
    :param eng: dict, sql engine conf
    :param bd: str, begin date
    :param ed: str, end date
    :return:
    """
    engine = conn_mysql(eng)

    if info['1D'] == 'F':  # table of shape (n_views, n_features)
        if views is None:
            query = f"SELECT {info['IND']},{info['COL']},{info['VAL']}" \
                    f" FROM {info['TABLE']}" \
                    f" WHERE {info['IND']}>='{bd}' AND {info['IND']}<='{ed}'" \
                    f"{' AND ' + info['WHERE'] if isinstance(info['WHERE'], str) else ''}" \
                    f" ORDER BY {info['IND']};"
        elif len(views) > 0:
            if isinstance(views[0], pd.Timestamp):
                views = [_.strftime('%Y-%m-%d') for _ in views]
            s_views = f"""('{"','".join(views)}')"""
            query = f"SELECT {info['IND']},{info['COL']},{info['VAL']}" \
                    f" FROM {info['TABLE']}" \
                    f" WHERE {info['IND']} in {s_views}" \
                    f"{' AND ' + info['WHERE'] if isinstance(info['WHERE'], str) else ''}" \
                    f" ORDER BY {info['IND']};"
        else:
            raise Exception(f'`views` empty, {views}')

    else:  # table of shape (n_views,)
        if views is None:
            query = f"SELECT {info['VAL']}" \
                    f" FROM {info['TABLE']}" \
                    f" WHERE {info['IND']}>='{bd}' AND {info['IND']}<='{ed}'" \
                    f"{' AND ' + info['WHERE'] if isinstance(info['WHERE'], str) else ''};"
        else:
            if isinstance(views[0], pd.Timestamp):
                views = [_.strftime('%Y-%m-%d') for _ in views]
            s_views = f"""('{"','".join(views)}')"""
            query = f"SELECT {info['VAL']}" \
                    f" FROM {info['TABLE']}" \
                    f" WHERE {info['IND']} in {s_views}" \
                    f"{' AND ' + info['WHERE'] if isinstance(info['WHERE'], str) else ''};"

    df = mysql_query(query, engine, telling=notify)

    if info['1D'] == 'F':
        val_col = info['VAL'].split('AS')[1].strip() if 'AS' in info['VAL'] else info['VAL']
        panel = df.pivot(index=info['IND'], columns=info['COL'], values=val_col)
        panel.index = pd.to_datetime(panel.index.to_series())
    else:
        panel = df
        panel.index = pd.to_datetime(panel[info['VAL']]).rename(None)

    return panel


def conn_mysql(eng: dict):
    """根据dict中的服务器信息，连接mysql"""
    user = eng['user']
    password = eng['password']
    host = eng['host']
    port = eng['port']
    dbname = eng['dbname']
    engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}?charset=UTF8MB4')
    return engine


def mysql_query(query, engine, telling=True):
    """mysql接口，返回DataFrame"""
    if telling:
        print(query)
    return pd.read_sql_query(query, engine)


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


def get_barra_factor_return_daily(conf, y=None) -> pd.DataFrame:
    if y is None:
        return pd.read_csv(conf['barra_fval'], index_col=0, parse_dates=True)
    else:
        return pd.DataFrame(pd.read_hdf(conf['barra_factor_value'], key=f'y{y}'))


def get_barra_factor_exposure_daily(conf, use_temp=False, y=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """获取风格因子暴露"""
    if y is not None:
        stk_expo = pd.DataFrame(pd.read_hdf(conf['barra_panel'], key=f'y{y}'))
    elif use_temp:
        stk_expo = pd.DataFrame(pd.read_hdf(conf['barra_panel_1222'], key='y1222'))
    else:
        # import h5py
        # for k in list(h5py.File(conf['barra_panel'], 'r').keys()):
        stk_expo = pd.DataFrame()
        for k in [f'y{_}' for _ in range(2012, 2023)]:
            stk_expo = stk_expo.append(pd.read_hdf(conf['barra_panel'], key=k))
        stk_expo.to_hdf(conf['barra_panel_1222'], key='y1222')
    return stk_expo['rtn_ctc'].unstack(), stk_expo[[c for c in stk_expo.columns if c != 'rtn_ctc']]


def get_tdays_series(conf, freq='w', bd=None, ed=None) -> pd.Series:
    df = pd.read_csv(conf[f'tdays_{freq}'], index_col=0, parse_dates=True)
    if bd is not None:
        df = df.loc[bd:]
    if ed is not None:
        df = df.loc[:ed]
    df['dates'] = df.index
    return df['dates']


def progressbar(cur, total, msg):
    """显示进度条"""
    import math
    percent = '{:.2%}'.format(cur / total)
    print("\r[%-50s] %s" % ('=' * int(math.floor(cur * 50 / total)), percent) + msg, end='')


def cov_newey_west_adj(ret, tau=90, q=2) -> pd.DataFrame:
    """
    Newey-West调整时序上相关性
    :param ret: 列为因子收益，行为时间，一般取T-252,T-1
    :param tau: 协方差计算半衰期
    :param q: 假设因子收益q阶MA过程
    :return: 经调整后的协方差矩阵
    """
    T, K = ret.shape
    if T <= q or T <= K:
        raise Exception("T <= q or T <= K")

    names = ret.columns
    weights = .5 ** (np.arange(T - 1, -1, -1) / tau)
    weights /= weights.sum()

    ret1 = np.matrix((ret - ret.T @ weights).values)
    gamma0 = [weights[t] * ret1[t].T @ ret1[t] for t in range(T)]
    v = np.array(gamma0).sum(0)

    for i in range(1, q + 1):
        gamma1 = [weights[i + t] * ret1[t].T @ ret1[t + i] for t in range(T - i)]
        cd = np.array(gamma1).sum(0)
        v += (1 - i / (1 + q)) * (cd + cd.T)

    return pd.DataFrame(v, columns=names, index=names)


def var_newey_west_adj(ret, tau=90, q=5) -> Tuple[pd.Series, pd.Series]:
    """
    Newey-West调整时序上相关性
    :param ret: column为特异收益，index为时间，一般取T-252,T-1
    :param tau: 协方差计算半衰期
    :param q: 假设因子收益q阶MA过程
    :return: 经调整后的协方差（截面）
    """
    T, K = ret.shape
    if T <= q:
        raise Exception("T <= q")

    names = ret.columns

    weights = .5 ** (np.arange(T - 1, -1, -1) / tau)
    weights = (~ret.isna()) * weights.reshape(-1, 1)  # w on all stocks, w=0 if missing
    weights /= weights.sum()

    # ret1 = np.matrix((ret - (ret * weights).sum()).values)
    ret1 = ret - (ret * weights).sum()
    gamma0 = (ret1 ** 2 * weights).sum()

    v = gamma0.copy()
    for i in range(1, q + 1):
        ret_i = ret1 * ret1.shift(i)
        weights_i = .5 ** (np.arange(T - 1, -1, -1) / tau)
        weights_i = (~ret_i.isna()) * weights_i.reshape(-1, 1)  # w on all stocks, w=0 if missing
        weights_i /= weights_i.sum()
        gamma_i = (ret_i * weights_i).sum()
        v += (1 - i / (1 + q)) * (gamma_i + gamma_i)

    sigma_raw = pd.Series(gamma0.apply(np.sqrt), index=names)
    sigma_nw = pd.Series(v.apply(np.sqrt), index=names)
    return sigma_raw, sigma_nw


def var_struct_mod_adj(U: pd.DataFrame, sigNW: pd.DataFrame, expo: pd.DataFrame, MV: pd.DataFrame, E=1.05) -> Tuple[
    pd.Series, pd.Series]:
    """
    Structural Model: robust estimates for stocks with specific return histories that are not well-behaved
    :param U: specific risk, parse last h=252 days
    :param sigNW: one day sigma after Newey-West adjustment
    :param expo: one day factor exposure
    :param MV: one day market value (raw, from local mv file rather than JoinQuant)
    :param E: sqrt(mv)-weighted average of the ratio between time series and structural specific risk forecasts,
    average over the back-testing periods, 1.026 for EUE3
    :return:
    """
    # %
    h = U.shape[0]
    sigTilde = (U.quantile(.75) - U.quantile(.25)) / 1.35
    sigEq = U[(U >= -10 * sigTilde) & (U <= 10 * sigTilde)].std()
    Z = (sigEq / sigTilde - 1).abs()

    gamma = Z.apply(
        lambda _: np.nan if np.isnan(_) else min(1., max(0., (h - 60) / 120)) * min(1., max(0., np.exp(1 - _))))
    stk_gamma_eq_1 = gamma.index[gamma == 1]
    stk_has_sigNW = sigNW.index.intersection(stk_gamma_eq_1)
    stk_has_expo = expo.index.intersection(stk_has_sigNW)
    stk_has_mv = MV.index.intersection(stk_has_expo)

    Y = sigNW.loc[stk_has_mv].apply(np.log)
    factor_i = [c for c in expo.columns if 'ind_' in c and expo[c].abs().sum() > 0]
    factor_cs = [c for c in expo.columns if 'ind_' not in c]
    X = expo.loc[stk_has_mv, factor_cs + factor_i].astype(float)
    assert 'size' in X.columns
    mv = MV.loc[stk_has_mv]  # raw market value
    w_mv = mv.apply(np.sqrt)  # stk with exposure must have market value, or Error
    w_mv /= w_mv.sum()

    # % WLS
    mat_v = np.diag(w_mv)
    mat_x = X.values
    mv_indus = mv @ X[factor_i]
    k = X.shape[1]
    mat_r = np.diag([1.] * k)[:, :-1]
    mat_r[-1:, -len(factor_i) + 1:] = -mv_indus[:-1] / mv_indus[-1]
    mat_omega = mat_r @ np.linalg.inv(
        mat_r.T @ mat_x.T @ mat_v @ mat_x @ mat_r
    ) @ mat_r.T @ mat_x.T @ mat_v

    mat_y = Y.values
    mat_b = mat_omega @ mat_y.reshape(-1, 1)
    b_hat = pd.DataFrame(mat_b, index=X.columns)
    sigSTR = E * np.exp(expo[factor_cs + factor_i] @ b_hat)
    sigma_hat = (gamma * sigNW + (1 - gamma) * sigSTR.iloc[:, 0]).dropna()

    # %
    return gamma, sigma_hat


def var_bayesian_shrink(sigSM: pd.Series, mv: pd.Series, gn=10, q=1) -> pd.Series:
    """Bayesian Shrinkage"""
    mv_group = mv.rank(pct=True, ascending=False).apply(lambda x: (1 - x) // (1 / gn))  # low-rank: small size
    # print(mv_group.isna().sum())
    tmp = pd.DataFrame(sigSM.rename('sig_hat'))
    tmp['mv'] = mv
    tmp['g'] = mv_group
    tmp = tmp.reset_index()
    tmp = tmp.merge(tmp.groupby('g')['mv'].sum().rename('mv_gsum').reset_index(), on='g', how='left')
    tmp['w'] = tmp['mv'] / tmp['mv_gsum']
    tmp = tmp.merge((tmp['w'] * tmp['sig_hat']).groupby(tmp['g']).sum().rename('sig_bar').reset_index(), on='g',
                    how='left')
    tmp['sig_d'] = tmp['sig_hat'] - tmp['sig_bar']
    tmp = tmp.merge((tmp['sig_d'] ** 2).groupby(tmp['g']).mean().apply(np.sqrt).rename('D').reset_index(), on='g',
                    how='left')
    tmp['v'] = tmp['sig_d'].abs() * q / (tmp['D'] + tmp['sig_d'].abs() * q)
    tmp['sig_sh'] = tmp['v'] * tmp['sig_bar'] + (1 - tmp['v']) * tmp['sig_hat']
    tmp = tmp.set_index(tmp.columns[0])  # 'index')
    sigSH = tmp['sig_sh']
    return sigSH


def cov_eigen_risk_adj(cov, T=1000, M=10000, scal=1.2) -> pd.DataFrame:
    """
    特征值调整 Eigenfactor Risk Adjustment
    :param cov: 待调整的协方差矩阵
    :param T: 模拟的序列长度
    :param M: 模拟次数
    :param scal: 经验系数
    :return: 经调整后的协方差矩阵
    """
    F0 = cov.copy().dropna(how='all').dropna(how='all', axis=1)
    K = F0.shape[0]
    D0, U0 = np.linalg.eig(F0)  # F0 = U0 @ np.diag(D0) @ U0.T

    if not all(D0 >= 0):  # 正定
        raise Exception('Covariance is not symmetric positive-semidefinite')

    v = []
    # print('Eigenfactor Risk Adjustment..')
    # for m in tqdm(range(M)):
    for m in range(M):
        np.random.seed(m + 1)
        bm = np.random.multivariate_normal(mean=[0] * K, cov=np.diag(D0), size=T).T  # 模拟因子特征收益
        rm = U0 @ bm  # 模拟因子收益
        Fm = np.cov(rm)  # 模拟因子收益协方差
        Dm, Um = np.linalg.eig(Fm)  # 协方差特征分解
        Dm_tilde = Um.T @ F0 @ Um  # 模拟特征真实协方差
        v.append(np.diag(Dm_tilde) / Dm)

    gamma = scal * (np.sqrt(np.mean(np.array(v), axis=0)) - 1) + 1  # 实际因子收益“尖峰厚尾”调整
    D0_tilde = np.diag(gamma ** 2 * D0)  # 特征因子协方差“去偏”
    F0_tilde = U0 @ D0_tilde @ U0.T  # 因子协方差“去偏”调整

    return pd.DataFrame(F0_tilde, columns=F0.columns, index=F0.columns).reindex_like(cov)


def specific_return_yxf(Y: pd.DataFrame, X: pd.DataFrame, F: pd.DataFrame) -> pd.DataFrame:
    """
    U = Y - F X^T
    :param Y: T*N 资产收益，T为日度（取h=252），N为资产数量（全A股）
    :param X: (T*N)*K 资产因子暴露，K为因子数
    :param F: T*K 纯因子收益，由Barra部分WLS回归得到
    :return N*N NW调整的特异风险（对角阵）
    """
    # Y0 = pd.DataFrame()
    Y0 = []
    cnt = 0
    for td in F.index:
        # Y0 = pd.concat([Y0, (X.loc[td].fillna(0) @ F.loc[td].fillna(0)).rename(td)], axis=1)
        Y0.append((X.loc[td].fillna(0) @ F.loc[td].fillna(0)).rename(td))
        cnt += 1
        progressbar(cnt, F.shape[0], msg=f'\tdate: {td.strftime("%Y-%m-%d")}')
    print()

    Y1 = pd.DataFrame(Y0)
    U = Y.reindex_like(Y1) - Y1  # T*N specific returns
    # U.isna().sum().plot.hist(bins=100, title='Missing U=Y-XF')
    # plt.show()
    return U


def keep_index_intersection(idx_ls):
    idx_intersect = None
    for x in idx_ls:
        idx_intersect = x if idx_intersect is None else idx_intersect.intersection(x)
    return idx_intersect


if __name__ == '__main__':
    main()
