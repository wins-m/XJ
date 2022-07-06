"""
(created by swmao on July 4th)

Step1: Calculate pure factor return (daily).
------
Cache table in local disk is optional;
Once cached, shall be manually removed;

`Y (T+1 asset return) ~ X (T+1 beta return) @ B (T0 beta exposure)`

Config:
- begin_date
- end_date
- data_path
- ipo_await
- access_target
- mysql_engine

Input:
- beta exposure (country + style + industry)
- close, daily close return

Output:
- barra_panel, `barra_exposure_orthogonal`
- barra_fval, `barra_pure_factor_return`
- barra_omega


"""
import time
import os
import pandas as pd
import numpy as np
from typing import Tuple
import statsmodels.api as sm
from sqlalchemy import create_engine
from typing import List
from tqdm import tqdm
from sqlalchemy.dialects.mysql import DOUBLE, VARCHAR, DATE
from supporter.cov_a import MFM, SRR

_PATH = '/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/'  # PyCharmProject


def main():
    # Determine begin_date & end_date first
    conf = {
        'begin_date': '2012-01-01',
        'end_date': '2022-07-01',
        'freq': 'd',
        'data_path': '/home/swmao/cache0705/',
        'ipo_await': 90,
        'access_target': f"{_PATH}/beta_optimize/access_target.xlsx",
        'ls_style': [
            'size', 'beta', 'momentum',
            'residual_volatility', 'non_linear_size',
            'book_to_price_ratio', 'liquidity',
            'earnings_yield', 'growth', 'leverage',
        ],
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
    os.makedirs(conf['data_path'], exist_ok=True)
    os.makedirs(conf['data_path'] + 'barra_panel/', exist_ok=True)
    os.makedirs(conf['data_path'] + 'barra_omega/', exist_ok=True)
    os.makedirs(conf['data_path'] + 'barra_fval/', exist_ok=True)

    bfm = BarraFM(conf)
    bfm.cache_all_style_exposure()  # 一次性将begin_date到end_date的style暴露读到本地；可以不执行。
    bfm.cal_pure_return_by_time(cache=True)  # cache=False时不保留本地缓存，但会尝试从本地读取缓存。
    # self.upload_results(eng=conf['mysql_engine']['engine4'], how='insert')  # TODO: 无法去重

    return


class BarraFM(object):

    def __init__(self, conf: dict):

        self.begin_date: str = conf['begin_date']
        self.end_date: str = conf['end_date']
        self.freq: str = conf['freq']
        self.data_path: str = conf['data_path']
        self.ipo_await: int = conf['ipo_await']
        self.ls_style: List[str] = conf['ls_style']
        self.ls_indus: List[str] = []
        self.CR = CsvReader(conf)

        # Load views: tradingdate with freq from begin_date to end_date
        self.views: pd.Series = load_tradedate_view(
            eng=conf['mysql_engine']['engine0'],
            bd=self.begin_date,
            ed=self.end_date,
            freq=self.freq
        )

        # Load asset returns
        self.asset_ret: pd.DataFrame = self._load_asset_ret()  # close return, next period
        self.views = self.views.iloc[:-1]  # last period: future return unknown
        self.T: int = len(self.views)

        # Result: pure factor return from begin_date to end_date
        self.expo_panel: pd.DataFrame = pd.DataFrame()
        self.factor_ret: pd.DataFrame = pd.DataFrame()

    def _load_asset_ret(self) -> pd.DataFrame:
        df = self.CR.read_csv(info='close_adj',
                              views=self.views,
                              bd=None,
                              ed=None,
                              local_path=None,
                              d_type=float,
                              cache=True)  # adjusted close price
        # TODO: exclude new IPO
        df = df.pct_change().shift(-1).iloc[:-1]
        # TODO: other winsorize methods
        df[df > 0.11] = 0.11
        df[df < -0.11] = -0.11
        return df

    def upload_results(self, eng: dict, how='insert'):
        """"""
        df = self.factor_ret
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

        df = self.expo_panel[self.ls_style].dropna(how='all')
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
        For every date in views, calculate pure factor return.
        :param cache: bool, True than save local cache
        :return:
        """
        tmp_panel = []
        tmp_fval = []
        print('\nCalculate pure factor returns ...')
        for td0 in tqdm(self.views):
            td = td0.strftime('%Y-%m-%d')
            barra_fval_path = self.data_path + 'barra_fval/' + td + '.csv'
            barra_expo_path = self.data_path + 'barra_panel/' + td + '.csv'
            barra_omega_path = self.data_path + 'barra_omega/' + td + '.csv'

            # Get pure factor return
            if os.path.exists(barra_fval_path) and os.path.exists(barra_expo_path):
                fv_1d = pd.read_csv(barra_fval_path, index_col=0, parse_dates=True)
                expo1do = pd.read_csv(barra_expo_path, index_col=0)
                fv_1d.index.name = 'tradingdate'

            else:

                # Load exposure (z-score, orthogonal)
                if os.path.exists(barra_expo_path):
                    expo1do = pd.read_csv(barra_expo_path, index_col=0)
                    self.ls_indus = [_ for _ in expo1do.columns if _[:4] == 'ind_']
                else:
                    # Load 1-period exposure
                    expo1d = self.get_expo1d(
                        td=td,
                        cache=True
                    )
                    # Orthogonalize
                    expo1do = exposure_orthogonal(
                        beta_exposure_csi=expo1d,
                        ls_style=self.ls_style,
                        ls_indus=self.ls_indus,
                        s_mv_raw='size',
                    )
                    if cache:
                        table_save_safe(df=expo1do, tgt=barra_expo_path, kind='csv')

                # WLS
                pf_w, fv_1d = wls_1d(
                    beta_exposure_csi=expo1do,
                    ls_style=self.ls_style,
                    ls_indus=self.ls_indus,
                    rtn_next_period=self.asset_ret.loc[td],
                    s_mv_raw='size',
                )
                fv_1d = pd.DataFrame(fv_1d.rename(pd.to_datetime(td))).T

                # Save in disk
                if cache:
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

        self.expo_panel = pd.concat(
            [self.expo_panel,
             pd.concat(tmp_panel)]
        )
        self.factor_ret = pd.concat(
            [self.factor_ret,
             pd.concat(tmp_fval)]
        )

    def cache_all_style_exposure(self):
        """Save all views' raw exposure in disk cache path"""
        ls_style = self.ls_style
        for info in ls_style:
            self.CR.read_csv(
                info=info,
                views=self.views,
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
        df = mysql_query(query, engine)
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
            bd = views.iloc[0]
            ed = views.iloc[-1]
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
                views_absent = [_ for _ in views if (_ not in df.index)]
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


def load_tradedate_view(eng: dict, bd: str, ed: str, freq='d') -> pd.Series:
    """
    Load views - tradingdate list from remote
    :param eng: dict, engine0
    :param bd: str, begin date
    :param ed: str, end date
    :param freq: str, frequency - 'd' 'w' or 'm'
    :return: List[str], tradingdate in the range
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

    return pd.to_datetime(df[col])


def load_remote_table(info, eng, bd=None, ed=None, notify=True, views=None) -> pd.DataFrame:
    """
    ..
    :param notify:
    :param info: series of shape (11,)
    :param views: list
    :param eng: dict, sql engine conf
    :param bd: str, begin date
    :param ed: str, end date
    :return:
    """
    engine = conn_mysql(eng)  # TODO: may need close connect

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

    if notify:
        print(query)
    df = mysql_query(query, engine)

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


def mysql_query(query, engine):
    """mysql接口，返回DataFrame"""
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
