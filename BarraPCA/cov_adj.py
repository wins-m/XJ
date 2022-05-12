"""
(created by swmao on April 28th)
风险矩阵估计，包括
- 共同因子协方差矩阵 MFM
- 特异风险方差矩阵 SRR
"""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import os, sys, time
from tqdm import tqdm
from multiprocessing import Pool

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")

import warnings

warnings.simplefilter("ignore")

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style("darkgrid")
# plt.rc("figure", figsize=(16, 6))
plt.rc("figure", figsize=(8, 3))
plt.rc("savefig", dpi=90)
# plt.rc("font", family="sans-serif")
# plt.rc("font", size=12)
plt.rc("font", size=10)

plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"


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
        gamma1 = [weights[i + t] * ret1[t].T @ ret1[t+i] for t in range(T - i)]
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
    gamma0 = (ret1**2 * weights).sum()

    v = gamma0.copy()
    for i in range(1, q+1):
        ret_i = ret1 * ret1.shift(i)
        weights_i = .5 ** (np.arange(T - 1, -1, -1) / tau)
        weights_i = (~ret_i.isna()) * weights_i.reshape(-1, 1)  # w on all stocks, w=0 if missing
        weights_i /= weights_i.sum()
        gamma_i = (ret_i * weights_i).sum()
        v += (1 - i / (1 + q)) * (gamma_i + gamma_i)

    sigma_raw = pd.Series(gamma0.apply(np.sqrt), index=names)
    sigma_nw = pd.Series(v.apply(np.sqrt), index=names)
    return sigma_raw, sigma_nw


def var_struct_mod_adj(U: pd.DataFrame, sigNW: pd.DataFrame, expo: pd.DataFrame, MV: pd.DataFrame, E=1.05) -> Tuple[pd.Series, pd.Series]:
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
    sigEq = U[(U >= -10*sigTilde) & (U <= 10*sigTilde)].std()
    Z = (sigEq / sigTilde - 1).abs()

    gamma = Z.apply(lambda _: np.nan if np.isnan(_) else min(1., max(0., (h-60)/120)) * min(1., max(0., np.exp(1 - _))))
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
    mat_r[-1:, -len(factor_i)+1:] = -mv_indus[:-1] / mv_indus[-1]
    mat_omega = mat_r @ np.linalg.inv(mat_r.T @ mat_x.T @ mat_v @ mat_x @ mat_r) @ mat_r.T @ mat_x.T @ mat_v

    mat_y = Y.values
    mat_b = mat_omega @ mat_y.reshape(-1, 1)
    b_hat = pd.DataFrame(mat_b, index=X.columns)
    sigSTR = E * np.exp(expo[factor_cs + factor_i] @ b_hat)
    sigma_hat = (gamma * sigNW + (1 - gamma) * sigSTR.iloc[:, 0]).dropna()

    # %
    return gamma, sigma_hat


def var_baysian_shrink(sigSM: pd.Series, mv: pd.Series, gn=10, q=1) -> pd.Series:
    """Baysian Shrinkage"""
    mv_group = mv.rank(pct=True, ascending=False).apply(lambda x: (1-x)//(1/gn))  # low-rank: small size
    # print(mv_group.isna().sum())
    tmp = pd.DataFrame(sigSM.rename('sig_hat'))
    tmp['mv'] = mv
    tmp['g'] = mv_group
    tmp = tmp.reset_index()
    tmp = tmp.merge(tmp.groupby('g')['mv'].sum().rename('mv_gsum').reset_index(), on='g', how='left')
    tmp['w'] = tmp['mv'] / tmp['mv_gsum']
    tmp = tmp.merge((tmp['w'] * tmp['sig_hat']).groupby(tmp['g']).sum().rename('sig_bar').reset_index(), on='g', how='left')
    tmp['sig_d'] = tmp['sig_hat'] - tmp['sig_bar']
    tmp = tmp.merge((tmp['sig_d']**2).groupby(tmp['g']).mean().apply(np.sqrt).rename('D').reset_index(), on='g', how='left')
    tmp['v'] = tmp['sig_d'].abs() * q / (tmp['D'] + tmp['sig_d'].abs() * q)
    tmp['sig_sh'] = tmp['v'] * tmp['sig_bar'] + (1 - tmp['v']) * tmp['sig_hat']
    tmp = tmp.set_index('index')
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
    Y1 = pd.DataFrame(Y0)
    U = Y.reindex_like(Y1) - Y1  # T*N specific returns
    # U.isna().sum().plot.hist(bins=100, title='Missing U=Y-XF')
    # plt.show()
    return U


class MFM(object):
    """
    纯因子方差调整
    fr: DataFrame 形如（最新日的因子收益率未知）
                 country      size  ...  ind_CI005028.WI  ind_CI005029.WI
    2022-03-25 -0.000808 -0.000332  ...         0.014670        -0.001120
    2022-03-28 -0.006781  0.001878  ...        -0.003114        -0.001607
    2022-03-29  0.013088  0.005239  ...        -0.002646         0.002034
    2022-03-30  0.001433 -0.007067  ...        -0.000453         0.006430
    2022-03-31        NA        NA  ...               NA               NA

    """
    def __init__(self, fr: pd.DataFrame = None):
        self.factor_ret = fr
        self.sorted_dates = pd.to_datetime(fr.index.to_series())
        self.T = len(fr)

        self.Newey_West_adj_cov: Dict[str, pd.DataFrame] = dict()
        # self.Newey_West_adj_cov: pd.DataFrame = pd.DataFrame()
        self.eigen_risk_adj_cov: Dict[str, pd.DataFrame] = dict()
        self.vol_regime_adj_cov: Dict[str, pd.DataFrame] = dict()

    def newey_west_adj_by_time(self, h=252, tau=90, q=2) -> Dict[str, pd.DataFrame]:
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
        print('\n\nNewey West Adjust...')
        for t in range(h, self.T):
            td = self.sorted_dates[t].strftime('%Y-%m-%d')
            try:
                # cov = cov_newey_west_adj(self.factor_ret[t - h:t], tau=tau, q=q)
                # self.Newey_West_adj_cov = self.Newey_West_adj_cov.append(frame1d_2d(cov, td))
                self.Newey_West_adj_cov[td] = cov_newey_west_adj(self.factor_ret[t-h:t], tau=tau, q=q)
            except:
                self.Newey_West_adj_cov[td] = (pd.DataFrame())

            progressbar(cur=t-h, total=self.T-h, msg=f'\tdate: {self.sorted_dates[t-1].strftime("%Y-%m-%d")}')

        return self.Newey_West_adj_cov

    def eigen_risk_adj_by_time(self, T=1000, M=100, scal=1.4) -> Dict[str, pd.DataFrame]:
        """
        逐个F_NW进行 Eigenfactor Rist Adjustment
        :param T: 模拟序列长度
        :param M: 模拟次数
        :param scal: scale coefficient for bias
        :return: dict, key=日期, val=协方差 F_Eigen
        """
        if len(self.Newey_West_adj_cov) == 0:
            raise Exception('run newey_west_adj_by_time first for F_NW')

        print('\n\nEigen-value Risk Adjust...')
        cnt = 0
        td = '2019-03-19'
        for td in self.Newey_West_adj_cov.keys():
            try:
                cov = self.Newey_West_adj_cov[td]
                self.eigen_risk_adj_cov[td] = cov_eigen_risk_adj(cov=cov, T=T, M=M, scal=scal)
            except:
                self.eigen_risk_adj_cov[td] = pd.DataFrame()

            cnt += 1
            progressbar(cnt, len(self.Newey_West_adj_cov), f'\tdate: {td}')

        return self.eigen_risk_adj_cov

    def vol_regime_adj_by_time(self, h=252, tau=42) -> Dict[str, pd.DataFrame]:
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
        tradedates = list(self.eigen_risk_adj_cov.keys())
        for td in tradedates:
            f_var_i = np.diag(self.eigen_risk_adj_cov[td])
            if len(f_var_i) == 0:
                f_var_i = np.array(K * [np.nan])
            factor_var.append(f_var_i)

        factor_var = np.array(factor_var)
        B2 = (self.factor_ret.loc[tradedates]**2 / factor_var).mean(axis=1)

        weights = .5**(np.arange(h-1, -1, -1) / tau)
        weights /= weights.sum()
        # lamb2 = {}
        print('\n\nVolatility Regime Adjustment...')
        cnt = 0
        for td0, td1 in zip(tradedates[:-h], tradedates[h-1:]):
            # lamb2[td1] = B2.loc[td0: td1] @ weights
            lamb2 = B2.loc[td0: td1] @ weights
            self.vol_regime_adj_cov[td1] = self.eigen_risk_adj_cov[td1] * lamb2
            cnt += 1
            progressbar(cnt, len(self.eigen_risk_adj_cov) - h, f'\tdate: {td1}')

        return self.vol_regime_adj_cov

    def save_vol_regime_adj_cov(self, path):
        """存储"""
        def frame1d_2d(df: pd.DataFrame, td: str) -> pd.DataFrame:
            df['names'] = df.index
            df['tradingdate'] = td
            df = df.set_index(['tradingdate', 'names'])
            return df

        def dict2frame(cov: Dict[str, pd.DataFrame]) -> pd.DataFrame:
            """字典形式存储的换到Frame"""
            tmp = pd.DataFrame()
            for td in cov.keys():
                df = cov[td]
                df = frame1d_2d(df, td)
                tmp = tmp.append(df)
            return tmp

        if len(self.vol_regime_adj_cov) == 0:
            raise Exception('run vol_regime_adj_by_time first for F_VRA')
        cov = dict2frame(self.vol_regime_adj_cov)
        file_name = f"F_NW_Eigen_VRA[{','.join(list(cov.index.get_level_values(0)[[0, -1]]))}].csv"
        cov.to_csv(path + '/' + file_name)


class SRR(object):
    """Specific Return Risk"""

    def __init__(self, sr, fr, expo, mv):
        self.stk_rtn: pd.DataFrame = sr
        self.fct_rtn: pd.DataFrame = fr
        self.exposure: pd.DataFrame = expo
        self.mkt_val: pd.DataFrame = mv

        self.sorted_dates = pd.to_datetime(fr.index.to_series())
        self.T = len(fr)

        self.u: pd.DataFrame = pd.DataFrame()
        self.SigmaRaw: pd.DataFrame = pd.DataFrame()
        self.SigmaNW: pd.DataFrame = pd.DataFrame()
        self.GammaSM: pd.DataFrame = pd.DataFrame()
        self.SigmaSM: pd.DataFrame = pd.DataFrame()
        self.SigmaSH: pd.DataFrame = pd.DataFrame()
        self.LambdaVRA: pd.DataFrame = pd.DataFrame()
        self.SigmaVRA: pd.DataFrame = pd.DataFrame()

    def specific_return_by_time(self):
        print('\n\nSpecific Return...')
        self.u = specific_return_yxf(Y=self.stk_rtn, X=self.exposure, F=self.fct_rtn)
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
        print('\n\nNewey West Adjust...')
        SigmaRaw = []
        SigmaNW = []
        t = h
        for t in range(h, self.T):
            td = self.sorted_dates[t].strftime('%Y-%m-%d')
            try:
                u0 = self.u.iloc[t-h:t]
                # u0.count().plot.hist(bins=100, title=f'{u0.index[0].strftime("%Y-%m-%d")},{td}')
                # plt.show()
                u1 = u0[u0.columns[u0.count() > h*NA_bar]]
                sigma_raw, sigma_nw = var_newey_west_adj(ret=u1, tau=tau, q=q)
                SigmaRaw.append(sigma_raw.rename(td))
                SigmaNW.append(sigma_nw.rename(td))
            except:
                SigmaRaw.append(pd.Series([]).rename(td))
                SigmaNW.append(pd.Series([]).rename(td))

            progressbar(cur=t-h+1, total=self.T-h, msg=f'\tdate: {td}')

        self.SigmaRaw = pd.DataFrame(SigmaRaw)
        self.SigmaNW = pd.DataFrame(SigmaNW)
        self.SigmaRaw.index = pd.to_datetime(self.SigmaRaw.index.to_series())
        self.SigmaNW.index = pd.to_datetime(self.SigmaNW.index.to_series())
        return self.SigmaRaw, self.SigmaNW

    def struct_mod_adj_by_time(self, h=252, NA_bar=.75, E=1.05):
        if len(self.SigmaNW) == 0:
            raise Exception('No Newey-West Adjusted Sigma SigmaNW')

        print('\n\nStructural Model Adjust...')
        GammaSM = []
        SigmaSM = []
        cnt = 0
        # td = self.sorted_dates[-2]
        for td in self.sorted_dates[h: self.T]:
            # %
            sigNW = self.SigmaNW.loc[td].dropna()
            U = self.u.loc[:td].iloc[-h:]
            U = U.loc[:, U.count() > h * NA_bar]
            expo = self.exposure.loc[td].dropna(axis=1, how='all').dropna(axis=0, how='any')  # 全空的风格暴露&有空的个股
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
            progressbar(cur=cnt, total=self.T-h, msg=f'\tdate: {td.strftime("%Y-%m-%d")}')

        self.SigmaSM = pd.DataFrame(SigmaSM)
        self.GammaSM = pd.DataFrame(GammaSM)

        return self.GammaSM, self.SigmaSM

    def baysian_shrink_by_time(self, q=1, gn=10):
        T = len(self.SigmaSM)
        if T == 0:
            raise Exception('No Newey-West Adjusted Sigma SigmaNW')

        print('\n\nBaysian Shrink Adjust...')
        cnt = 0
        SigmaSH = []
        # td = self.SigmaSM.index[-1]
        for td in self.SigmaSM.index:
            MV = self.mkt_val.loc[td]
            sigSM = self.SigmaSM.loc[td].rename('sig_hat')
            mv: pd.Series = MV[sigSM.index]  # TODO: why MV=NA?
            # print(mv.isna().sum())
            try:
                SigmaSH.append(var_baysian_shrink(sigSM=sigSM, mv=mv, q=q, gn=gn).rename(td))
            except:
                SigmaSH.append(pd.Series([]).rename(td))
            cnt += 1
            progressbar(cur=cnt, total=T, msg=f'\tdate: {td.strftime("%Y-%m-%d")}')

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
        print('\n\nVolatility Regime Adjustment...')
        for td0, td1 in zip(tradedates[:-h], tradedates[h - 1:]):
            # lamb2[td1] = B2.loc[td0: td1] @ weights
            lamb = np.sqrt(B2.loc[td0: td1] @ weights)
            Lambda[td1] = lamb
            SigmaVRA[td1] = self.SigmaSH.loc[td1] * lamb
            cnt += 1
            progressbar(cnt, len(tradedates) - h, f'\tdate: {td1.strftime("%Y-%m-%d")}')
        self.LambdaVRA = Lambda.T
        self.SigmaVRA = SigmaVRA.T
        return self.LambdaVRA, self.SigmaVRA

    def cal_volatility_cross_section(self):
        sr_mv = (self.mkt_val.reindex_like(self.u) * (1 - self.u.isna())).apply(lambda s: s / s.sum(), axis=1)
        sr_mv @ self.u**2
        pass

    def plot_structural_model_gamma(self):
        if len(self.GammaSM) == 0:
            raise Exception('No Structural Model Gamma')
        g = self.GammaSM.copy()
        ratio = (g == 1).sum(axis=1) / g.count(axis=1)
        ratio.plot(title='ratio of good-quality specific return (with $\gamma=1$)')
        plt.show()


# %%
def main():
    # %%
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    fr = get_barra_factor_return_daily(conf)
    mkt_val = pd.read_csv(conf['marketvalue'], index_col=0, parse_dates=True)
    sr, exposure = get_barra_factor_exposure_daily(conf, use_temp=True)  # 注意T0期因子收益对应T+1期个股收益

    # %%
    self = MFM(fr.iloc[-1000:])
    Newey_West_adj_cov = self.newey_west_adj_by_time()
    eigen_risk_adj_cov = self.eigen_risk_adj_by_time()
    vol_regime_adj_cov = self.vol_regime_adj_by_time()
    self.Newey_West_adj_cov = Newey_West_adj_cov
    self.eigen_risk_adj_cov = eigen_risk_adj_cov
    self.vol_regime_adj_cov = vol_regime_adj_cov
    self.save_vol_regime_adj_cov(conf['dat_path_barra'])

    # %%
    fbegin_date = '2012-01-01'  # '2019-07-01'
    self = SRR(fr=fr.loc[fbegin_date:], sr=sr.loc[fbegin_date:], expo=exposure.loc[fbegin_date:], mv=mkt_val.loc[fbegin_date:])
    print(self.T)
    # Ret_U = self.specific_return_by_time()
    self.u = Ret_U
    # Raw_var, Newey_West_adj_var = self.newey_west_adj_by_time()
    self.SigmaRaw, self.SigmaNW = Raw_var, Newey_West_adj_var
    # Gamma_STR, Sigma_STR = self.struct_mod_adj_by_time()
    self.GammaSM, self.SigmaSM = Gamma_STR, Sigma_STR
    # self.plot_structural_model_gamma()
    # self.SigmaSM.count(axis=1).plot(); plt.show()
    # Sigma_Shrink = self.baysian_shrink_by_time()
    self.SigmaSH = Sigma_Shrink
    Lambda_VRA = Sigma_VRA = self.vol_regime_adj_by_time()
    # self.SigmaVRA = Sigma_VRA


# %% TODO: Plot and Check
tmp = pd.DataFrame()
for k, sr in zip(['SigmaRaw', 'SigmaNW', 'SigmaSM', 'SigmaSH', 'SigmaVRA'],
                 [self.SigmaRaw, self.SigmaNW, self.SigmaSM, self.SigmaSH, self.SigmaVRA]):
    B = (self.u.reindex_like(sr) / sr).rolling(21).std()
    w = (self.mkt_val.reindex_like(B) * (1 - B.isna())).apply(lambda s: s / s.sum(), axis=1)
    tmp = tmp.append((B * w).sum(axis=1).rolling(120).mean().rename(k))
    # tmp = tmp.append((self.u.reindex_like(sr) / sr).rolling(21).std().mean(axis=1).rename(k))
tmp = tmp.T
tmp.plot()
plt.show()


