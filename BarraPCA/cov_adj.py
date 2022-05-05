"""
(created by swmao on April 28th)

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
    gamma0 = [weights[t] * ret1[t].T @ ret1[i+t] for t in range(T)]
    v = np.array(gamma0).sum(0)

    for i in range(1, q + 1):
        gamma1 = [weights[i + t] * ret1[t].T @ ret1[t] for t in range(T - i)]
        cd = np.array(gamma1).sum(0)
        v += (1 - i / (1 + q)) * (cd + cd.T)

    return pd.DataFrame(v, columns=names, index=names)


def var_newey_west_adj(ret, tau=90, q=5) -> Tuple[pd.Series, pd.Series]:
    """
    Newey-West调整时序上相关性
    :param ret: column为特异收益，index为时间，一般取T-252,T-1
    :param tau: 协方差计算半衰期
    :param q: 假设因子收益q阶MA过程
    :return: 经调整后的协方差矩阵
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

    sigma_raw = pd.Series(gamma0, index=names)
    sigma_nw = pd.Series(v, index=names)
    return sigma_raw, sigma_nw


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
        progressbar(cnt, F.shape[0], msg=f'\tdates: {td.strftime("%Y-%m-%d")}')
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

    def __init__(self, sr, fr, expo):
        self.stk_rtn: pd.DataFrame = sr
        self.fct_rtn: pd.DataFrame = fr
        self.exposure: pd.DataFrame = expo

        self.sorted_dates = pd.to_datetime(fr.index.to_series())
        self.T = len(fr)

        self.u: pd.DataFrame = pd.DataFrame()
        self.SigmaRaw: pd.DataFrame = pd.DataFrame()
        self.SigmaNW: pd.DataFrame = pd.DataFrame()

    def specific_return_by_time(self):
        self.u = specific_return_yxf(Y=self.stk_rtn, X=self.exposure, F=self.fct_rtn)

    def newey_west_adj_by_time(self, h=252, NA_bar=.75, tau=90, q=5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        特异收益率全历史计算协方差进行 Newey West 调整
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

            progressbar(cur=t-h+1, total=self.T-h, msg=f'\tdate: {self.sorted_dates[t-1].strftime("%Y-%m-%d")}')

        self.SigmaRaw = pd.DataFrame(SigmaRaw)
        self.SigmaNW = pd.DataFrame(SigmaNW)

        return self.SigmaRaw, self.SigmaNW


def main():
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    fr = get_barra_factor_return_daily(conf)
    self = MFM(fr.iloc[-1000:])
    Newey_West_adj_cov = self.newey_west_adj_by_time()
    eigen_risk_adj_cov = self.eigen_risk_adj_by_time()
    vol_regime_adj_cov = self.vol_regime_adj_by_time()
    self.Newey_West_adj_cov = Newey_West_adj_cov
    self.eigen_risk_adj_cov = eigen_risk_adj_cov
    self.vol_regime_adj_cov = vol_regime_adj_cov
    self.save_vol_regime_adj_cov(conf['dat_path_barra'])

    sr, expo = get_barra_factor_exposure_daily(conf, use_temp=True)  # 注意T0期因子收益对应T+1期个股收益
    self = SRR(fr=fr.loc['2019-07-01':], sr=sr.loc['2019-07-01':], expo=expo.loc['2019-07-01':])
    print(self.T)
    self.specific_return_by_time()
    # Raw_var, Newey_West_adj_var = self.newey_west_adj_by_time()
    self.SigmaRaw = Raw_var
    self.SigmaNW = Newey_West_adj_var

    sig = Newey_West_adj_var.copy()


# if __name__ == '__main__':
#     main()
