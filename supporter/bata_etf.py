"""
(created by swmao on May 23rd)
supporter for BarraPCA/optimize.py

"""
import os
import pandas as pd
import numpy as np
import cvxpy as cp
from tqdm import tqdm
import sys
from typing import Tuple
import time

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")

# %matplotlib inline
import warnings
warnings.simplefilter("ignore")
import seaborn
seaborn.set_style("darkgrid")
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(9, 5))
plt.rc("font", size=12)
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rc("savefig", dpi=90)
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"

np.random.seed(9)


def second2clock(x: int):
    if x < 3600:
        return f"{(x // 60):02d}:{x % 60:02d}"
    elif x < 3600 * 24:
        return f"{(x // 3600):02d}:{(x // 60) % 60:02d}:{x % 60:02d}"
    else:
        d = x // (24 * 3600)
        x = x % (24 * 3600)
        return f"{d}d {(x // 3600):02d}:{(x // 60) % 60:02d}:{x % 60:02d}"


def progressbar(cur, total, msg, stt=None):
    """显示进度条"""
    import math
    percent = '{:.2%}'.format(cur / total)
    lth = int(math.floor(cur * 25 / total))
    if stt is None:
        print("\r[%-25s] %s (%d/%d)" % ('=' * lth, percent, cur, total) + msg, end='')
    else:
        time_used = time.time() - stt
        time_left = second2clock(round(time_used / cur * (total - cur)))
        time_used = second2clock(round(time_used))
        print("\r[%-25s] %s (%s<%s)" % ('=' * lth, percent, time_used, time_left) + msg, end='')


def io_make_sub_dir(path, force=False):
    if force:
        os.makedirs(path, exist_ok=True)
    else:
        if os.path.exists(path):
            if os.path.isdir(path) and len(os.listdir(path)) == 0:
                pass;  # print(f"Write in empty dir '{path}'")
            else:
                cmd = input(f"Write in non-empty dir '{path}' ?(y/N)")
                if cmd != 'y' and cmd != 'Y':
                    raise FileExistsError(path)
        else:
            os.makedirs(path, exist_ok=False)
    pass;  # print(f'Save in: {path}')


def get_alpha_dat(alpha_name, mkt_type, conf, begin_date, end_date) -> Tuple[str, pd.DataFrame]:
    """
    Get Alpha Value
    :param alpha_name:
    :param mkt_type:
    :param conf:
    :param begin_date:
    :param end_date:
    :return:

    """

    def get_fval_alpha(_pred_days=5, _wn_m=0., _wn_s=0.) -> pd.DataFrame:
        def wn(s):
            mean = _wn_m
            std = _wn_s
            return np.random.normal(loc=mean, scale=std, size=s)

        close_adj = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True)
        res = close_adj.pct_change(_pred_days).shift(-_pred_days).loc[begin_date: end_date]  # 未来5日的收益
        if (_wn_m == 0) and (_wn_s == 0):
            res = res.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        else:
            res = res.apply(lambda x: (x - x.mean()) / x.std() + wn(len(x)), axis=1)
            pass;  # print("future return + white noise", end='\n\t')
            pass;  # print(f"cross-section mu={res.mean(axis=1).mean():.3f} sigma={res.std(axis=1).mean():.3f}")

        return res

    def get_fval_apm(kind='zscore') -> Tuple[str, pd.DataFrame]:
        """Get APM"""
        res = pd.read_csv(conf['data_path'] + 'factor_apm.csv', index_col=0, parse_dates=True)
        if kind == 'uniform':
            res = res.rank(pct=True, axis=1) * 2 - 1
            # alpha_name = 'APM(-1t1)'
        elif kind == 'zscore':
            res = res.apply(lambda s: (s - s.mean()) / s.std(), axis=1)
            # alpha_name = 'APM'
        elif kind == 'reverse':
            res = res.apply(lambda s: (s.mean() - s) / s.std(), axis=1)
            # alpha_name = 'APM(R)'
        else:
            raise Exception(f'APM kind not in `zscore, uniform, reverse`')
        _alpha_name = f'APM({kind})'
        res = res.shift(1).loc[begin_date: end_date]
        return _alpha_name, res

    pass;  # print(f"Factor name: {alpha_name}")
    save_suffix = f'OptResWeekly[{mkt_type}]{alpha_name}'
    save_path = f"{conf['factorsres_path']}{save_suffix}/"
    os.makedirs(save_path, exist_ok=True)
    alpha_save_name = save_path + f'factor_{alpha_name}.csv'
    if os.path.exists(alpha_save_name):
        dat = pd.read_csv(alpha_save_name, index_col=0, parse_dates=True)
    else:
        a_kind, a_para = alpha_name[:-1].split('(')
        if a_kind[:4] == 'FRtn':  # Use Fake Alpha
            pred_days = int(a_kind[4:-1])
            wn_m, wn_s = [float(_) for _ in a_para.split(',')]
            assert alpha_name == f'FRtn{pred_days}D({wn_m},{wn_s})'
            dat = get_fval_alpha(pred_days, wn_m, wn_s)
        elif a_kind == 'APM':  # Use APM
            a_name, dat = get_fval_apm(kind=a_para)
            assert a_name == alpha_name
        else:
            raise Exception(f'Invalid alpha_name `{alpha_name}`')
        dat.to_csv(alpha_save_name)
    return save_path, dat


def check_ic_5d(closeAdj_path, dat, begin_date, end_date, lag=5, ranked=True) -> float:
    """check IC-5days"""
    from supporter.factor_operator import cal_ic
    close_adj = pd.read_csv(closeAdj_path, index_col=0, parse_dates=True).pct_change()
    close_adj = close_adj.loc[begin_date: end_date]
    dat_ic = cal_ic(fv_l1=dat, ret=close_adj, lag=lag, ranked=ranked).mean()[0]
    pass;  # print(f"{lag}-day{' rank ' if ranked else ' '}IC = {dat_ic:.6f}")
    return dat_ic


def get_beta_expo_cnstr(beta_kind, conf, begin_date, end_date, H0, H1, beta_args, l_cvg_fill=True):
    from supporter.transformer import cvg_f_fill

    def get_barra_exposure() -> pd.DataFrame:
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
        expo = pd.DataFrame()
        begin_year = int(begin_date.split('-')[0])
        end_year = int(end_date.split('-')[0])
        for _ in range(begin_year, end_year + 1):
            df = pd.DataFrame(pd.read_hdf(conf['barra_panel'], key=f'y{_}'))
            expo = pd.concat([expo, df])
        expo: pd.DataFrame = expo.loc[begin_date: end_date]

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
                df = df.loc[begin_date: end_date]
                expo = pd.concat([expo, df.stack().rename(kw)], axis=1)
            #
            expo.to_pickle(tgt)
            return expo

        def get_suffix(s):
            s1 = s.split('-')
            return s1[0][-2:] + s1[1]

        suffix = f"{get_suffix(begin_date)}_{get_suffix(end_date)}"
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
        expo_beta = get_barra_exposure()
        sty_c = beta_args[0]  # ['size', 'beta', 'momentum']
        ind_c = [c for c in expo_beta.columns if 'ind' == c[:3]]
        # cnstr_info = [(sty_c, expoL, expoH), (ind_c, expoL, expoH)]
        cnstr_info = [(sty_c, -H0, H0), (ind_c, -H1, H1)]
        cnstr_beta = get_beta_constraint(all_c=expo_beta.columns, info=cnstr_info)

    elif beta_kind == 'PCA':
        principal_number = beta_args[0]  # 20
        expo_beta = get_pca_exposure(PN=principal_number)
        cnstr_info = [(list(expo_beta.columns), -H0, H0)]
        cnstr_beta = get_beta_constraint(all_c=expo_beta.columns, info=cnstr_info)

    else:
        raise Exception('beta_kind {Barra, PCA}')

    if l_cvg_fill:  # beta覆盖不足，用上一日的beta暴露填充
        expo_beta = cvg_f_fill(expo_beta, w=10, q=.75, ishow=False)

    return expo_beta, cnstr_beta


def get_index_constitution(src, begin_date, end_date) -> pd.DataFrame:
    """Baseline Portfolio 股指成分股权重"""
    ind_cons = pd.read_csv(src, index_col=0, parse_dates=True)
    ind_cons = ind_cons.loc[begin_date: end_date]
    ind_cons = ind_cons.dropna(how='all', axis=1)
    # ind_cons = ind_cons.fillna(0)
    ind_cons /= 100
    return ind_cons


def get_tradedates(conf, begin_date, end_date, kind) -> pd.Series:
    tdays_d = pd.read_csv(conf[kind], header=None, index_col=0, parse_dates=True)
    tdays_d = tdays_d.loc[begin_date: end_date]
    tdays_d['tdays_d'] = tdays_d.index
    tradedates = tdays_d.tdays_d
    return tradedates


def get_accessible_stk(i: set, a: set, b: set, s: set = None) -> Tuple[list, list, dict]:
    bs = b.intersection(s) if s else b  # beta (and sigma)
    i_a = i.difference(a)  # idx cons w/o alpha
    i_b = i.difference(bs)  # idx cons w/o beta (and sigma)
    info = {'#i_a': len(i_a), 'i_a': list(i_a),
            '#i_b(s)': len(i_b), 'i_b(s)': list(i_b)}
    pool = list(a.intersection(bs))  # accessible asset pool
    base = list(i.intersection(bs))  # index component with beta (and sigma)
    return pool, base, info


def tf_portfolio_weight(portfolio_weight, tab_path, gra_path, ishow=False):
    """Table & Graph: Holding weight, holding number"""
    portfolio_weight.sum(axis=1)
    portfolio_weight.to_csv(tab_path)
    # (portfolio_weight > 0).sum(axis=1).plot()
    # plt.tight_layout()
    # plt.savefig(gra_path)  # dat_path + g_file.format(suffix))
    # if ishow:
    #     plt.show()
    # else:
    #     plt.close()


def tf_historical_result(close_adj, tradedates, begin_date, end_date, portfolio_weight, ind_cons, mkt_type, gra_path,
                         tab_path):
    """Table & Graph: wealth curve comparison"""
    df = pd.DataFrame()

    # rtn_w2w = close_adj.pct_change(5).loc[tradedates.loc[begin_date: end_date].index]
    rtn_w2w = close_adj.loc[tradedates.index].pct_change().shift(-1).loc[begin_date: end_date]
    rtn_portfolio = (portfolio_weight * rtn_w2w.reindex_like(portfolio_weight)).sum(axis=1)
    df['portfolio'] = rtn_portfolio

    ind_cons_w = ind_cons.loc[tradedates.index].loc[begin_date: end_date]
    rtn_ind = (ind_cons_w * rtn_w2w.reindex_like(ind_cons_w)).sum(axis=1)
    df[mkt_type] = rtn_ind
    df['Excess'] = df['portfolio'] - df[mkt_type]

    tmp = df.cumsum().add(1)
    tmp.plot()
    plt.tight_layout()
    plt.savefig(gra_path)
    plt.close()

    tmp.to_excel(tab_path)
    del tmp


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


def get_risk_matrix(path, td, max_backward=7, notify=False) -> pd.DataFrame:
    """stock risk matrix on ${td}"""
    if path is None:
        return pd.DataFrame()

    def _get_mat(_tdd):
        k = f"TD{_tdd.strftime('%Y%m%d')}"
        p = path.format(k)
        if os.path.exists(p):
            return pd.DataFrame(pd.read_hdf(p, key=k))
        else:
            return pd.DataFrame()

    tdd = td0 = pd.to_datetime(td)
    df = _get_mat(tdd)
    cnt = 0
    while (len(df) < 100) and (cnt < max_backward):
        from datetime import timedelta
        tdd -= timedelta(1)
        df = _get_mat(tdd)
        cnt += 1
    if notify and (tdd != td0):
        print(f"replace insufficient risk matrix {td0.strftime('%Y-%m-%d')} <- {tdd.strftime('%Y-%m-%d')}")

    return df.loc[tdd]


def get_factor_covariance(path_F, bd=None, ed=None, fw=0) -> pd.DataFrame:
    """
    mat F: Sigma = X F X - D
    :param path_F: .../F_NW_Eigen_VRA[yyyy-mm-dd,yyyy-mm-dd].csv
    :param bd: begin date
    :param ed: end date
    :param fw:
    :return: factor covariance matrix or matrices, like
    """
    df = pd.read_csv(path_F, index_col=[0, 1], parse_dates=[0])
    if fw > 0:
        df = df.groupby(['names']).shift(fw)
        # df = df.dropna(how='all')
        df = df.loc[df.index.get_level_values(0).unique()[fw]:]
        # df.index.get_level_values(0).value_counts().sort_index().plot()
        # plt.tight_layout()
        # plt.show()
    df = df.loc[bd:] if bd is not None else df
    df = df.loc[:ed] if ed is not None else df

    return df


def get_specific_risk(path_D, bd=None, ed=None, fw=0) -> pd.DataFrame:
    """
    mat D: Sigma = X F X - D
    :param path_D: .../D_NW_SM_SH_VRA[yyyy-mm-dd,yyyy-mm-dd].csv
    :param bd:
    :param ed:
    :param fw:
    :return: dataframe of diag item of D

    """
    df = pd.read_csv(path_D, index_col=0, parse_dates=True)
    df = df.shift(fw).iloc[fw:] if fw > 0 else df
    df = df.loc[bd:] if bd is not None else df
    df = df.loc[:ed] if ed is not None else df
    return df


def info2suffix(ir1: pd.Series) -> str:
    """suffix for all result file"""
    return f"{ir1['beta_suffix']}(B={ir1['B']},E={ir1['E']},D={ir1['D']},H0={ir1['H0']}" + \
           ('' if np.isnan(float(ir1['H1'])) else f",H1={ir1['H1']}") + \
           (f",G={ir1['G']}" if float(ir1['G']) > 0 else '') + \
           (f",S={ir1['S']}" if float(ir1['S']) < np.inf else '') + ')'


class PortfolioOptimizer(object):

    def __init__(self, conf: dict, OPTIMIZE_TARGET: str, notify=False, mkdir_force=False):
        self.init_time = time.time()
        self.conf: dict = conf
        self.notify: bool = notify
        self.mkdir_force = mkdir_force
        self.opt_tgt: pd.DataFrame = pd.DataFrame()

        self.load_optimize_target(src=OPTIMIZE_TARGET)

    def load_optimize_target(self, src: str):
        self.opt_tgt = pd.read_excel(src, index_col=0, dtype=object).loc[1:1]
        if not self.notify:
            print(self.opt_tgt)

    def begin_optimize(self, p_num=1):
        if p_num > 1:
            from multiprocessing import freeze_support, Pool, RLock
            print(f'father process {os.getpid()}')
            freeze_support()
            p = Pool(p_num, initializer=tqdm.set_lock, initargs=(RLock(),))
            cnt = 0
            for ir in self.opt_tgt.iterrows():
                ir1 = ir[1]
                p.apply_async(self.optimize1,
                              args=[(self.conf, ir1, self.mkdir_force, cnt % p_num)])
                cnt += 1
            p.close()
            p.join()
        else:
            for ir in self.opt_tgt.iterrows():
                ir1 = ir[1]
                args = (self.conf, ir1, self.mkdir_force, 0)
                self.optimize1(args)

    @staticmethod
    def optimize1(args):
        optimizer = Optimizer(args)
        optimizer.optimize()


class Optimizer(object):

    def __init__(self, args):
        self.conf: dict = args[0]
        self.mkdir_force: bool = args[2]
        self.pos: int = args[3]

        ir1: pd.Series = args[1]
        self.mkt_type = ir1['mkt_type']
        self.begin_date = ir1['begin_date']
        self.end_date = ir1['end_date']
        self.opt_verbose = (ir1['opt_verbose'] == 'TRUE')
        self.wei_tole = float(ir1['wei_tole'])
        self.N = float(ir1['N'])
        self.B = float(ir1['B']) / 100
        self.E = float(ir1['E']) / 100
        self.H0 = float(ir1['H0'])
        self.H1 = float(ir1['H1'])
        self.D = float(ir1['D'])
        self.G = float(ir1['G']) * 1e6
        self.S = float(ir1['S'])
        self.alpha_name = ir1['alpha_name']
        self.beta_kind = ir1['beta_kind']
        self.suffix = info2suffix(ir1)

        self.info = {
            'alpha_name': self.alpha_name,
            'beta_kind': self.beta_kind,
            'begin_date': self.begin_date,
            'end_date': self.end_date,
            'mkt_type': self.mkt_type,
            'N': self.N,
            'H0': self.H0,
            'H1': self.H1,
            'B': self.B,
            'E': self.E,
            'D': self.D,
            'G': self.G,
            'S': self.S,
            'wei_tole': self.wei_tole,
            'opt_verbose': self.opt_verbose,
            'suffix': self.suffix,
        }
        self.beta_args = eval(ir1['beta_args'])

    def optimize(self):
        pass

