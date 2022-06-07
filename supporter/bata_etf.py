"""
(created by swmao on May 23rd)


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
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style("darkgrid")
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
                pass;#print(f"Write in empty dir '{path}'")
            else:
                cmd = input(f"Write in non-empty dir '{path}' ?(y/N)")
                if cmd != 'y' and cmd != 'Y':
                    raise FileExistsError(path)
        else:
            os.makedirs(path, exist_ok=False)
    pass;#print(f'Save in: {path}')


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
            pass;#print("future return + white noise", end='\n\t')
            pass;#print(f"cross-section mu={res.mean(axis=1).mean():.3f} sigma={res.std(axis=1).mean():.3f}")

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

    pass;#print(f"Factor name: {alpha_name}")
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
    pass;#print(f"{lag}-day{' rank ' if ranked else ' '}IC = {dat_ic:.6f}")
    return dat_ic


def get_beta_expo_cnstr(beta_kind, conf, begin_date, end_date, H0, H1, beta_args, l_cvg_fill=True):
    def cvg_f_fill(fr, w=10, q=.75, ishow=False) -> pd.DataFrame:
        """F-Fill if Low Coverage: 日覆盖率过去w日均值的q倍时填充"""
        beta_covered_stk_num = fr.index.get_level_values(0).value_counts().sort_index()
        if ishow:
            beta_covered_stk_num.plot()
            plt.tight_layout()
            plt.show()
        mask_l_cvg = beta_covered_stk_num < (beta_covered_stk_num.shift(1).rolling(w).mean() * q)
        rep_tds = beta_covered_stk_num[mask_l_cvg]
        # print(rep_tds)
        tds = fr.index.get_level_values(0).unique()
        tds = pd.Series(tds, index=tds)
        # td = rep_tds.index[0]
        for td in rep_tds.index:
            td_1 = tds[:td].iloc[-2]
            td1 = tds[td:].iloc[1]
            pass;#print(td.strftime('%Y-%m-%d'), '->', td_1.strftime('%Y-%m-%d'), len(fr.loc[td]), '->', len(fr.loc[td_1]))
            fr = fr.loc[:td_1].append(fr.loc[td_1:td_1].rename(index={td_1: td})).append(fr.loc[td1:])
        return fr

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
            pass;#print(f'\nMerge PCA exposure PN={PN} {suf}')
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


def get_accessible_stk(i: set, a: set, b: set) -> Tuple[list, list, dict]:
    i_a = i.difference(a)
    i_b = i.difference(b)
    res = {'#i_b': len(i_b), '#i_a': len(i_a),
           'i_a': list(i_a), 'i_b': list(i_b)}
    ab = list(a.intersection(b))
    ib = list(i.intersection(b))
    return ab, ib, res


def portfolio_optimize(all_args, telling=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # %%
    tradedates, beta_expo, beta_cnstr, ind_cons, dat, args = all_args
    mkt_type, N, D, B, E, wei_tole, opt_verbose, desc, pos = args

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

    cur_td = 0
    # start_time = time.time()
    loop_bar = tqdm(range(len(tradedates)), ncols=99, desc=desc, delay=0.01, position=pos, ascii=False)
    # %%
    for cur_td in loop_bar:
        # %%
        td = tradedates.iloc[cur_td].strftime('%Y-%m-%d')

        # asset pool accessible
        stk_alpha = get_stk_alpha(dat.loc[td])
        stk_index = set(ind_cons.loc[td].dropna().index)
        stk_beta = set(beta_expo.loc[td].index)
        ls_ab, ls_ib, sp_info = get_accessible_stk(i=stk_index, a=stk_alpha, b=stk_beta)
        ls_clear = list(set(df_lst_w.index).difference(ls_ab))  # 上期持有 组合资产未覆盖
        if telling:
            print(f"\n\t{mkt_type} - alpha({len(stk_alpha)}) = {sp_info['#i_a']} [{','.join(sp_info['i_a'])}]")
            print(f"\t{mkt_type} - beta({len(stk_beta)}) = {sp_info['#i_b']} [{','.join(sp_info['i_b'])}]")
            print(f'\talpha exposed ({len(ls_ab)}/{len(stk_alpha)})')
            print(f'\t{mkt_type.lower()} exposed ({len(ls_ib)}/{len(stk_index)})')
            print(f'\tformer holdings not exposed ({len(ls_clear)}/{len(df_lst_w)}) [{",".join(ls_clear)}]')

        a = dat.loc[td, ls_ab]  # alpha
        wb = ind_cons.loc[td, ls_ib]
        wb /= wb.sum()  # part of index-constituent are not exposed to beta factors; (not) treat them as zero-exposure.
        xf = beta_expo.loc[td].loc[ls_ab].dropna(axis=1)
        f_del = beta_expo.dropna(axis=1).loc[td].loc[ls_ib].T @ wb  # - f_del_overflow
        fl = (f_del + beta_cnstr.loc[f_del.index, 'L']).dropna()
        fh = (f_del + beta_cnstr.loc[f_del.index, 'H']).dropna()

        d_del = df_lst_w.loc[ls_clear].abs().sum().values[0] if len(ls_clear) > 0 else 0
        d = D - d_del

        # %% Constraints
        wN = len(ls_ab)
        w = cp.Variable((wN, 1), nonneg=True)
        opt_cnstr = OptCnstr()
        opt_cnstr.sum_bound(w, e=np.ones([1, wN]), down=None, up=1)  # (1)
        opt_cnstr.sum_bound(w, e=(1 - pd.Series(wb, index=ls_ab).isna()).values.reshape(1, -1), down=B, up=None)  # (3)
        tmp = pd.Series(wb, index=ls_ab).fillna(0).values.reshape(-1, 1)
        opt_cnstr.uni_bound(w, down=tmp - E, up=tmp + E)  # (4)
        del tmp
        opt_cnstr.sum_bound(w, e=xf[fl.index].values.T, down=fl.values.reshape(-1, 1), up=fh.values.reshape(-1, 1))  # (5)

        if len(df_lst_w) > 0:  # turnover constraint
            w_lst = df_lst_w.reindex_like(pd.DataFrame(index=ls_ab, columns=df_lst_w.columns)).fillna(0)
            # w_lst = w_lst - pd.DataFrame(w_overflow, index=ls_ab, columns=df_lst_w.columns).fillna(0)
            w_lst = w_lst.values
            opt_cnstr.norm_bound(w, w0=w_lst, d=d, L=1)  # (2)
        else:
            w_lst = np.zeros([len(ls_ab), 1]) if w_lst is None else w_lst  # former holding

        # %% Solve optimize problem
        objective = cp.Maximize(a.values.reshape(1, -1) @ w)
        constraints = opt_cnstr.get_constraints()
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=opt_verbose, solver='ECOS', max_iters=1000)
        # result = prob.solve(verbose=True, solver='ECOS', max_iters=1000)
        if prob.status == 'optimal_inaccurate':
            result = prob.solve(verbose=opt_verbose, solver='ECOS', max_iters=10000)

        if prob.status == 'optimal':
            w1 = w.value.copy()
            w1[w1 < wei_tole] = 0
            w1 /= w1.sum()
            df_w = pd.DataFrame(w1, index=ls_ab, columns=[td])
            turnover = np.abs(w_lst - df_w.values).sum() + d_del
            hdn = (df_w.values > 0).sum()
            df_lst_w = df_w.replace(0, np.nan).dropna()
        else:
            raise Exception(f'{prob.status} problem')
            # turnover = 0
            # print(f'.{prob.status} problem, portfolio ingredient unchanged')
            # if len(lst_w) > 0:
            #     lst_w.columns = [td]
        holding_weight = pd.concat([holding_weight, df_lst_w.T])

        # Update optimize iteration information
        iter_info = {'#alpha^beta': len(ls_ab), '#index^beta': len(ls_ib), '#index': len(stk_index),
                     'turnover': turnover, 'holding': hdn,
                     'status': prob.status, 'opt0': result, 'opt1': (a @ w1)[0],  # TODO
                     }
        iter_info = iter_info | {'index-alpha': sp_info['#i_a'], 'index-beta': sp_info['#i_b'],
                                 'stk_i_a': ', '.join(sp_info['i_a']), 'stk_i_b': ', '.join(sp_info['i_b'])}
        iter_info = iter_info | f_del.to_dict()
        optimize_iter_info[td] = pd.Series(iter_info)
        # progressbar(cur_td + 1, len(tradedates), msg=f' {td} turnover={turnover:.3f} #stk={hdn}', stt=start_time)

    # %%
    return holding_weight, optimize_iter_info


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

    def __init__(self):
        self.cnstr = []

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
