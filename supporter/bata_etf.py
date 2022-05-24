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
import math

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.factor_operator import cal_ic

# %matplotlib inline
import warnings

warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style("darkgrid")
plt.rc("figure", figsize=(8, 5))
plt.rc("savefig", dpi=90)
plt.rc("font", size=10)
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"

np.random.seed(9)


def progressbar(cur, total, msg):
    """显示进度条"""
    import math
    percent = '{:.2%}'.format(cur / total)
    lth = int(math.floor(cur * 25 / total))
    print("\r[%-25s] %s (%d/%d)" % ('=' * lth, percent, cur, total) + msg, end='')


def get_fval_alpha(conf, begin_date, end_date, pred_days=5, wn_m=0., wn_s=0.) -> pd.DataFrame:
    def wn(s):
        mean = wn_m
        std = wn_s
        return np.random.normal(loc=mean, scale=std, size=s)

    close_adj = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True)
    dat = close_adj.pct_change(pred_days).shift(-pred_days).loc[begin_date: end_date]  # 未来5日的收益
    if (wn_m == 0) and (wn_s == 0):
        dat = dat.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    else:
        dat = dat.apply(lambda x: (x - x.mean()) / x.std() + wn(len(x)), axis=1)
        print("future return + white noise", end='\n\t')
        print(f"cross-section mu={dat.mean(axis=1).mean():.3f} sigma={dat.std(axis=1).mean():.3f}")

    return dat


def get_fval_apm(conf, begin_date, end_date, kind='zscore') -> Tuple[str, pd.DataFrame]:
    dat = pd.read_csv(conf['data_path'] + 'factor_apm.csv', index_col=0, parse_dates=True)
    if kind == 'uniform':
        dat = dat.rank(pct=True, axis=1) * 2 - 1
        alpha_name = 'APM(-1t1)'
    elif kind == 'zscore':
        dat = dat.apply(lambda s: (s - s.mean()) / s.std(), axis=1)
        alpha_name = 'APM'
    elif kind == 'reverse':
        dat = dat.apply(lambda s: (s.mean() - s) / s.std(), axis=1)
        alpha_name = 'APM(R)'
    else:
        raise Exception(f'APM kind not in `zscore, uniform, reverse`')
    dat = dat.loc[begin_date: end_date].shift(1)
    return alpha_name, dat


def check_ic_5d(conf, dat, begin_date, end_date, lag=5) -> float:
    """check IC-5days"""
    close_adj = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True).pct_change()
    close_adj = close_adj.loc[begin_date: end_date]
    dat_ic = cal_ic(fv_l1=dat, ret=close_adj, lag=lag, ranked=True).mean()[0]
    print(f"{lag}-day rank IC = {dat_ic:.6f}")
    return dat_ic


def get_style_indus_exposure(conf, begin_date, end_date) -> Tuple[pd.DataFrame, pd.DataFrame]:
    expo = pd.DataFrame()
    for _ in range(int(begin_date.split('-')[0]), int(end_date.split('-')[0]) + 1):
        expo = pd.concat([expo, pd.DataFrame(pd.read_hdf(conf['barra_panel'], key=f'y{_}'))])
    expo: pd.DataFrame = expo.loc[begin_date: end_date]

    cols_style = [c for c in expo.columns if 'rtn' not in c and 'ind' not in c and 'country' != c]
    cols_indus = [c for c in expo.columns if 'ind' in c]
    expo_style: pd.DataFrame = expo[cols_style].groupby(
        expo[cols_style].index.get_level_values(0)).apply(lambda s: (s - s.mean()) / s.std())
    expo_indus: pd.DataFrame = expo[cols_indus]

    return expo_style, expo_indus


def get_index_constitution(conf, mkt_type, begin_date, end_date) -> pd.DataFrame:
    """Baseline Portfolio 股指成分"""
    _ = conf['idx_constituent'].format(mkt_type)
    ind_cons = pd.read_csv(_, index_col=0, parse_dates=True)
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


def get_pca_exposure(conf, begin_date, end_date, PN=20, suffix='1602_2203') -> pd.DataFrame:
    """Exposure on Zscore(PCA) factors top 20"""

    def combine_pca_exposure(src, tgt, bd, ed, pn):
        """Run it ONCE to COMBINE principal exposures, when PCA is updated"""
        expo = pd.DataFrame()
        print(f'\nMerge PCA exposure PN={PN} {suffix}')
        for pn in tqdm(range(pn)):
            kw = f'pc{pn:03d}'
            df = pd.read_csv(src + f'{kw}.csv', index_col=0, parse_dates=True)
            df = df.loc[bd: ed]
            expo = pd.concat([expo, df.stack().rename(kw)], axis=1)
        #
        expo.to_pickle(tgt)
        return expo

    pkl_path = conf['data_path'] + f"exposure_pca{PN}_{suffix}.pkl"
    if os.path.exists(pkl_path):
        expo_pca: pd.DataFrame = pd.read_pickle(pkl_path)
    else:
        expo_pca = combine_pca_exposure(conf['dat_path_pca'], pkl_path, begin_date, end_date, PN)

    ind_date = expo_pca.index.get_level_values(0)
    expo_pca = expo_pca.groupby(ind_date).apply(lambda s: (s - s.mean()) / s.std())
    return expo_pca


def get_accessible_stk(i: set, a: set, b: set) -> Tuple[list, list, dict]:
    i_a = i.difference(a)
    i_b = i.difference(b)
    res = {'#i_b': len(i_b), '#i_a': len(i_a),
           'i_a': list(i_a), 'i_b': list(i_b)}
    ab = list(a.intersection(b))
    ib = list(i.intersection(b))
    return ab, ib, res


def portfolio_optimize(all_args, telling=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tradedates, expo_pca, expo_style, expo_indus, mkt_type, ind_cons, dat, args = all_args
    N, FL, FH, HL, HH, PL, PH, D, K, wei_tole, opt_verbose, use_barra = args
    holding_weight = pd.DataFrame()
    df_lst_w: pd.DataFrame = pd.DataFrame()
    optimize_iter_info = pd.DataFrame()
    w_lst = None
    cur_td = 0

    for cur_td in range(len(tradedates)):

        td = tradedates.iloc[cur_td].strftime('%Y-%m-%d')
        # pool coverage
        stk_alpha = set(dat.loc[td].rank(ascending=False).sort_values().index[:N]) if (N < np.inf) else set(
            dat.loc[td].dropna().index)
        stk_index = set(ind_cons.loc[td].dropna().index)
        stk_beta = set(expo_style.loc[td].index) if use_barra else set(expo_pca.loc[td].index)
        ls_ab, ls_ib, sp_info = get_accessible_stk(i=stk_index, a=stk_alpha, b=stk_beta)
        ls_clear = list(set(df_lst_w.index).difference(ls_ab))  # 上期持有 组合资产未覆盖
        if telling:
            print(f"\n\t{mkt_type} - alpha({len(stk_alpha)}) = {sp_info['#i_a']} [{','.join(sp_info['i_a'])}]")
            print(f"\t{mkt_type} - beta({len(stk_beta)}) = {sp_info['#i_b']} [{','.join(sp_info['i_b'])}]")
            print(f'\talpha exposed ({len(ls_ab)}/{len(stk_alpha)})')
            print(f'\t{mkt_type.lower()} exposed ({len(ls_ib)}/{len(stk_index)})')
            print(f'\tformer holdings not exposed ({len(ls_clear)}/{len(df_lst_w)}) [{",".join(ls_clear)}]')

        a = dat.loc[td, ls_ab]  # alpha
        xf: pd.DataFrame = expo_style.loc[td].loc[ls_ab] if use_barra else None  # style beta
        h: pd.DataFrame = expo_indus.loc[td].loc[ls_ab].dropna(axis=1) if use_barra else None  # industry beta
        p: pd.DataFrame = expo_pca.loc[td].loc[ls_ab] if (not use_barra) else None  # pca beta
        wb = ind_cons.loc[td, ls_ib]
        wb /= wb.sum()  # part of index-constituent are not exposed to beta factors; (not) treat them as zero-exposure.
        f_del = expo_style.loc[td].loc[ls_ib].T @ wb if use_barra else None
        h_del = expo_indus.loc[td].loc[ls_ib].dropna(axis=1).T @ wb if use_barra else None
        p_del = expo_pca.loc[td].loc[ls_ib].T @ wb if (not use_barra) else None
        fl = np.ones([xf.shape[-1], 1]) * FL + f_del.values.reshape(-1, 1) if use_barra else None
        fh = np.ones([xf.shape[-1], 1]) * FH + f_del.values.reshape(-1, 1) if use_barra else None
        hl = np.ones([h.shape[-1], 1]) * HL + h_del.values.reshape(-1, 1) if use_barra else None
        hh = np.ones([h.shape[-1], 1]) * HH + h_del.values.reshape(-1, 1) if use_barra else None
        pl = np.ones([p.shape[-1], 1]) * PL + p_del.values.reshape(-1, 1) if (not use_barra) else None
        ph = np.ones([p.shape[-1], 1]) * PH + p_del.values.reshape(-1, 1) if (not use_barra) else None
        K0 = wb.max() * 100
        K1 = math.ceil(K0)
        k = np.ones([len(ls_ab), 1]) * (max(K, K1) / 100)
        d_del = df_lst_w.loc[ls_clear].abs().sum().values[0] if len(ls_clear) > 0 else 0
        d = D - d_del

        # Solve optimize problem
        w = cp.Variable((len(ls_ab), 1), nonneg=True)
        objective = cp.Maximize(a.values.reshape(1, -1) @ w)
        # objective = cp.Minimize(a.values.reshape(1, -1) @ w)
        constraints = [w <= k, cp.sum(w) == 1]
        constraints.extend([
                               fl - xf.values.T @ w <= 0,
                               xf.values.T @ w - fh <= 0,
                               hl - h.values.T @ w <= 0,
                               h.values.T @ w <= hh] if use_barra else [
            pl - p.values.T @ w <= 0,
            p.values.T @ w - ph <= 0])
        if len(df_lst_w) > 0:
            w_lst = df_lst_w.reindex_like(pd.DataFrame(index=ls_ab, columns=df_lst_w.columns)).fillna(0).values
            constraints.append(cp.norm(w - w_lst, 1) <= d)
        else:
            w_lst = np.zeros([len(ls_ab), 1]) if w_lst is None else w_lst  # former holding
        prob = cp.Problem(objective, constraints)
        result = prob.solve(verbose=opt_verbose, solver='ECOS', max_iters=1000)
        if prob.status == 'optimal_inaccurate':
            result = prob.solve(verbose=opt_verbose, solver='ECOS', max_iters=10000)
        # result = prob.solve(verbose=True, solver='ECOS', abstol=1e-8, max_iters=100)

        if prob.status == 'optimal':
            w1 = w.value.copy()
            w1[w1 < wei_tole] = 0
            w1 = w1 / w1.sum()
            turnover = np.abs(w_lst - w1).sum() + d_del
            hdn = (w1 > 0).sum()

            df_lst_w = pd.DataFrame(w1, index=ls_ab, columns=[td])
            df_lst_w = df_lst_w.replace(0, np.nan).dropna()
        else:
            raise Exception(f'{prob.status} problem')
            # turnover = 0
            # print(f'.{prob.status} problem, portfolio ingredient unchanged')
            # if len(lst_w) > 0:
            #     lst_w.columns = [td]
        holding_weight = pd.concat([holding_weight, df_lst_w.T])

        # update optimize iteration information
        progressbar(cur_td + 1, len(tradedates), msg=f' {td} turnover={turnover:.3f} #stk={hdn}')
        iter_info = {'#alpha^beta': len(ls_ab), '#index^beta': len(ls_ib), '#index': len(stk_index),
                     'turnover': turnover, 'holding': hdn, 'use_barra': use_barra, 'wei_tole': wei_tole,
                     'status': prob.status, 'opt0': result, 'opt1': (a @ w1)[0],
                     'N': N, 'FL': FL, 'FH': FH, 'HL': HL, 'HH': HH, 'PL': PL, 'PH': PH, 'D': D,
                     'K': K, 'K0': K0, 'K1': K1}
        iter_info = iter_info | (f_del.to_dict() | h_del.to_dict() if use_barra else p_del.to_dict())
        iter_info = iter_info | {'index-alpha': sp_info['#i_a'], 'index-beta': sp_info['#i_b'],
                                 'stk_i_a': ', '.join(sp_info['i_a']), 'stk_i_b': ', '.join(sp_info['i_b'])}
        optimize_iter_info[td] = pd.Series(iter_info)

    return holding_weight, optimize_iter_info


def tf_portfolio_weight(portfolio_weight, tab_path, gra_path, ishow=False):
    """Table & Graph: Holding weight, holding number"""
    portfolio_weight.sum(axis=1)
    portfolio_weight.to_csv(tab_path)
    (portfolio_weight > 0).sum(axis=1).plot()
    plt.savefig(gra_path)  # dat_path + g_file.format(suffix))
    if ishow:
        plt.show()
    else:
        plt.close()


def tf_historical_result(close_adj, tradedates, begin_date, end_date, portfolio_weight, ind_cons, mkt_type, gra_path,
                         tab_path):
    """Table & Graph: wealth curve comparison"""
    df = pd.DataFrame()

    rtn_w2w = close_adj.pct_change(5).loc[tradedates.loc[begin_date: end_date].index]
    rtn_portfolio = (portfolio_weight * rtn_w2w.reindex_like(portfolio_weight)).sum(axis=1)
    df['portfolio'] = rtn_portfolio

    ind_cons_w = ind_cons.loc[tradedates.loc[begin_date: end_date].index]
    rtn_ind = (ind_cons_w * rtn_w2w.reindex_like(ind_cons_w)).sum(axis=1)
    df[mkt_type] = rtn_ind
    df['Excess'] = df['portfolio'] - df[mkt_type]

    tmp = df.cumsum().add(1)
    tmp.plot()
    plt.savefig(gra_path)
    plt.close()

    tmp.to_excel(tab_path)
    del tmp

