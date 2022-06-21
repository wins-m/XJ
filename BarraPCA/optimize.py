"""
(created by swmao on April 22nd)

"""
import os
import sys
import time
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple
import cvxpy as cp
from multiprocessing import Pool, RLock, freeze_support

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.bata_etf import second2clock, info2suffix, get_tradedates, get_beta_expo_cnstr, get_index_constitution, \
    get_factor_covariance, get_specific_risk, link_alpha_dat, get_save_path, io_make_sub_dir, get_accessible_stk, \
    OptCnstr
from BarraPCA.opt_res_ana import OptRes

OPTIMIZE_TARGET = '/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/BarraPCA/optimize_target_v2.xlsx'
PROCESS_NUM = 4
mkdir_force = True
TELLING = False


# %%
def main():
    # %%
    t0 = time.time()

    # Configs:
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    # Optimize Target:
    optimize_target = pd.read_excel(OPTIMIZE_TARGET, index_col=0, dtype=object).loc[1:1]
    print(optimize_target)

    # Run optimize:
    ir1 = optimize_target.iloc[0]
    args = (conf, ir1, mkdir_force, 0)
    # %%
    if PROCESS_NUM > 1:
        print(f'father process {os.getpid()}')
        freeze_support()
        p = Pool(PROCESS_NUM, initializer=tqdm.set_lock, initargs=(RLock(),))
        cnt = 0
        for ir in optimize_target.iterrows():
            ir1 = ir[1]
            p.apply_async(optimize, args=[(conf, ir1, mkdir_force, cnt % PROCESS_NUM)])
            cnt += 1
        p.close()
        p.join()
    else:
        for ir in optimize_target.iterrows():
            ir1 = ir[1]
            args = (conf, ir1, mkdir_force, 0)
            optimize(args)
    # %% Exit:
    print(f'\nTime used: {second2clock(round(time.time() - t0))}')


def optimize(args):
    """"""
    # %% Decode setting and parameters
    conf: dict = args[0]
    ir1: pd.Series = args[1]
    dir_force: bool = args[2]
    pos: int = args[3]

    mkt_type = ir1['mkt_type']
    begin_date = ir1['begin_date']
    end_date = ir1['end_date']
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
    # %% Load DataFrames
    tradedates = get_tradedates(conf, begin_date, end_date, kind='tdays_w')
    # tradedates = get_tradedates(conf, begin_date, end_date, kind='tdays_d')
    beta_expo, beta_cnstr = get_beta_expo_cnstr(beta_kind, conf, begin_date, end_date, H0, H1, beta_args)
    ind_cons = get_index_constitution(conf['idx_constituent'].format(mkt_type), begin_date, end_date)
    fct_cov = get_factor_covariance(path_F=conf['factor_covariance'], bd=begin_date, ed=end_date, fw=1)
    stk_rsk = get_specific_risk(path_D=conf['specific_risk'], bd=begin_date, ed=end_date, fw=1)
    save_path = get_save_path(conf['factorsres_path'], mkt_type, alpha_name)
    alpha: pd.DataFrame = link_alpha_dat(alpha_name, conf['factorscsv_path'], begin_date, end_date, save_path)

    # alpha_5d_rank_ic = check_ic_5d(conf['closeAdj'], dat, begin_date, end_date, lag=5)  # cal ic
    # script_info['alpha_5d_rank_ic'] = str(alpha_5d_rank_ic)
    save_path_sub = f'{save_path}{suffix}/'
    io_make_sub_dir(save_path_sub, force=dir_force)

    desc = suffix
    all_args = tradedates, beta_expo, beta_cnstr, ind_cons, fct_cov, stk_rsk, alpha, (
        mkt_type, N, D, B, E, G, S, wei_tole, opt_verbose, desc, pos)
    telling = TELLING
    # %% Optimize:
    portfolio_weight, optimize_iter_info = portfolio_optimize(all_args, telling=telling)

    # %% Save Historical Optimize Results:
    with open(save_path_sub + 'config_optimize.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(script_info, f)
    optimize_iter_info.T.to_excel(save_path_sub + f'opt_info_{suffix}.xlsx')
    portfolio_weight.to_csv(save_path_sub + 'portfolio_weight_{}.csv'.format(suffix))
    # Graphs & Tables:
    opt_res = OptRes(ir1, conf['closeAdj'], conf['idx_constituent'], conf['factorsres_path'])
    opt_res.tf_historical_result()
    opt_res.tf_portfolio_weight()
    opt_res.tf_turnover()
    opt_res.tf_optimize_time()
    opt_res.tf_risk_and_result()


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

    # %%
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
    td = '2021-12-31'
    loop_bar = tqdm(range(len(tradedates)), ncols=99, desc=desc, delay=0.01, position=pos, ascii=False)
    # %%
    for cur_td in loop_bar:
        td = tradedates.iloc[cur_td].strftime('%Y-%m-%d')
        # %%
        use_sigma = (G > 0) or (S < np.inf)  # Specific Risk
        # sigma: pd.DataFrame = get_risk_matrix(path_sigma, td, max_backward=5, notify=False)

        # Asset pool accessible
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

        # alpha.std(axis=1)
        # cross_section_sd = alpha.loc['2013-01-01': '2022-03-31'].std(axis=1)
        # cross_section_sd.plot()
        # plt.title('cross-section standard deviation of FRtn5d(0.0,3.0)')
        # plt.tight_layout()
        # plt.show()
        # cross_section_sd.describe()

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

        # %%
        # path = '/mnt/c/Users/Winst/desktop/'
        # a.to_pickle(path + "alpha.pkl")
        # mat_F.to_pickle(path + "factor_covariance.pkl")
        # srs_D.to_pickle(path + "specific_risk.pkl")
        # xf.to_pickle(path + "factor_exposure.pkl")

        def wtf(a, mat_F, srs_D, xf, df_lst_w, w_lst, G, ishow=False):  # TODO: out
            a = a.values.reshape(1, -1)
            mat_F = np.matrix(mat_F)
            srs_D = np.matrix(srs_D ** 2)
            f_del = beta_expo.dropna(axis=1).loc[td].loc[ls_base].T @ wb  # - f_del_overflow
            fl = (f_del + beta_cnstr.loc[f_del.index, 'L']).dropna()
            fh = (f_del + beta_cnstr.loc[f_del.index, 'H']).dropna()

            D_offset = df_lst_w.loc[ls_clear].abs().sum().values[0] if len(ls_clear) > 0 else 0

            # Constraints
            wN = len(ls_pool)
            w = cp.Variable((wN, 1), nonneg=True)
            opt_cnstr = OptCnstr()

            # (1) sum 1
            opt_cnstr.sum_bound(w, e=np.ones([1, wN]), down=None, up=1)

            # (2) cons component percentage
            opt_cnstr.sum_bound(w, e=(1 - pd.Series(wb, index=ls_pool).isna()).values.reshape(1, -1), down=B, up=None)

            # (3) cons weight deviation
            offset = wb_ls_pool.apply(lambda _: max(E, _ / 2))  # max(E, 0.5w) as offset
            down = (wb_ls_pool - offset).values.reshape(-1, 1)
            up = (wb_ls_pool + offset).values.reshape(-1, 1)
            opt_cnstr.uni_bound(w, down=down, up=up)
            del offset, down, up

            # (4)(5) beta exposure
            opt_cnstr.sum_bound(w, e=xf[fl.index].values.T, down=fl.values.reshape(-1, 1), up=fh.values.reshape(-1, 1))

            # (6) turnover constraint
            if len(df_lst_w) > 0:  # not first optimization
                w_lst = df_lst_w.reindex_like(pd.DataFrame(index=ls_pool, columns=df_lst_w.columns)).fillna(0)
                w_lst = w_lst.values
                d = D - D_offset
                opt_cnstr.norm_bound(w, w0=w_lst, d=d, L=1)
            else:  # first iteration, holding 0
                w_lst = np.zeros([len(ls_pool), 1]) if w_lst is None else w_lst  # former holding

            # (7) specific risk
            # if use_sigma:
            wbp = wb_ls_pool.values.reshape(-1, 1)
            x = w - wbp
            risk = cp.quad_form(xf.values.T @ x, mat_F) + srs_D @ (x ** 2)
            if S < np.inf:
                opt_cnstr.add_constraints(risk <= S)
            # G = 20000
            objective = cp.Maximize(a @ w - G * risk) if G > 0 else cp.Maximize(a @ w)
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
                w1 = w.value.copy()
                w1[w1 < wei_tole] = 0
                w1 /= w1.sum()
                df_w = pd.DataFrame(w1, index=ls_pool, columns=[td])
                turnover = np.abs(w_lst - df_w.values).sum() + D_offset
                hdn = (df_w.values > 0).sum()
                df_lst_w = df_w.replace(0, np.nan).dropna()
                if ishow:  # Graph:
                    import matplotlib.pyplot as plt
                    plt.rc("figure", figsize=(9, 5))
                    plt.rc("font", size=12)
                    plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
                    plt.rcParams['axes.xmargin'] = 0
                    plt.rcParams['axes.ymargin'] = 0
                    plt.rc("savefig", dpi=90)
                    plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"
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
                'risk': risk.value, 'opt0': result, 'opt1': (a @ w1)[0],
                'solver': prob.solver_stats.solver_name, 'status': prob.status, 'stime': prob.solver_stats.solve_time,
            }

            #
            return iter_info, f_del, df_lst_w, w_lst

        # %%
        iter_info, f_del, df_lst_w, w_lst = wtf(a, mat_F, srs_D, xf, df_lst_w, w_lst, G=G, ishow=False)

        # %% Update optimize iteration information
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


if __name__ == '__main__':
    main()
