"""
(created by swmao on April 22nd)

"""
import sys
from multiprocessing import Pool, RLock, freeze_support
import yaml

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.bata_etf import *
from BarraPCA.opt_res_ana import OptRes

OPTIMIZE_TARGET = '/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/BarraPCA/optimize_target_v2.xlsx'
PROCESS_NUM = 4
mkdir_force = False
TELLING = False


# %%
def main():
    # %%
    t0 = time.time()

    # Configs:
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    optimize_target: pd.DataFrame = pd.read_excel(OPTIMIZE_TARGET, index_col=0, dtype=object).loc[1:1]
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
        cnt = 0
        for ir in optimize_target.iterrows():
            ir1 = ir[1]
            args = (conf, ir1, mkdir_force, cnt % PROCESS_NUM)
            optimize(args)
            cnt += 1

    # Exit:
    print(f'\nTime used: {second2clock(round(time.time() - t0))}')


def optimize(args):
    # %%
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
    G = float(ir1['G'])
    S = float(ir1['S'])
    wei_tole = float(ir1['wei_tole'])
    alpha_name = ir1['alpha_name']
    beta_kind = ir1['beta_kind']
    beta_args = eval(ir1['beta_args'])
    beta_suffix = ir1['beta_suffix']
    suffix = f"{beta_suffix}(B={ir1['B']},E={ir1['E']},D={ir1['D']},H0={ir1['H0']}" + \
             ('' if np.isnan(H1) else f",H1={ir1['H1']}") + \
             (f"G={ir1['G']}" if G else '') + \
             (f"S={ir1['S']}" if S < np.inf else '') + \
             ')'  # suffix for all result file
    script_info = {
        'opt_verbose': opt_verbose, 'begin_date': begin_date, 'end_date': end_date, 'mkt_type': mkt_type,
        'N': N, 'H0': H0, 'H1': H1, 'B': B, 'E': E, 'D': D, 'G': G, 'S': S, 'wei_tole': wei_tole,
        'alpha_name': alpha_name, 'beta_kind': beta_kind, 'alpha_5d_rank_ic': 'NA', 'suffix': suffix,
    }

    # %% Load Data
    beta_expo, beta_cnstr = get_beta_expo_cnstr(beta_kind, conf, begin_date, end_date, H0, H1, beta_args)
    save_path, alpha = get_alpha_dat(alpha_name, mkt_type, conf, begin_date, end_date)
    # alpha_5d_rank_ic = check_ic_5d(conf['closeAdj'], dat, begin_date, end_date, lag=5)  # TODO: cal ic once
    # script_info['alpha_5d_rank_ic'] = str(alpha_5d_rank_ic)
    save_path_sub = f'{save_path}{suffix}/'
    io_make_sub_dir(save_path_sub, force=dir_force)
    ind_cons = get_index_constitution(conf['idx_constituent'].format(mkt_type), begin_date, end_date)
    tradedates = get_tradedates(conf, begin_date, end_date, kind='tdays_w')

    path_sigma = conf['risk_matrix'] if (G > 0 or S < np.inf) else None
    desc = alpha_name + '/' + suffix
    all_args = tradedates, beta_expo, beta_cnstr, ind_cons, alpha, (
        mkt_type, N, D, B, E, G, S, path_sigma, wei_tole, opt_verbose, desc, pos)
    telling = TELLING
    # %% Optimize
    portfolio_weight, optimize_iter_info = portfolio_optimize(all_args, telling=telling)

    # Save:
    with open(save_path_sub + 'config_optimize.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(script_info, f)
    optimize_iter_info.T.to_excel(save_path_sub + f'opt_info{suffix}.xlsx')
    portfolio_weight.to_csv(save_path_sub + 'portfolio_weight_{}.csv'.format(suffix))
    # Graphs & Tables:
    opt_res = OptRes(ir1, conf)
    opt_res.tf_historical_result()
    opt_res.tf_portfolio_weight()


def portfolio_optimize(all_args, telling=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # %%
    tradedates, beta_expo, beta_cnstr, ind_cons, alpha, args = all_args
    mkt_type, N, D, B, E, G, S, path_sigma, wei_tole, opt_verbose, desc, pos = args

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

        # Specific Risk
        sigma: pd.DataFrame = get_risk_matrix(path_sigma, td, max_backward=5, notify=False)

        # Asset pool accessible
        stk_alpha = get_stk_alpha(alpha.loc[td])
        stk_index = set(ind_cons.loc[td].dropna().index)
        stk_beta = set(beta_expo.loc[td].index)
        stk_sigma = set(sigma.index) if path_sigma else None
        ls_pool, ls_base, sp_info = get_accessible_stk(i=stk_index, a=stk_alpha, b=stk_beta, s=stk_sigma)
        ls_clear = list(set(df_lst_w.index).difference(ls_pool))  # 上期持有 组合资产未覆盖
        if telling:
            print(f"\n\t{mkt_type} - alpha({len(stk_alpha)}) = {sp_info['#i_a']} [{','.join(sp_info['i_a'])}]")
            print(f"\t{mkt_type} - beta({len(stk_beta)}) ^ sigma({len(stk_sigma)})"
                  f" = {sp_info['#i_b(s)']} [{','.join(sp_info['i_b(s)'])}]" if path_sigma else
                  f"\t{mkt_type} - beta({len(stk_beta)}) = {sp_info['#i_b(s)']} [{','.join(sp_info['i_b(s)'])}]")
            print(f'\talpha exposed ({len(ls_pool)}/{len(stk_alpha)})')
            print(f'\t{mkt_type.lower()} exposed ({len(ls_base)}/{len(stk_index)})')
            print(f'\tformer holdings not exposed ({len(ls_clear)}/{len(df_lst_w)}) [{",".join(ls_clear)}]')

        a = alpha.loc[td, ls_pool].values.reshape(1, -1)  # alpha
        wb = ind_cons.loc[td, ls_base]
        wb /= wb.sum()  # part of index-constituent are not exposed to beta factors; (not) treat them as zero-exposure.
        xf = beta_expo.loc[td].loc[ls_pool].dropna(axis=1)
        f_del = beta_expo.dropna(axis=1).loc[td].loc[ls_base].T @ wb  # - f_del_overflow
        fl = (f_del + beta_cnstr.loc[f_del.index, 'L']).dropna()
        fh = (f_del + beta_cnstr.loc[f_del.index, 'H']).dropna()

        D_offset = df_lst_w.loc[ls_clear].abs().sum().values[0] if len(ls_clear) > 0 else 0
        Sigma = sigma.loc[ls_pool, ls_pool]

        # %% Constraints
        wN = len(ls_pool)
        w = cp.Variable((wN, 1), nonneg=True)
        opt_cnstr = OptCnstr()

        # (1) sum 1
        opt_cnstr.sum_bound(w, e=np.ones([1, wN]), down=None, up=1)

        # (2) cons component percentage
        opt_cnstr.sum_bound(w, e=(1 - pd.Series(wb, index=ls_pool).isna()).values.reshape(1, -1), down=B, up=None)

        # (3) cons weight deviation
        wb_ls_pool = pd.Series(wb, index=ls_pool).fillna(0)  # cons w, index broadcast as pool
        wbp = wb_ls_pool.values.reshape(-1, 1)
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
        if path_sigma:
            risk = cp.quad_form(w - wbp, Sigma)
            if S < np.inf:  # Attention: very slow if (G > 0) & (S < inf)
                opt_cnstr.add_constraints(risk <= S)
            objective = cp.Maximize(a @ w - G * risk) if G > 0 else cp.Maximize(a @ w)
        else:
            objective = cp.Maximize(a @ w)

        # %% Solve
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
            df_w = pd.DataFrame(w1, index=ls_pool, columns=[td])
            turnover = np.abs(w_lst - df_w.values).sum() + D_offset
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
        iter_info = {
            '#alpha^beta': len(ls_pool), '#index^beta': len(ls_base), '#index': len(stk_index),
            'turnover': turnover, 'holding': hdn,
            'status': prob.status, 'opt0': result, 'opt1': (a @ w1)[0],  # TODO
        }
        iter_info = iter_info | {'# cons w/o alpha': sp_info['#i_a'],
                                 '# cons w/o beta(sigma)': sp_info['#i_b(s)'],
                                 'cons w/o alpha': ', '.join(sp_info['i_a']),
                                 'cons w/o beta(sigma)': ', '.join(sp_info['i_b(s)'])}
        iter_info = iter_info | f_del.to_dict()
        optimize_iter_info[td] = pd.Series(iter_info)
        # progressbar(cur_td + 1, len(tradedates), msg=f' {td} turnover={turnover:.3f} #stk={hdn}', stt=start_time)

    return holding_weight, optimize_iter_info


if __name__ == '__main__':
    main()
