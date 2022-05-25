"""
(created by swmao on April 22nd)

"""
import os

import numpy as np
import pandas as pd
import yaml
import sys

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.bata_etf import *


# %%
def main():
    # %% Configs:
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    mkt_type = 'CSI500'
    begin_date = '2016-02-01'
    end_date = '2022-03-31'
    N = np.inf  # 2000  stk with largest N alpha
    opt_verbose = False

    # expoH = 0.00
    expoH = 0.10
    # expoH = 0.05
    # expoH = 0.20
    expoL = -expoH

    # K = .2
    # K = .5
    K = 5
    D = 2
    wei_tole = 1e-5

    # alpha_name = 'FRtn5D(0.0,3.0)'
    # alpha_name = 'FRtn5D(0.0,1.0)'
    alpha_name = 'FRtn5D(0.0,0.0)'
    # alpha_name = 'APM(zscore)'
    # alpha_name = 'APM(reverse)'
    # alpha_name = 'APM(uniform)'

    beta_kind = 'Barra'
    beta_args = (['size', 'beta', 'momentum',
                  # 'residual_volatility', 'non_linear_size',
                  # 'book_to_price_ratio', 'liquidity',
                  # 'earnings_yield', 'growth', 'leverage'
                  ],)
    beta_suffix = f'barra{len(beta_args[0])}'
    # beta_kind = 'PCA'
    # beta_args = (20, )
    # beta_suffix = f'pca{beta_args[0]}'

    suffix = f'{beta_suffix}(H={expoH},L={expoL},K={K})'  # suffix for all result file
    script_info = {
        'opt_verbose': opt_verbose, 'begin_date': begin_date, 'end_date': end_date, 'mkt_type': mkt_type,
        'N': N, 'expoH': expoH, 'expoL': expoL, 'D': D, 'K': K, 'wei_tole': wei_tole,
        'alpha_name': alpha_name, 'beta_kind': beta_kind, 'alpha_5d_rank_ic': 'NA', 'suffix': suffix,
    }

    # %%
    beta_expo, beta_cnstr = get_beta_expo_cnstr(beta_kind, conf, begin_date, end_date, expoL, expoH, beta_args)
    save_path, dat = get_alpha_dat(alpha_name, mkt_type, conf, begin_date, end_date)
    alpha_5d_rank_ic = check_ic_5d(conf['closeAdj'], dat, begin_date, end_date, lag=5)
    script_info['alpha_5d_rank_ic'] = str(alpha_5d_rank_ic)
    save_path_sub = f'{save_path}{suffix}/'
    io_make_sub_dir(save_path_sub, force=False)
    ind_cons = get_index_constitution(conf['idx_constituent'].format(mkt_type), begin_date, end_date)
    tradedates = get_tradedates(conf, begin_date, end_date, kind='tdays_w')

    # %% Optimize
    with open(save_path_sub + 'config_optimize.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(script_info, f)

    args = (mkt_type, N, D, K, wei_tole, opt_verbose)
    all_args = tradedates, beta_expo, beta_cnstr, ind_cons, dat, args
    telling = False  # True

    portfolio_weight, optimize_iter_info = portfolio_optimize(all_args, telling)

    # %% Save:
    optimize_iter_info.T.to_excel(save_path_sub + f'opt_info{suffix}.xlsx')

    tab_path = save_path_sub + 'portfolio_weight_{}.csv'.format(suffix)
    gra_path = save_path_sub + 'figure_portfolio_size_{}.png'.format(suffix)
    tf_portfolio_weight(portfolio_weight, tab_path, gra_path, ishow=False)

    close_adj = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True)

    tab_path = save_path_sub + 'table_result_wealth_{}.xlsx'.format(suffix)
    gra_path = save_path_sub + 'figure_result_wealth_{}.png'.format(suffix)
    tf_historical_result(close_adj, tradedates, begin_date, end_date,
                         portfolio_weight, ind_cons, mkt_type, gra_path, tab_path)


if __name__ == '__main__':
    main()
