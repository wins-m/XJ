"""
(created by swmao on April 22nd)

"""
import yaml
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.bata_etf import *


def main():
    # %%
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    opt_verbose = False
    use_barra = False
    begin_date = '2016-02-01'
    end_date = '2022-03-31'
    mkt_type = 'CSI500'
    PN = 20
    pca_suffix = '1602_2203'
    N = np.inf  # 2000
    expoL = FL = HL = PL = -.2
    expoH = FH = HH = PH = .2
    D = 2
    K = .5
    wei_tole = 1e-5

    # %% Alpha

    # # Use Fake Alpha
    # pred_days = 5
    # wn_m = 0.
    # wn_s = 3.
    # alpha_name = f'FRtn{pred_days}D({wn_m},{wn_s})'
    # save_suffix = f'OptResWeekly[{mkt_type}]{alpha_name}'
    # save_path = f"{conf['factorsres_path']}{save_suffix}/"
    # os.makedirs(save_path, exist_ok=True)  
    # alpha_save_name = save_path + f'factor_{alpha_name}.csv'
    # if os.path.exists(alpha_save_name):
    #     dat = pd.read_csv(alpha_save_name, index_col=0, parse_dates=True)
    # else:
    #     dat = get_fval_alpha(conf, begin_date, end_date, pred_days, wn_m, wn_s)
    #     dat.to_csv(alpha_save_name) 

    # Use APM
    apm_kind = 'reverse'
    alpha_name, dat = get_fval_apm(conf, begin_date, end_date, kind=apm_kind)
    save_suffix = f'OptResWeekly[{mkt_type}]{alpha_name}'
    save_path = f"{conf['factorsres_path']}{save_suffix}/"
    os.makedirs(save_path, exist_ok=True)
    alpha_save_name = save_path + f'factor_{alpha_name}.csv'
    dat.to_csv(alpha_save_name)

    # %%
    suffix = f'(H={expoH},L={expoL},K={K})'
    suffix = ('' if use_barra else 'pca') + suffix
    save_path1 = f'{save_path}{suffix}/'
    os.makedirs(save_path1, exist_ok=False)
    print(f'Save in: {save_path1}')

    print(f"Factor name: {alpha_name}")
    alpha_5d_rank_ic = check_ic_5d(conf, dat, begin_date, end_date, lag=5)
    expo_style, expo_indus = get_style_indus_exposure(conf, begin_date, end_date)
    ind_cons = get_index_constitution(conf, mkt_type, begin_date, end_date)
    tradedates = get_tradedates(conf, begin_date, end_date, kind='tdays_w')
    expo_pca = get_pca_exposure(conf, begin_date, end_date, PN=PN, suffix=pca_suffix)

    # %% Optimize
    args = (N, FL, FH, HL, HH, PL, PH, D, K, wei_tole, opt_verbose, use_barra)
    all_args = tradedates, expo_pca, expo_style, expo_indus, mkt_type, ind_cons, dat, args
    telling = False  # True

    portfolio_weight, optimize_iter_info = portfolio_optimize(all_args, telling)

    # %% Save:
    script_info = {
        'begin_date': begin_date, 'end_date': end_date, 'mkt_type': mkt_type,
        'PN': PN, 'suffix': suffix, 'N': N, 'FL': FL, 'FH': FH, 'HL': HL, 'HH': HH,
        'PL': PL, 'PH': PH, 'D': D, 'K': K, 'wei_tole': wei_tole, 'use_barra': use_barra,
        'alpha_name': alpha_name, 'alpha_5d_rank_ic': str(alpha_5d_rank_ic),
    }
    with open(save_path1 + 'config_optimize.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(script_info, f)

    optimize_iter_info.T.to_excel(save_path1 + f'opt_info{suffix}.xlsx')

    tab_path = save_path1 + 'portfolio_weight_{}.csv'.format(suffix)
    gra_path = save_path1 + 'figure_portfolio_size_{}.png'.format(suffix)
    tf_portfolio_weight(portfolio_weight, tab_path, gra_path, ishow=False)

    close_adj = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True)

    tab_path = save_path1 + 'table_result_wealth_{}.xlsx'.format(suffix)
    gra_path = save_path1 + 'figure_result_wealth_{}.png'.format(suffix)
    tf_historical_result(close_adj, tradedates, begin_date, end_date,
                         portfolio_weight, ind_cons, mkt_type, gra_path, tab_path)


if __name__ == '__main__':
    main()
