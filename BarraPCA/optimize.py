"""
(created by swmao on April 22nd)

"""
import sys
from multiprocessing import Pool, RLock, freeze_support
import yaml

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.bata_etf import *

OPTIMIZE_TARGET = '/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/BarraPCA/optimize_target.xlsx'
PROCESS_NUM = 4
mkdir_force = False


def optimize(args):
    conf: dict = args[0]
    ir1: pd.Series = args[1]
    dir_force: bool = args[2]
    pos: int = args[3]

    mkt_type = ir1['mkt_type']
    begin_date = ir1['begin_date']
    end_date = ir1['end_date']
    N = float(ir1['N'])
    opt_verbose = (ir1['opt_verbose'] == 'TRUE')
    expoH = float(ir1['expoH'])
    expoL = float(ir1['expoL'])
    K = float(ir1['K'])
    D = float(ir1['D'])
    wei_tole = float(ir1['wei_tole'])
    alpha_name = ir1['alpha_name']
    beta_kind = ir1['beta_kind']
    beta_args = eval(ir1['beta_args'])
    beta_suffix = ir1['beta_suffix']

    suffix = f'{beta_suffix}(H={expoH},L={expoL},K={K})'  # suffix for all result file
    script_info = {
        'opt_verbose': opt_verbose, 'begin_date': begin_date, 'end_date': end_date, 'mkt_type': mkt_type,
        'N': N, 'expoH': expoH, 'expoL': expoL, 'D': D, 'K': K, 'wei_tole': wei_tole,
        'alpha_name': alpha_name, 'beta_kind': beta_kind, 'alpha_5d_rank_ic': 'NA', 'suffix': suffix,
    }

    # Load Data
    beta_expo, beta_cnstr = get_beta_expo_cnstr(beta_kind, conf, begin_date, end_date, expoL, expoH, beta_args)
    save_path, dat = get_alpha_dat(alpha_name, mkt_type, conf, begin_date, end_date)
    alpha_5d_rank_ic = check_ic_5d(conf['closeAdj'], dat, begin_date, end_date, lag=5)  # TODO: cal ic once
    script_info['alpha_5d_rank_ic'] = str(alpha_5d_rank_ic)
    save_path_sub = f'{save_path}{suffix}/'
    io_make_sub_dir(save_path_sub, force=dir_force)
    ind_cons = get_index_constitution(conf['idx_constituent'].format(mkt_type), begin_date, end_date)
    tradedates = get_tradedates(conf, begin_date, end_date, kind='tdays_w')

    # Optimize
    desc = alpha_name + '/' + suffix
    args = (mkt_type, N, D, K, wei_tole, opt_verbose, desc, pos)
    all_args = tradedates, beta_expo, beta_cnstr, ind_cons, dat, args
    telling = False  # True
    portfolio_weight, optimize_iter_info = portfolio_optimize(all_args, telling)

    # Save:
    with open(save_path_sub + 'config_optimize.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(script_info, f)

    optimize_iter_info.T.to_excel(save_path_sub + f'opt_info{suffix}.xlsx')

    tab_path = save_path_sub + 'portfolio_weight_{}.csv'.format(suffix)
    gra_path = save_path_sub + 'figure_portfolio_size_{}.png'.format(suffix)
    tf_portfolio_weight(portfolio_weight, tab_path, gra_path, ishow=False)

    close_adj = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True)

    tab_path = save_path_sub + 'table_result_wealth_{}.xlsx'.format(suffix)
    gra_path = save_path_sub + 'figure_result_wealth_{}.png'.format(suffix)
    tf_historical_result(close_adj, tradedates, begin_date, end_date,
                         portfolio_weight, ind_cons, mkt_type, gra_path, tab_path)


def main():
    t0 = time.time()
    # Configs:
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    optimize_target: pd.DataFrame = pd.read_excel(OPTIMIZE_TARGET, index_col=0, dtype=object).loc[1:1]
    print(optimize_target)
    # Run optimize:
    print(f'father process {os.getpid()}')
    freeze_support()
    p = Pool(PROCESS_NUM, initializer=tqdm.set_lock, initargs=(RLock(),))
    cnt = 0
    # ir1 = optimize_target.iloc[0]
    for ir in optimize_target.iterrows():
        ir1 = ir[1]
        p.apply_async(optimize, args=[(conf, ir1, mkdir_force, cnt % PROCESS_NUM)])
        cnt += 1
    p.close()
    p.join()
    # Exit:
    print(f'\nTime used: {second2clock(round(time.time() - t0))}')


if __name__ == '__main__':
    main()
