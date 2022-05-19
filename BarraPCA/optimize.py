"""
(created by swmao on April 22nd)

"""
import os
import pandas as pd
import numpy as np
import cvxpy as cp
import yaml
from tqdm import tqdm
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.factor_operator import cal_ic

# %matplotlib inline
import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import seaborn
seaborn.set_style("darkgrid")
# plt.rc("figure", figsize=(16, 10))
plt.rc("figure", figsize=(8, 5))
plt.rc("savefig", dpi=90)
# plt.rc("font", family="sans-serif")
# plt.rc("font", size=12)
plt.rc("font", size=10)
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"


def main():
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    dat_path = conf['dat_path_barra']
    opt_verbose = False
    #
    save_suffix = '中证500_等暴露_K=5_APM1'
    os.makedirs(dat_path + save_suffix, exist_ok=True)
    begin_date = '2016-02-01'
    end_date = '2022-03-31'
    mkt_type = 'CSI500'
    N = np.inf  # 2000
    FL = HL = PL = -.0
    FH = HH = PH = .0
    D = 2
    K = 5  # %
    wei_tole = 1e-5
    # FL = -.1
    # FH = .1
    # HL = -.1
    # HH = .1
    # PL = -.1
    # PH = .1


def get_fval_alpha(conf, begin_date, end_date):
    alpha_name = 'FRtn5D'
    close_adj = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True)
    pred_days = 5
    rtn_rt5c = close_adj.pct_change(pred_days).shift(-pred_days)  # 未来5日的收益
    alpha_val = rtn_rt5c.shift(-1)
    dat = alpha_val.loc[begin_date: end_date]
    return alpha_name, dat


def get_fval_apm(conf, begin_date, end_date):
    alpha_name = 'APM'
    alpha_val = pd.read_csv(conf['data_path'] + 'factor_apm.csv', index_col=0, parse_dates=True)
    alpha_val = alpha_val.apply(lambda s: (s - s.min()) / (s.max() - s.min()), axis=1)
    alpha_val = alpha_val.apply(lambda s: (s - s.mean()) / s.std(), axis=1)
    dat = alpha_val.loc[begin_date: end_date].shift(1)
    return alpha_name, dat


"""check IC-5days"""
#
close_adj = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True)
dat_ic = cal_ic(fv_l1=dat,
                ret=close_adj.pct_change().loc[begin_date: end_date],
                lag=1,
                ranked=True)
dat_ic.mean()

# %%
# Exposure 当天的因子暴露
expo = pd.DataFrame()
for _ in range(int(begin_date.split('-')[0]), int(end_date.split('-')[0]) + 1):
    expo = pd.concat([expo, pd.DataFrame(pd.read_hdf(conf['barra_panel'], key=f'y{_}'))])
expo: pd.DataFrame = expo.loc[begin_date: end_date]
#
cols_style = [c for c in expo.columns if 'rtn' not in c and 'ind' not in c and 'country' != c]
cols_indus = [c for c in expo.columns if 'ind' in c]
expo_style: pd.DataFrame = expo[cols_style].groupby(
    expo[cols_style].index.get_level_values(0)).apply(lambda s: (s - s.mean()) / s.std())
expo_indus: pd.DataFrame = expo[cols_indus]
# Baseline Portfolio 股指成分
ind_cons = pd.read_csv(conf['idx_constituent'].format(mkt_type), index_col=0, parse_dates=True).reindex_like(
    dat).fillna(0)
ind_cons = ind_cons / 100
ind_cons.tail()
# Trade days 交易日期
tdays_d = pd.read_csv(conf['tdays_w'], header=None, index_col=0, parse_dates=True).loc[begin_date: end_date]
tdays_d['tdays_d'] = tdays_d.index
tradedates = tdays_d.tdays_d
# Exposure on PCA factors top 20
expo_pca: pd.DataFrame = pd.read_pickle(conf['data_path'] + 'exposure_pca20_1602_2203.pkl')
expo_pca = expo_pca.groupby(expo_pca.index.get_level_values(0)).apply(lambda s: (s - s.mean()) / s.std())


def combine_pca_exposure(conf, begin_date: str, end_date: str, PN=20, suffix='1602_2203'):
    """Run it ONCE to COMBINE principal exposures, when PCA is updated"""
    expo_pca = pd.DataFrame()
    for pn in tqdm(range(PN)):
        kw = f'pc{pn:03d}'
        df = pd.read_csv(conf['dat_path_pca'] + f'{kw}.csv', index_col=0, parse_dates=True).loc[begin_date: end_date]
        expo_pca = pd.concat([expo_pca, df.stack().rename(kw)], axis=1)
    #
    file_name = f'exposure_pca{PN}_{suffix}.pkl'
    expo_pca.to_pickle(conf['data_path'] + file_name)
    return expo_pca


# %%
portfolio_weight = pd.DataFrame()
lst_w: pd.DataFrame = pd.DataFrame()
w_lst = None
optimize_iter_info = pd.DataFrame()
td0 = tradedates[0]
td = td0.strftime('%Y-%m-%d')
for td0 in tradedates.loc[td:]:
    td = td0.strftime('%Y-%m-%d')
    print(f'\n{td}')

    # 个股覆盖
    stk_exposed_pca = set(expo_pca.loc[td].dropna(axis=1).index)  # PCA 暴露覆盖个股
    stk_exposed_barra = set(expo.loc[td].index)  # Barra 暴露 覆盖个股
    stk_exposed = stk_exposed_barra.intersection(stk_exposed_pca)  # 存在暴露的个股
    stk_ind_cons = set(ind_cons.loc[td][ind_cons.loc[td] > 0].index)  # 指数成分股
    stk_alpha = set(dat.loc[td].dropna().index)  # Alpha 覆盖个股

    # alpha覆盖个股中选
    N_ingredient = len(stk_alpha) if (N == np.inf) else N
    stk_ingredient = set(dat.loc[td, list(stk_alpha)].rank(ascending=False).sort_values().index[:N_ingredient])  # 组合用股

    # 缺失不考虑
    ind_ingredient0 = set(ind_cons.loc[td].replace(0, np.nan).dropna().index)
    ind_diff_alpha = ind_ingredient0.difference(stk_ingredient)
    print(f'\t忽略不在alpha最高{N_ingredient}支的({len(ind_diff_alpha)}支){mkt_type}成分股: {",".join(list(ind_diff_alpha))}')
    ind_ingredient = ind_ingredient0.difference(ind_diff_alpha)
    if FH < np.inf:
        barra_ingredient = set(expo_style.loc[td].dropna().index)
        ind_diff_barra = ind_ingredient.difference(barra_ingredient)
        print(f'\t忽略缺少barra暴露的({len(ind_diff_barra)}支){mkt_type}成分股:  {",".join(list(ind_diff_barra))}')
        alpha_diff_barra = stk_ingredient.difference(set(barra_ingredient))
        print(f'\t忽略缺少barra暴露的({len(alpha_diff_barra)}支)备选个股')  # :  {",".join(list(alpha_diff_barra))}')
        ind_ingredient = ind_ingredient.difference(ind_diff_barra)
        stk_ingredient = stk_ingredient.difference(alpha_diff_barra)
    if PH < np.inf:
        pca_ingredient = set(expo_pca.loc[td].dropna().index)
        ind_diff_pca = ind_ingredient.difference(pca_ingredient)
        print(f'\t忽略缺少pca暴露的({len(ind_diff_pca)}支){mkt_type}成分股:  {",".join(list(ind_diff_pca))}')
        alpha_diff_pca = stk_ingredient.difference(set(pca_ingredient))
        print(f'\t忽略缺少pca暴露的({len(alpha_diff_pca)}支)备选个股')  # :  {",".join(list(alpha_diff_pca))}')
        ind_ingredient = ind_ingredient.difference(ind_diff_pca)
        stk_ingredient = stk_ingredient.difference(alpha_diff_pca)
    print(f'.从{len(stk_ingredient)}支股中组合增强{mkt_type}中的({len(ind_ingredient)}/{len(ind_ingredient0)})')

    # 存储日度选股前的情况
    ind_w_cvg = ind_cons.loc[td, ind_ingredient].sum() / ind_cons.loc[td, ind_ingredient0].sum()
    optimize_iter_info[td] = pd.Series({'pool_size': len(stk_ingredient), 'ind_w_cvg': ind_w_cvg,
                                        'ind_cvr': len(ind_ingredient), 'ind_size': len(ind_ingredient0),
                                        'N': N, 'FL': FL, 'FH': FH, 'HL': HL, 'HH': HH,
                                        'PL': PL, 'PH': PH, 'D': D, 'K': K, 'K1': K, 'wei_tole': wei_tole})

    # alpha中存在暴露的前N支个股中选
    a = dat.loc[td, list(stk_ingredient)]  # 最大化的 N_ingredient 个alpha

    # 上期持仓权重
    w_lst = np.zeros([len(stk_ingredient), 1]) if w_lst is None else w_lst

    # 公共因子暴露，缺失填充以 0
    xf: pd.DataFrame = expo_style.loc[td].reindex_like(
        pd.DataFrame(index=list(stk_ingredient), columns=cols_style)).dropna(axis=0)  # 组合资产 风格因子暴露
    h: pd.DataFrame = expo_indus.loc[td].reindex_like(
        pd.DataFrame(index=list(stk_ingredient), columns=cols_indus)).dropna(axis=1, how='all').dropna(
        axis=0)  # 组合资产 行业因子暴露
    p: pd.DataFrame = expo_pca.loc[td].reindex_like(
        pd.DataFrame(index=list(stk_ingredient), columns=expo_pca.columns)).dropna(axis=0)  # 组合资产，PCA因子暴露

    # 约束相对指数的超额暴露
    wb = ind_cons.loc[td, list(ind_ingredient)]
    wb /= wb.sum()
    f_del = expo_style.loc[td].reindex_like(pd.DataFrame(index=list(wb.index), columns=xf.columns)).T @ wb
    h_del = expo_indus.loc[td].reindex_like(pd.DataFrame(index=list(wb.index), columns=h.columns)).T @ wb
    p_del = expo_pca.loc[td].reindex_like(pd.DataFrame(index=list(wb.index), columns=expo_pca.columns)).T @ wb
    assert f_del.isna().sum() + h_del.isna().sum() + p_del.isna().sum() == 0
    fl = np.ones([xf.shape[-1], 1]) * FL + f_del.values.reshape(-1, 1)
    fh = np.ones([xf.shape[-1], 1]) * FH + f_del.values.reshape(-1, 1)
    hl = np.ones([h.shape[-1], 1]) * HL + h_del.values.reshape(-1, 1)
    hh = np.ones([h.shape[-1], 1]) * HH + h_del.values.reshape(-1, 1)
    pl = np.ones([p.shape[-1], 1]) * PL + p_del.values.reshape(-1, 1)
    ph = np.ones([p.shape[-1], 1]) * PH + p_del.values.reshape(-1, 1)
    # 约束组合内个股最大权重
    K1 = wb.max() * 100
    if K1 > K:
        K2 = int(K1) + 1
        print(f'*warning: K={K}% < 指数中成分股最大权重{K1:.2}%, 放宽到{K2}%')
        K1 = K2
        optimize_iter_info.loc['K1', td] = K1
    k = np.ones([len(stk_ingredient), 1]) * (max(K, K1) / 100)
    # 约束换手率
    complement_lst = set(lst_w.replace(0, np.nan).dropna().index).difference(stk_ingredient)  # 上期持有 组合资产未覆盖
    print(f'\t清除在上期持仓内但不在当前选股池的({len(complement_lst)}支): {",".join(list(complement_lst))}')
    d_del = lst_w.loc[list(complement_lst)].abs().sum().values[0] if len(lst_w) > 0 else 0
    d = D - d_del

    # 优化问题求解
    w = cp.Variable((len(stk_ingredient), 1), nonneg=True)
    objective = cp.Maximize(a.values.reshape(1, -1) @ w)
    # objective = cp.Minimize(a.values.reshape(1, -1) @ w)
    constraints = [
        fl - xf.values.T @ w <= 0,
        xf.values.T @ w - fh <= 0,
        hl - h.values.T @ w <= 0,
        h.values.T @ w <= hh,
        pl - p.values.T @ w <= 0,
        p.values.T @ w - ph <= 0,
        w <= k,
        cp.sum(w) == 1
    ]
    if len(lst_w) > 0:
        w_lst = lst_w.reindex_like(pd.DataFrame(index=list(stk_ingredient), columns=lst_w.columns)).fillna(0).values
        constraints.append(cp.norm(w - w_lst, 1) <= d)
    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=opt_verbose, solver='ECOS', max_iters=1000)
    if prob.status == 'optimal_inaccurate':
        result = prob.solve(verbose=opt_verbose, solver='ECOS', max_iters=10000)
    # result = prob.solve(verbose=True, solver='ECOS', abstol=1e-8, max_iters=100)

    if prob.status == 'optimal':
        w1 = w.value.copy()
        w1[w1 < wei_tole] = 0
        w1 = w1 / w1.sum()
        print(f'.换手率({np.abs(w_lst - w1).sum() + d_del:.3f}/2)\t.持仓个股数 {(w1 > 0).sum()}')
        lst_w = pd.DataFrame(w1, index=list(stk_ingredient), columns=[td])
    else:
        raise Exception(f'.{prob.status} problem')
        # print(f'.{prob.status} problem, portfolio ingredient unchanged')
        # if len(lst_w) > 0:
        #     lst_w.columns = [td]
    portfolio_weight = pd.concat([portfolio_weight, lst_w.T])


# %%
def tf_portfolio_weight(portfolio_weight, dat_path, suffix, ishow=False, t_file='portfolio_weight_{}.csv', g_file='figure_portfolio_size_{}.png'):
    """Table & Graph: Holding weight, holding number"""
    portfolio_weight.sum(axis=1)
    portfolio_weight.to_csv(dat_path + t_file.format(suffix))
    (portfolio_weight > 0).sum(axis=1).plot()
    plt.savefig(dat_path + g_file.format(suffix))
    if ishow:
        plt.show()
    else:
        plt.close()


def tf_historical_result(close_adj, tdays_d, begin_date, end_date, portfolio_weight, ind_cons, mkt_type, dat_path, suffix, p_file='figure_result_wealth_{}.png', t_file='table_result_wealth_{}.xlsx'):
    """Table & Graph: wealth curve comparison"""
    df = pd.DataFrame()

    rtn_w2w = close_adj.pct_change(5).loc[tdays_d.loc[begin_date: end_date].index]
    rtn_portfolio = (portfolio_weight * rtn_w2w.reindex_like(portfolio_weight)).sum(axis=1)
    df['portfolio'] = rtn_portfolio

    ind_cons_w = ind_cons.loc[tdays_d.loc[begin_date: end_date].index]
    rtn_ind = (ind_cons_w * rtn_w2w.reindex_like(ind_cons_w)).sum(axis=1)
    df[mkt_type] = rtn_ind

    df['Excess'] = df['portfolio'] - df[mkt_type]

    tmp = df.cumsum().add(1)
    tmp.plot()
    plt.savefig(dat_path + p_file.format(suffix))
    plt.close()
    tmp.to_excel(dat_path + t_file.format(suffix))
    del tmp


suffix = f'{alpha_name}_weekly[{mkt_type}]'
optimize_iter_info.T.to_excel(dat_path + save_suffix + '/' + f'opt_info_{suffix}.xlsx')
tf_portfolio_weight(portfolio_weight, f'{dat_path}{save_suffix}/', suffix)
tf_historical_result(close_adj, tdays_d, begin_date, end_date, portfolio_weight, ind_cons, mkt_type, f'{dat_path}{save_suffix}/', suffix)
