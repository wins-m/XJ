"""
(created by swmao on April 22nd)

"""
import warnings
warnings.simplefilter("ignore")

# %matplotlib inline
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

import pandas as pd
import numpy as np
import cvxpy as cp
import yaml
from tqdm import tqdm

conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

dat_path = conf['dat_path_barra']
begin_date = '2016-02-01'
end_date = '2022-03-31'

N_ingredient = 2000
FL = -.1
FH = .1
HL = -.2
HH = .2
PL = -.1
PH = .1
D = 2
K = .01
wei_tole = 1e-5

# %% Target to maximize
alpha_val = pd.read_csv(conf['data_path'] + 'factor_apm.csv', index_col=0, parse_dates=True)
alpha_val = alpha_val.apply(lambda s: (s - s.min()) / (s.max() - s.min()))
dat = alpha_val.loc[begin_date: end_date]
dat.tail()

# # Risk Value 风格收益（用到未来的资产收益），风格暴露不用到未来情况
# fv_barra = pd.read_csv(conf['barra_fval'], index_col=0, parse_dates=True).loc[begin_date: end_date]
# fv_barra.tail()

# %% check IC
import sys

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")

from supporter.factor_operator import cal_ic

close_adj = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True)
dat_ic = cal_ic(fv_l1=alpha_val.shift(1).loc[begin_date: end_date],
                ret=close_adj.pct_change().loc[begin_date: end_date],
                lag=5,
                ranked=True)
dat_ic.mean()

# %%
# Exposure 当天的因子暴露
expo = pd.DataFrame()
for _ in range(int(begin_date.split('-')[0]), int(end_date.split('-')[0]) + 1):
    expo = pd.concat([expo, pd.DataFrame(pd.read_hdf(conf['barra_panel'], key=f'y{_}'))])
expo: pd.DataFrame = expo.loc[begin_date: end_date]

cols_style = [c for c in expo.columns if 'rtn' not in c and 'ind' not in c and 'country' != c]
cols_indus = [c for c in expo.columns if 'ind' in c]
expo_style: pd.DataFrame = expo[cols_style].groupby(
    expo[cols_style].index.get_level_values(0)).apply(lambda s: (s - s.mean()) / s.std())
expo_indus: pd.DataFrame = expo[cols_indus]

expo_style.tail()

# Baseline Portfolio 股指成分
mkt_type = 'CSI500'
ind_cons = pd.read_csv(conf['idx_constituent'].format(mkt_type), index_col=0, parse_dates=True).reindex_like(dat).fillna(0)
ind_cons = ind_cons / 100
ind_cons.tail()

# Trade days 交易日期
tdays_d = pd.read_csv(conf['tdays_w'], header=None, index_col=0, parse_dates=True).loc[begin_date: end_date]
tdays_d['tdays_d'] = tdays_d.index
tradedates = tdays_d.tdays_d

# Exposure on PCA factors top 20
expo_pca: pd.DataFrame = pd.read_pickle(conf['data_path'] + 'exposure_pca20_1602_2203.pkl')
expo_pca = expo_pca.groupby(expo_pca.index.get_level_values(0)).apply(lambda s: (s - s.mean()) / s.std())
# expo_pca = pd.DataFrame()
# for pn in tqdm(range(20)):
#     kw = f'pc{pn:03d}'
#     df = pd.read_csv(conf['dat_path_pca'] + f'{kw}.csv', index_col=0, parse_dates=True).loc[begin_date: end_date]
#     expo_pca = pd.concat([expo_barra, df.stack().rename(kw)], axis=1)
# expo_pca.to_pickle(conf['data_path'] + 'exposure_pca20_1602_2203.pkl')

# %%
portfolio_weight = pd.DataFrame()
lst_w: pd.DataFrame = pd.DataFrame()
w_lst = np.zeros([N_ingredient, 1])

td0 = tradedates[-10]
for td0 in tradedates:
    td = td0.strftime('%Y-%m-%d')
    # 个股覆盖
    stk_exposed_pca = set(expo_pca.loc[td].dropna(axis=1).index)  # PCA 暴露覆盖个股
    stk_exposed_barra = set(expo.loc[td].index)  # Barra 暴露 覆盖个股
    stk_exposed = stk_exposed_barra.intersection(stk_exposed_pca)  # 存在暴露的个股

    stk_ind_cons = set(ind_cons.loc[td][ind_cons.loc[td] > 0].index)  # 指数成分股
    stk_alpha = set(dat.loc[td].dropna().index)  # Alpha 覆盖个股

    tmp = stk_alpha.difference(stk_exposed)  # alpha 覆盖个股中barra暴露未知的
    # stk_ingredient = set(dat.loc[td, list(stk_alpha)].rank(ascending=False).sort_values().index[:N_ingredient])  # 组合用股
    stk_ingredient = set(dat.loc[td, list(stk_alpha.difference(tmp))].rank(ascending=False).sort_values().index[:N_ingredient])  # 组合用股
    print(td + f'\tAlpha:{stk_alpha.__len__()}({tmp.__len__()})\t覆盖(暴露缺失)')
    del tmp
    complement_lst = set(lst_w.index).difference(stk_ingredient)  # 上期持有 组合资产未覆盖
    # complement_ind = stk_ind_cons.difference(stk_exposed)  # 指数成分股 组合资产未覆盖

    a = dat.loc[td, list(stk_ingredient)]  # 最大化的 N_ingredient 个alpha

    xf = expo_style.loc[td].reindex_like(pd.DataFrame(index=list(stk_ingredient), columns=cols_style)).fillna(0)  # 组合资产 风格因子暴露
    h = expo_indus.loc[td].reindex_like(pd.DataFrame(index=list(stk_ingredient), columns=cols_indus)).fillna(0)  # 组合资产 行业因子暴露
    p = expo_pca.loc[td].reindex_like(pd.DataFrame(index=list(stk_ingredient), columns=expo_pca.columns)).fillna(0)  # 组合资产，PCA因子暴露

    wb = ind_cons.loc[td, list(stk_ind_cons)]
    f_del = expo_style.loc[td].reindex_like(pd.DataFrame(index=list(wb.index), columns=cols_style)).fillna(0).T @ wb
    h_del = expo_indus.loc[td].reindex_like(pd.DataFrame(index=list(wb.index), columns=cols_indus)).fillna(0).T @ wb
    p_del = expo_pca.loc[td].reindex_like(pd.DataFrame(index=list(wb.index), columns=expo_pca.columns)).fillna(0).T @ wb
    d_del = lst_w.loc[list(complement_lst)].abs().sum().values[0] if len(lst_w) > 0 else 0

    fl = np.ones([len(cols_style), 1]) * FL - f_del.values.reshape(-1, 1)
    fh = np.ones([len(cols_style), 1]) * FH - f_del.values.reshape(-1, 1)
    hl = np.ones([len(cols_indus), 1]) * HL - h_del.values.reshape(-1, 1)
    hh = np.ones([len(cols_indus), 1]) * HH - h_del.values.reshape(-1, 1)
    pl = np.ones([expo_pca.shape[1], 1]) * PL - p_del.values.reshape(-1, 1)
    ph = np.ones([expo_pca.shape[1], 1]) * PH - p_del.values.reshape(-1, 1)
    k = np.ones([N_ingredient, 1]) * K
    d = D - d_del

    w = cp.Variable((N_ingredient, 1), nonneg=True)
    objective = cp.Maximize(a.values.reshape(1, -1) @ w)
    constraints = [
        fl <= xf.values.T @ w,
        xf.values.T @ w <= fh,
        # hl <= h.values.T @ w,
        # h.values.T @ w <= hh,
        pl <= p.values.T @ w,
        p.values.T @ w <= ph,
        w <= k, cp.sum(w) == 1
    ]
    if len(lst_w) > 0:
        w_lst = lst_w.reindex_like(pd.DataFrame(index=list(stk_ingredient), columns=lst_w.columns)).fillna(0).values
        constraints.append(cp.norm(w - w_lst, 1) <= d)

    prob = cp.Problem(objective, constraints)
    result = prob.solve(verbose=False)

    if prob.status == 'optimal':
        w1 = w.value.copy()
        w1[w1 < wei_tole] = 0
        w1 = w1 / w1.sum()
        print(f' turnover {np.abs(w_lst - w1).sum() + d_del:.3f};\tportfolio size {(w1 > 0).sum()}')
        lst_w = pd.DataFrame(w1, index=list(stk_ingredient), columns=[td])
    else:
        print(f' {prob.status} problem, portfolio ingredient unchanged')
        if len(lst_w) > 0:
            lst_w.columns = [td]
    portfolio_weight = pd.concat([portfolio_weight, lst_w.T])

# %%
portfolio_weight.to_csv(dat_path + 'portfolio_factor_apm_weekly.csv')
portfolio_weight.sum(axis=1)
(portfolio_weight > 0).sum(axis=1).plot()
plt.show()
(portfolio_weight.iloc[46].dropna() > wei_tole).sum()


# %%

rtn_w2w = close_adj.pct_change(5).loc[tdays_d.loc[begin_date: end_date].index]
rtn_portfolio = (portfolio_weight * rtn_w2w.reindex_like(portfolio_weight)).sum(axis=1)

ind_cons_w = ind_cons.loc[tdays_d.loc[begin_date: end_date].index]
rtn_csi500 = (ind_cons_w * rtn_w2w.reindex_like(ind_cons_w)).sum(axis=1)

df = pd.DataFrame()
df['portfolio'] = rtn_portfolio
df['csi500'] = rtn_csi500
tmp = df.cumsum().add(1)
tmp.plot()
plt.show()
tmp.to_excel(dat_path + 'wealth_portfolio_factor_apm_weekly.xlsx')
del tmp
