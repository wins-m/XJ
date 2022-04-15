"""
(created by swmao on April 11th)
选股超额 调仓日最高X只指增 相对全体同类产品 相对指数 表现对比

"""
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style("darkgrid")
plt.rc("figure", figsize=(16, 10))
# plt.rc("figure", figsize=(8, 3))
plt.rc("savefig", dpi=90)
# plt.rc("font", family="sans-serif")
plt.rc("font", size=12)
# plt.rc("font", size=10)

plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"


import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml

conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

dat_path = '/mnt/c/Users/Winst/Documents/data_local/BARRA/'  # conf['dat_path_barra']
cache_path = dat_path + 'barra_exposure.h5'
panel_path = dat_path + 'barra_panel.h5'
fval_path = dat_path + 'barra_fval.h5'
omega_path = dat_path + 'barra_omega.h5'


# %%
f_kind = '300'  # '500'
idx_close = pd.read_csv(conf['idx_marketdata_close'], index_col=0, parse_dates=True, dtype=float)  # 股指
idx_rtn_ctc = idx_close.pct_change().iloc[1:]
ers = pd.read_excel(dat_path + f'excess_return_selecting[{f_kind}_tdays_d].xlsx', index_col=0, parse_dates=True)
er = pd.read_excel(dat_path + f'excess_return_raw[{f_kind}_tdays_d].xlsx', index_col=0, parse_dates=True)
fund_net_val = pd.DataFrame(pd.read_hdf(conf['fund_net_value'], key='refactor_net_value'))[ers.columns]
fund_return = fund_net_val.pct_change().iloc[1:]
ers_er = ers/er


# %%
def fund_selected_cum_ret(fv, fbegin_date='2018-04-01'):
    fund_rtn = fund_return.loc[fv.index[0]: fv.index[-1]]
    baseline = fund_rtn.mean(axis=1).rename(f'baseline({fund_rtn.shape[1]})')
    exc_rtn_sel_5 = fund_rtn[fv.rank(axis=1, ascending=False) <= 5].mean(axis=1).rename('ERS_top5')
    # exc_rtn_sel_10 = fund_rtn[fv.rank(axis=1, ascending=False) <= 10].mean(axis=1).rename('ERS_top10')
    # exc_rtn_sel_25 = fund_rtn[fv.rank(axis=1, ascending=False) <= 25].mean(axis=1).rename('ERS_top25')
    # exc_rtn_sel_50 = fund_rtn[fv.rank(axis=1, ascending=False) <= 50].mean(axis=1).rename('ERS_top50')
    idx_rtn = idx_rtn_ctc.loc[fv.index[0]: fv.index[-1], '000905.SH'].rename('Index')

    result_abs = pd.concat([
        exc_rtn_sel_5,
        # exc_rtn_sel_10,
        # exc_rtn_sel_25,
        # exc_rtn_sel_50,
        baseline, idx_rtn], axis=1)
    result_abs = result_abs.dropna()
    result_abs.loc[fbegin_date:].cumsum().add(1).plot(linewidth=3); plt.show()

    result_exc = result_abs.iloc[:, :-1] - result_abs.iloc[:, -1:].values.reshape(-1, 1)
    result_exc.loc[fbegin_date:].cumsum().add(1).plot(linewidth=3); plt.show()


fund_selected_cum_ret(ers.shift(1).iloc[1:])  # 日度选基

fund_selected_cum_ret(ers.shift(5).iloc[5::5, :].reindex_like(ers).fillna(method='ffill'))  # 周度


# %%
kw = 'ers'
fv = eval(kw)
ic_stat = pd.DataFrame(index=range(1, 21), columns=['IC', 'IC_IR', 'Rank_IC', 'Rank_IC_IR'])
for t in tqdm(range(1, 21)):
    ic = fv.shift(t).iloc[t:].apply(lambda s: s.corr(fund_return.loc[s.name]), axis=1)
    ic_rk = fv.shift(t).iloc[t:].apply(lambda s: s.corr(fund_return.loc[s.name], method='spearman'), axis=1)
    ic_stat.loc[t, 'IC'] = ic.mean()
    ic_stat.loc[t, 'IC_IR'] = ic.mean() / ic.std()
    ic_stat.loc[t, 'Rank_IC'] = ic_rk.mean()
    ic_stat.loc[t, 'Rank_IC_IR'] = ic_rk.mean() / ic_rk.std()

ic_stat.to_excel(dat_path + f'IC_Stat_{kw}.xlsx')
