"""
(created on April 9th by swmao)
**收益归因**
- 调仓日，回看60日，用T-1纯因子解释T日收益率（T-1纯因子由T的风格暴露算得）
- 指数和指增对风格因子的暴露，指增超额暴露，超额暴露获取的超额收益，60日求和
- 指增相对指数的超额收益，60日求和
- 总超额 - 超额暴露获得超额 = 选股超额

"""
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

sample_size = 60  # 归因回看天数

# %%  收益归因-基于净值（回归法）
b_date, e_date = '2012-01-01', '2022-03-31'

# 最后一个交易日收盘后，用T-1纯因子解释T日收益率，进行归因（T-1纯因子由T的风格暴露算得）
kind_freq = 'tdays_d'
tdays_freq = pd.read_csv(conf[kind_freq], header=None, dtype=object)
tdays_freq.index = pd.to_datetime(tdays_freq.iloc[:, 0])
tdays_freq = tdays_freq.loc[b_date: e_date].iloc[:, 0]

# 股指表现
idx_close = pd.read_csv(conf['idx_marketdata_close'], index_col=0, parse_dates=True, dtype=float)  # 股指
idx_rtn_ctc = idx_close.pct_change().iloc[1:].loc[b_date: e_date]

fund_val = pd.read_pickle(conf['refactor_net_value_5003'])  # 资产收益
all_fund_rtn: pd.DataFrame = fund_val.pct_change()  # 累计复权净值，盘前9:00更新


# %%
# kind_b, idx_rtn = '300', idx_rtn_ctc['000300.SH'].rename('baseline')  # CSI300
kind_b, idx_rtn = '500', idx_rtn_ctc['000905.SH'].rename('baseline')  # CSI500

attr_funds = pd.read_excel(dat_path + f'fund_stat[{kind_b}].xlsx', dtype=object)
fund_rtn = all_fund_rtn[attr_funds.main_code].astype(float).copy()
fund_rtn[fund_rtn.abs() > .1] = 0  # 基金日收益超过10%是异常值，取0
dat = pd.concat([idx_rtn, fund_rtn], axis=1).loc[b_date: e_date]

# 纯因子收益
barra_rtn = pd.read_csv(conf['barra_fval'], index_col=0, parse_dates=True, dtype=float)
barra_rtn = barra_rtn.shift(1).iloc[1:].loc[b_date: e_date]  # T0的风格因子用于解释T+1的资产收益
# barra_rtn.stack().hist(bins=100); plt.show()

# %%
src, tgt = barra_rtn, dat  # 风险因子日收益 解释 资产日收益

all_er = pd.DataFrame()
all_selected_er = pd.DataFrame()
date = '2012-02-07'  # tdays_freq[2]
for date in tqdm(tdays_freq):
    date_idx = src.loc[:date].iloc[-sample_size:].index  # 回看sample_size日
    if len(date_idx) < sample_size:
        continue
    src1 = src.loc[date_idx].dropna(axis=1)
    F = src1.values  # F, Y 只要过去sample_size日缺失，就不留
    if F.shape[0] == 0:
        continue
    tgt1 = tgt.loc[date_idx].dropna(axis=1)
    Y = tgt1.values
    X = (np.linalg.inv(F.T @ F) @ F.T @ Y)  # OLS coefficient, X = (F^T F)^{-1} F^T Y
    exposure = pd.DataFrame(X, index=src1.columns, columns=tgt1.columns)  # （基准，产品）因子暴露
    exposure_excess = exposure.iloc[:, 1:] - exposure['baseline'].values.reshape(-1, 1)  # 产品 超额暴露
    exposed_excess_rtn = pd.DataFrame(F @ exposure_excess.values, index=date_idx, columns=tgt1.columns[1:])  # 产品 超额暴露 的 超额收益
    exposed_er = exposed_excess_rtn.sum()  # 暴露获取超额，sample_size日求和
    excess_rtn = tgt1.iloc[:, 1:] - tgt1['baseline'].values.reshape(-1, 1)  # 超过基准指数的超额收益
    er = excess_rtn.sum()   # sample_size日 超过基准指数的 总超额收益
    selected_er = er - exposed_er  # sample_size日 来自选股的超额
    all_er = pd.concat([all_er, pd.DataFrame(er.rename(date)).T])
    all_selected_er = pd.concat([all_selected_er, pd.DataFrame(selected_er.rename(date)).T])

all_selected_er.to_excel(dat_path + f'excess_return_selecting[{kind_b}_{kind_freq}].xlsx')
all_er.to_excel(dat_path + f'excess_return_raw[{kind_b}_{kind_freq}].xlsx')
(all_er - all_selected_er).to_excel(dat_path + f'excess_return_exposed[{kind_b}_{kind_freq}].xlsx')

# %%
# from matplotlib import pyplot as plt
# exposed.sum().plot.bar(figsize=(16, 10))
# plt.show()
# df = pd.concat([tgt.loc[date_idx].sum().rename('actual'), exposed.sum().rename('exposed')], axis=1)
# df.plot.bar(figsize=(16, 10))
# plt.show()
