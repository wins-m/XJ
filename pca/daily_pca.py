"""
(created by swmao on April 13th)
# PCA

- 股池：全市场

- 上市：去除新上市90日

- 日度迭代

    - 过去150天
    - 存在完整收益序列的股池
    - 日收益超过0.1，缩到0.1
    - 日收益协方差矩阵，PCA分解
    - 主成分权重，获得主成分150日收益
    - 个股收益对主成分收益回归，得到日度截面因子暴露

- `data_local/PCA/`

    - `pc00X.csv`: 纯因子00X在个股上的暴露
    - `PricipalCompnent/`: 所有纯因子，单日在个股上的暴露（回归系数）
    - `PrincipalFactorReturn/`: 所有纯因子，单日过去150日，因子收益
    - `PrincipalStockWeight/`: 所有纯因子，单日计算的特征向量，（平方后是）在个股上的权重

"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import yaml

conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

dat_path = conf['data_path'] + 'PCA/'
sample_size = 150
principle_num = 10
error_tolerance = 1e-5
ipo_period = 90
component_name = 'principalStockWeight'
factor_name = 'PrincipalFactorReturn'
exposure_name = 'PrincipalExposure'
principal_path = dat_path + 'PrincipalComponent/'
principal_weight_path = dat_path + 'PrincipalStockWeight/'
principal_return_path = dat_path + 'PrincipalFactorReturn/'

fbegin_date = '2012-01-01'
fend_date = '2022-03-31'

os.makedirs(dat_path, exist_ok=True)
os.makedirs(principal_path, exist_ok=True)
os.makedirs(principal_weight_path, exist_ok=True)
os.makedirs(principal_return_path, exist_ok=True)


# %%
rtn_ctc = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True).pct_change().iloc[1:]

pd.read_csv(conf['idx_constituent'].format('CSI500'), index_col=0, parse_dates=True)
tradeable_ipo = pd.DataFrame(pd.read_hdf(conf['a_list_tradeable'], key='ipo')).shift(ipo_period).iloc[ipo_period:].fillna('False')
tdays_d = pd.read_csv(conf['tdays_d'], header=None, index_col=0, parse_dates=True)
tdays_d['tdays_d'] = tdays_d.index


# %%
def pca(x, n_dim=None):
    s = x.reshape(x.shape[0], -1)
    # s = s - s.mean(0)
    g = np.cov(s, rowvar=False)
    d, v = np.linalg.eig(g)
    return v[:, :n_dim].real if n_dim else v.real


factor_values = [pd.DataFrame() for _ in range(principle_num)]
# td = tdays_d.loc[fbegin_date: fend_date].iloc[0, 0]
for td in (tdays_d.loc[fbegin_date: fend_date, 'tdays_d']):
    src_range = tdays_d.loc[:td].iloc[-sample_size:].index
    dat = rtn_ctc.loc[src_range]
    dat1 = dat[tradeable_ipo.reindex_like(dat).fillna(False)]
    dat2 = dat1.loc[:, dat1.std() > 1e-5].dropna(axis=1).copy()
    dat2[dat2 > .11] = .1
    dat2[dat2 < -.11] = -.1
    print(f"{td.strftime('%Y-%m-%d')}:\t{dat.shape} -> {dat2.shape}")

    u = pca(dat2.values, n_dim=principle_num)
    wei = pd.DataFrame(u, index=dat2.columns, columns=[f'pc{fn:03d}' for fn in range(principle_num)])
    wei.to_csv(principal_weight_path + f"{component_name}{td.strftime('%Y%m%d')}.csv")

    f_rtn = (dat2 @ u**2)
    fr = f_rtn.rename(columns={x: f'pc{x:03d}' for x in range(principle_num)})
    fr.to_csv(principal_return_path + f"{factor_name}{td.strftime('%Y%m%d')}.csv")

    Y, X = dat2, f_rtn  # sm.add_constant(f_rtn)
    # mod = sm.OLS(dat2.iloc[:, 4], X).fit()
    fval = (np.linalg.inv(X.T @ X) @ X.T @ Y)
    fv = fval.T.rename(columns={x: f'pc{x:03d}' for x in range(principle_num)})
    fv.to_csv(principal_path + f"{exposure_name}{td.strftime('%Y%m%d')}.csv")
    for fn in range(principle_num):
        factor_values[fn] = pd.concat([factor_values[fn], fval.loc[fn].rename(td.strftime("%Y-%m-%d"))], axis=1)


# %%
for fn in range(principle_num):
    # factor_values[fn] = pd.read_csv(dat_path + f'pc{fn:03d}.csv', index_col=0, parse_dates=True)
    factor_values[fn].T.to_csv(dat_path + f'pc{fn:03d}.csv')
