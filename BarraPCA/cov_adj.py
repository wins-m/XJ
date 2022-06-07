"""
(created by swmao on April 28th)
风险矩阵估计，包括
- 共同因子协方差矩阵 MFM
- 特异风险方差矩阵 SRR
"""
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.cov_a import *


# %%
def main():
    # %%
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    fr = get_barra_factor_return_daily(conf)
    mkt_val = pd.read_csv(conf['marketvalue'], index_col=0, parse_dates=True)
    sr, exposure = get_barra_factor_exposure_daily(conf, use_temp=True)  # 注意T0期因子收益对应T+1期个股收益

    # %%
    self = MFM(fr.iloc[-1000:])
    Newey_West_adj_cov = self.newey_west_adj_by_time()
    eigen_risk_adj_cov = self.eigen_risk_adj_by_time()
    vol_regime_adj_cov = self.vol_regime_adj_by_time()
    self.Newey_West_adj_cov = Newey_West_adj_cov
    self.eigen_risk_adj_cov = eigen_risk_adj_cov
    self.vol_regime_adj_cov = vol_regime_adj_cov
    self.save_vol_regime_adj_cov(conf['dat_path_barra'])

    # %%
    fbegin_date = '2012-01-01'  # '2019-07-01'
    self = SRR(fr=fr.loc[fbegin_date:], sr=sr.loc[fbegin_date:], expo=exposure.loc[fbegin_date:], mv=mkt_val.loc[fbegin_date:])
    print(self.T)
    Ret_U = self.specific_return_by_time()
    # self.u = Ret_U
    Raw_var, Newey_West_adj_var = self.newey_west_adj_by_time()
    # self.SigmaRaw, self.SigmaNW = Raw_var, Newey_West_adj_var
    Gamma_STR, Sigma_STR = self.struct_mod_adj_by_time()
    # self.GammaSM, self.SigmaSM = Gamma_STR, Sigma_STR
    self.plot_structural_model_gamma()
    self.SigmaSM.count(axis=1).plot(); plt.show()
    Sigma_Shrink = self.baysian_shrink_by_time()
    # self.SigmaSH = Sigma_Shrink
    Lambda_VRA = Sigma_VRA = self.vol_regime_adj_by_time()
    # self.SigmaVRA = Sigma_VRA


# %% TODO: Plot and Check
# tmp = pd.DataFrame()
# for k, sr in zip(['SigmaRaw', 'SigmaNW', 'SigmaSM', 'SigmaSH', 'SigmaVRA'],
#                  [self.SigmaRaw, self.SigmaNW, self.SigmaSM, self.SigmaSH, self.SigmaVRA]):
#     B = (self.u.reindex_like(sr) / sr).rolling(21).std()
#     w = (self.mkt_val.reindex_like(B) * (1 - B.isna())).apply(lambda s: s / s.sum(), axis=1)
#     tmp = tmp.append((B * w).sum(axis=1).rolling(120).mean().rename(k))
#     # tmp = tmp.append((self.u.reindex_like(sr) / sr).rolling(21).std().mean(axis=1).rename(k))
# tmp = tmp.T
# tmp.plot()
# plt.show()


# %% TODO: Save Results
