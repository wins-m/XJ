"""
(created by swmao on April 28th)
风险矩阵估计，包括
- 共同因子协方差矩阵 MFM
- 特异风险方差矩阵 SRR
"""
import os
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.cov_a import *
from supporter.transformer import cvg_f_fill
from tqdm import tqdm
R_MFM = True
R_SRR = True
R_MERGE = False
P_NUM = 4  # suggest 1 or 4


def factor_covariance_model(conf):
    """共同因子协方差矩阵 MFM"""
    fr = get_barra_factor_return_daily(conf)
    self = MFM(fr)  # self = MFM(fr.iloc[-1000:])
    Newey_West_adj_cov = self.newey_west_adj_by_time()
    eigen_risk_adj_cov = self.eigen_risk_adj_by_time()
    vol_regime_adj_cov = self.vol_regime_adj_by_time()
    self.save_factor_covariance(conf['dat_path_barra'], level='NW')
    self.save_factor_covariance(conf['dat_path_barra'], level='Eigen')
    self.save_factor_covariance(conf['dat_path_barra'], level='VRA')
    # mfm = self
    # self.Newey_West_adj_cov = Newey_West_adj_cov
    # self.eigen_risk_adj_cov = eigen_risk_adj_cov
    # self.vol_regime_adj_cov = vol_regime_adj_cov


def specific_risk_model(conf):
    """特异风险矩阵 SRR"""
    fr = get_barra_factor_return_daily(conf)
    mkt_val = pd.read_csv(conf['marketvalue'], index_col=0, parse_dates=True)
    sr, exposure = get_barra_factor_exposure_daily(conf, use_temp=True)  # 注意T0期因子收益对应T+1期个股收益
    exposure = cvg_f_fill(exposure, w=10, q=.75, ishow=False)  # f-fill barra exposure
    bd = '2012-01-01'  # '2019-07-01'
    factor_return = fr.loc[bd:]
    asset_return = sr.loc[bd:]
    factor_exposure = exposure.loc[bd:]
    market_value = mkt_val.loc[bd:]

    self = SRR(fr=factor_return, sr=asset_return, expo=factor_exposure, mv=market_value)
    Ret_U = self.specific_return_by_time()  # Specific Risk
    Raw_var, Newey_West_adj_var = self.newey_west_adj_by_time()  # New-West Adjustment
    Gamma_STR, Sigma_STR = self.struct_mod_adj_by_time()  # Structural Model Adjustment
    Sigma_Shrink = self.bayesian_shrink_by_time()  # Bayesian Shrinkage Adjustment
    Lambda_VRA, Sigma_VRA = self.vol_regime_adj_by_time()  # Volatility Regime Adjustment
    self.save_vol_regime_adj_risk(conf['dat_path_barra'], level='Raw')
    self.save_vol_regime_adj_risk(conf['dat_path_barra'], level='NW')
    self.save_vol_regime_adj_risk(conf['dat_path_barra'], level='SM')
    self.save_vol_regime_adj_risk(conf['dat_path_barra'], level='SH')
    self.save_vol_regime_adj_risk(conf['dat_path_barra'], level='VRA')
    # srr = self
    # self.u = Ret_U
    # self.SigmaRaw, self.SigmaNW = Raw_var, Newey_West_adj_var
    # self.GammaSM, self.SigmaSM = Gamma_STR, Sigma_STR
    # # self.plot_structural_model_gamma()
    # # self.SigmaSM.count(axis=1).plot(); plt.show()
    # self.SigmaSH = Sigma_Shrink
    # self.LambdaVRA, self.SigmaVRA = Lambda_VRA, Sigma_VRA

    # TODO: Plot and Check
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


def combine_risk_matrices(conf):
    """Combine X F X + Delta as V"""
    # print('\nV = X F X + Delta ...')
    if P_NUM == 1:
        _combine_risk_matrices(conf, '2014-01-01', '2022-12-31')
    else:
        from multiprocessing import Pool, RLock, freeze_support
        freeze_support()
        p = Pool(P_NUM, initializer=tqdm.set_lock, initargs=(RLock(),))
        cnt = 0
        for year in range(2014, 2023):
            cnt += 1
            p.apply_async(_combine_risk_matrices, args=(conf, f'{year}-01-01', f'{year}-12-31', cnt % P_NUM))
        p.close()
        p.join()


def _combine_risk_matrices(conf, bd='2014-01-01', ed='2022-12-31', pos=0):
    _, X = get_barra_factor_exposure_daily(conf, use_temp=True)  # 注意T0期因子收益对应T+1期个股收益
    X = cvg_f_fill(X, w=10, q=.75, ishow=False).loc[bd: ed]
    F = pd.read_csv(conf['factor_covariance'], index_col=[0, 1], parse_dates=[0]).loc[bd: ed]
    D = pd.read_csv(conf['specific_risk'], index_col=0, parse_dates=True).loc[bd: ed]

    td_intersect = keep_index_intersection((X.index.get_level_values(0), F.index.get_level_values(0), D.index))
    path = conf['dat_path_barra'] + 'V_VRA_VRA/'
    os.makedirs(path, exist_ok=True)
    path += '{}.h5'

    # td = '2015-01-05'
    desc = f'[{bd},{ed}]'
    loop_bar = tqdm(range(len(td_intersect)), ncols=80, desc=desc, delay=0.01, position=pos, ascii=False)
    for i in loop_bar:  # range(len(td_intersect)):
        td = td_intersect[i]

        x, f, d = X.loc[td], F.loc[td], D.loc[td]
        x = x.dropna(how='all', axis=1)  # degree of beta exposure all missing: ind#29
        f = f.dropna(how='all', axis=1).dropna(how='all')  # degree of beta covariance all missing: ind#29
        d = d.dropna()  # exclude asset without individual specific risk
        fct_intersect = keep_index_intersection((x.columns, f.index))
        x, f = x.loc[:, fct_intersect], f.loc[fct_intersect, fct_intersect]
        stk_intersect = keep_index_intersection((x.index, d.index,))
        x, d = x.loc[stk_intersect], d.loc[stk_intersect]
        d = pd.DataFrame(np.diag(d ** 2), index=d.index, columns=d.index)
        # assert d.isna().sum().sum() == 0
        # assert x.isna().sum().sum() == 0
        # assert f.isna().sum().sum() == 0
        v = x @ f @ x.T + d
        v = v.set_index([[td] * len(v), v.index])

        key = td.strftime('TD%Y%m%d')
        v.to_hdf(path.format(key), key=key, complevel=9)
        # progressbar(i+1, len(td_intersect), msg=f'\tdate: {td.strftime("%Y-%m-%d")}')
    # print()

    # print(f'Save in {path}, key like `TD%Y%m%d`')


def main():
    # %%
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    # %%
    if R_MFM:
        factor_covariance_model(conf)
    if R_SRR:
        specific_risk_model(conf)
    if R_MERGE:
        combine_risk_matrices(conf)


# %%
if __name__ == '__main__':
    main()

