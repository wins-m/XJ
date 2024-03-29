"""
(created by swmao on April 1st)
### 纯因子收益率计算（国家+风格+行业）`cal_factor_return.py`
- 计算纯因子收益率：T-1期结束时的风格暴露，对应T期的资产收益，得到T期的纯因子收益预测（解释T期的资产收益）
- 时间范围：2012-01-01 ~ 2022-03-31
- 缓存/结果地址：
    - `/mnt/c/Users/Winst/Documents/data_local/BARRA/`
- Y: 全市场 ctc收益率（昨日Close买，今日Close卖）
    - 去除：上市 120 交易日内；昨日、今日停牌；~~昨日涨停、今日跌停~~
- X: 国家(1) 风格(10) 行业(29/30)
    - 风格因子，历年原始值存在`barra_exposure.h5`（key形如 `y2022`）
    - 风格因子日截面，对行业正交（size）or 对行业和size正交（其他9个）
    - 行业选用中信一级`indus_citic`29或30个 2019.12.2开始30个（新增“综合金融”）
    - 正交化后的面板存在`barra_panel.h5`（key形如 `y2022`）
- WLS（Menchero & Lee, 2015)
    $$
    \bold{\Omega} = \bold{R} (\bold{R}^T \bold{X}^T \bold{V}
    \bold{X} \bold{R})^{-1} \bold{R}^T \bold{X}^T \bold{V} \\

    \bold{F}_{K \times 1} = \bold{\Omega}_{K \times N} \bold{Y}_{N \times 1}
    $$
    - 纯因子构成存在`barra_omega.h5`（key形如`d20220101`）
    - 纯因子收益率存在`barra_fval.h5`（key形如 `y2022`）
    - 历年合并为`barra_fval_20120104_20220331.csv`

"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from tqdm import tqdm
import sys

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.request import get_ind_citic_all_tradingdate
from supporter.transformer import cvg_f_fill


#
def main():
    #
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    #
    cal_fac_ret(conf)
    combine(conf['barra_factor_value'])


def cal_fac_ret(conf):
    """计算纯因子收益率，缓存过程文件，分年循环"""
    #
    data_pat = conf['dat_path_barra']
    cache_path = conf['barra_exposure']  # 对齐的风格因子原始值
    panel_path = conf['barra_panel']  # 对行业、规模正交后，WLS用的面板
    fval_path = conf['barra_factor_value']  # 纯因子收益率
    omega_path = conf['barra_wls_omega']  # 纯因子构成（各资产权重）

    bd, ed = '2012-01-01', '2099-12-31'
    industry: pd.DataFrame = get_ind_citic_all_tradingdate(conf, bd=bd, ed=ed)
    del bd, ed

    # Stock Returns, without ST, new-IPO
    close_adj = pd.read_csv(conf['closeAdj'], dtype='float', index_col=0, parse_dates=True)

    a_list_tradeable = conf['a_list_tradeable']  # "/mnt/c/Users/Winst/Documents/data_local/a_list_tradeable.hdf"
    t_ipo = pd.DataFrame(pd.read_hdf(a_list_tradeable, key='ipo'))
    t_suspend = pd.DataFrame(pd.read_hdf(a_list_tradeable, key='suspend'))
    mul = pd.DataFrame(np.ones_like(close_adj), index=close_adj.index, columns=close_adj.columns)
    mul *= t_ipo.shift(120).fillna(False)  # 上市 120 个交易日后
    mul *= t_suspend & t_suspend.shift(1).fillna(True)  # 昨日、今日均不停牌

    rtn_close_adj = close_adj.pct_change()  # ctc 收益率
    rtn_close_adj *= mul.reindex_like(rtn_close_adj).replace(False, np.nan)  # 要去除的，令当天收益率为空值
    rtn_close_adj = rtn_close_adj.shift(-1)  # T期的资产收益由T-1期因子暴露解释
    # rtn_closeAdj.loc['2020-01-01':'2020-12-31'].count(axis=1).plot()

    #  TODO: stat of return
    rtn = rtn_close_adj
    rtn[rtn > 0.11] = np.nan
    rtn[rtn < -0.11] = np.nan

    # year = 2021
    for year in range(2022, 2011, -1):
        #
        print('\n', year)
        begin_date = f'{year}-01-01'  # '2012-01-01'
        end_date = f'{year}-12-31'

        # dat = pd.read_pickle(data_pat + 'dat2122.pkl')
        try:
            dat = pd.DataFrame(pd.read_hdf(cache_path, key=f'y{year}'))
        except KeyError:
            access_barra = pd.read_excel(conf['access_barra'])  # get_barra.py 得到的csv文件
            barra_files = access_barra.CSV.to_list()
            # access_barra.head(2)
            dat = pd.DataFrame()
            for file in tqdm(barra_files):
                fv = pd.read_csv(data_pat + file, index_col=0, parse_dates=True).loc[begin_date: end_date]
                fn = file.split('.')[0]
                fv1 = fv.stack().rename(fn)
                dat = pd.concat([dat, fv1], axis=1)
            dat.to_hdf(cache_path, key=f'y{year}', complevel=9)  # 缓存一年的barra因子
        dat = cvg_f_fill(fr=dat, w=10, q=.75, ishow=False, notify=True)  # f-fill barra exposure

        #
        indus = industry.stack().reindex_like(dat).rename('indus')
        print(f'Industry Missing {100 * indus.isna().mean():.2f} %')
        indus = pd.get_dummies(indus, prefix='ind')

        rtn_ctc = rtn_close_adj.stack().reindex_like(dat).rename('rtn_ctc')  # 21-22年收益率
        print(f'Return Missing {100 * rtn_ctc.isna().mean():.2f} %')  # 缺失情况
        #
        rtn_ctc.unstack().cov()

        panel = pd.concat([rtn_ctc, dat, indus, ], axis=1)
        panel['country'] = 1

        print('before', panel.shape)
        panel = panel.dropna()
        print('after missing-drop', panel.shape)

        factor_style = dat.columns.to_list()
        factor_indus = indus.columns.to_list()

        fval = pd.DataFrame()
        all_dates = panel.index.get_level_values(0).unique()

        # Cross-section (daily) WLS
        td = all_dates[0]
        for td in tqdm(all_dates[:]):
            #
            pan = panel.loc[td].copy()
            factor_i = [col for col in factor_indus if pan[col].sum() > 0]

            # WLS 权重：市值对数
            mv = panel.loc[td, 'size'].apply(lambda _: np.exp(_))
            w_mv = mv.apply(lambda _: np.sqrt(_))
            w_mv = w_mv / w_mv.sum()
            mat_v = np.diag(w_mv)

            # 风格因子对行业、市值正交：注意此后的size已经调整！
            for col in factor_style:
                if col == 'size':
                    x = panel.loc[td, factor_i]
                else:
                    x = panel.loc[td, ['size'] + factor_i]
                y = panel.loc[td, col]
                est = sm.OLS(y, x).fit()
                pan.loc[:, col] = est.resid

            # date日panel中的因子用正交化后的替换
            panel.loc[td, factor_style] = pan.loc[:, factor_style].values

            # 最终进入回归的因子
            if pan[factor_i].isna().any().sum() > 1:  # 19年前行业分类只有30个
                f_cols = ['country'] + factor_style + factor_i[:-1]
            else:
                f_cols = ['country'] + factor_style + factor_i  # [:-1]
            mat_x = pan[f_cols].values

            # 行业因子约束条件
            mv_indus = mv.values.T @ pan[factor_i].values
            pan[factor_i].sum()
            assert mv_indus.prod() != 0
            k = len(f_cols)  # 1 + len(factor_style) + len(factor_i) - 1
            mat_r = np.diag([1.] * k)[:, :-1]
            # pan[factor_i].sum()
            mat_r[-1:, -len(factor_i) + 1:] = - mv_indus[:-1] / mv_indus[-1]

            # WLS求解（Menchero & Lee, 2015)
            mat_omega = mat_r @ np.linalg.inv(mat_r.T @ mat_x.T @ mat_v @ mat_x @ mat_r) @ mat_r.T @ mat_x.T @ mat_v

            mat_y = pan['rtn_ctc'].values
            fv_1d = pd.DataFrame(mat_omega @ mat_y, index=f_cols, columns=[td])

            fval = pd.concat([fval, fv_1d.T])

            # 等效计算，条件处理后的WLS
            # mod = sm.WLS(mat_y, mat_x @ mat_r, weights=w_mv)
            # res = mod.fit()
            # fv = pd.DataFrame(mat_r @ res.params, index=f_cols, columns=[date])
            # res.summary()

            # 该日各纯因子构成
            pf_w = pd.DataFrame(mat_omega.T, index=pan.index, columns=f_cols)
            pf_w.to_hdf(omega_path, key=td.strftime('d%Y%m%d'), complevel=9)

        #
        fval.to_hdf(fval_path, key=f'y{year}')
        panel.to_hdf(panel_path, key=f'y{year}', complevel=9)


def combine(fval_path):
    """合并key=`y%Y`存在hdf中的纯因子收益率，注意存储名由开始、结束日期决定"""
    res_path = '/mnt/c/Users/Winst/Documents/data_local/BARRA/barra_fval_{}.csv'
    # year = 2012
    fval = pd.DataFrame()
    for year in range(2012, 2023):
        fv = pd.read_hdf(fval_path, key=f'y{year}')
        fval = pd.concat([fval, fv])

    fval.to_csv(res_path.format(fval.index[0].strftime('%Y%m%d') + '_' + fval.index[-1].strftime('%Y%m%d')))


if __name__ == '__main__':
    main()
