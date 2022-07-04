"""
(created by swmao on June 29th)
Remote:
1. get_barra.py
2. cal_factor_return.py
3. cov_adjust.py
4. optimize.py
5. opt_res_ana.py

"""


def get_ind_citic_all_tradingdate(conf: dict, begin_date, end_date) -> pd.DataFrame:
    """Get citic industry label, index is all tradingdate from begin_date to end_date"""
    bd, begin_date, ed = pd.to_datetime(begin_date) - timedelta(60), pd.to_datetime(begin_date), pd.to_datetime(end_date)
    ind_citic = pd.read_csv(conf['ind_citic'], index_col=0, parse_dates=True, dtype=object).loc[bd:ed]
    tdays_d = pd.read_csv(conf['tdays_d'], header=None, index_col=0, parse_dates=True).loc[bd:ed]
    tdays_d = tdays_d.reset_index().rename(columns={0:'tradingdate'})
    ind_citic = ind_citic.reset_index().merge(tdays_d, on='tradingdate', how='right')
    ind_citic = ind_citic.set_index('tradingdate').fillna(method='ffill').loc[begin_date:]

    return ind_citic


def cal_fac_ret(conf, data_pat, cache_path, panel_path, fval_path, omega_path):
    """计算纯因子收益率，缓存过程文件，分年循环"""
    industry: pd.DataFrame = get_ind_citic_all_tradingdate(conf, '2012-01-01', '2099-12-31')

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

    # TODO: stat of return
    rtn = rtn_close_adj
    rtn[rtn > 0.11] = np.nan
    rtn[rtn < -0.11] = np.nan

    cross_section_sd = rtn.loc['2013-01-01': '2022-03-31'].std(axis=1)
    cross_section_sd.plot()
    plt.title('cross-section standard deviation of daily return')
    plt.tight_layout()
    plt.show()

    cross_section_sd.describe()

    # year = 2015
    for year in range(2022, 2011, -1):
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

        # cross-section (daily) WLS
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

            # 该日各纯因子构成
            pf_w = pd.DataFrame(mat_omega.T, index=pan.index, columns=f_cols)
            pf_w.to_hdf(omega_path, key=td.strftime('d%Y%m%d'), complevel=9)

        fval.to_hdf(fval_path, key=f'y{year}')
        panel.to_hdf(panel_path, key=f'y{year}', complevel=9)


