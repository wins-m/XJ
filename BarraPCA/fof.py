"""
(created by swmao on April 15th)
用PCA选择指增

"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style("darkgrid")
# plt.rc("figure", figsize=(16, 10))
plt.rc("figure", figsize=(8, 4))
plt.rc("savefig", dpi=90)
# plt.rc("font", family="sans-serif")
# plt.rc("font", size=12)
plt.rc("font", size=10)

plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"


def cal_exc_rtn_sel(conf, dat_path, fbegin_date, fend_date, sample_size, kind_b, idx_rtn, pc_num):
    """计算超额收益选股部分"""
    pca_rtn_150 = conf['pca_rtn_150']
    tdays_d = pd.read_csv(conf['tdays_d'], header=None, index_col=0, parse_dates=True)
    tdays_d['tdays_d'] = tdays_d.index

    fund_val = pd.read_pickle(conf['refactor_net_value_5003'])  # 资产收益
    all_fund_rtn: pd.DataFrame = fund_val.pct_change()  # 累计复权净值，盘前9:00更新

    attr_funds = pd.read_excel(conf['dat_path_barra'] + f'fund_stat[{kind_b}].xlsx', dtype=object)
    fund_rtn = all_fund_rtn[attr_funds.main_code].astype(float).copy()
    fund_rtn[fund_rtn > .1] = .1
    fund_rtn[fund_rtn < -.1] = -.1
    dat = pd.concat([idx_rtn, fund_rtn], axis=1)

    excess_return_raw = pd.DataFrame()
    excess_return_selected = pd.DataFrame()
    td = tdays_d.loc[fbegin_date: fend_date, 'tdays_d'][sample_size]
    for td in tqdm(tdays_d.loc[fbegin_date: fend_date, 'tdays_d']):
        date_idx = tdays_d.loc[:td].iloc[-sample_size:].index
        if dat.index.intersection(date_idx).__len__() < sample_size:
            continue
        tgt = dat.loc[date_idx].dropna(axis=1, how='any')
        tgt = tgt.loc[:, tgt.std() > .001]
        src = pd.read_csv(pca_rtn_150.format(td.strftime('%Y%m%d')), index_col=0, parse_dates=True).loc[date_idx]
        src = src.iloc[:, :pc_num]

        x, y = src, tgt
        exposure = f = np.linalg.inv(x.T @ x) @ x.T @ y
        exposure.index = src.columns
        exposure_excess = exposure.iloc[:, 1:] - exposure['baseline'].values.reshape(-1, 1)  # 产品 超额暴露
        exposed_excess_rtn = pd.DataFrame(src @ exposure_excess).sum()  # 产品 超额暴露 的 超额收益，sample_size日求和
        excess_rtn = (tgt.iloc[:, 1:] - tgt['baseline'].values.reshape(-1, 1)).sum()  # 超过基准指数的超额收益
        selected_excess_rtn = excess_rtn - exposed_excess_rtn  # sample_size日 来自选股的超额
        excess_return_raw = pd.concat([excess_return_raw, excess_rtn.rename(td)], axis=1)
        excess_return_selected = pd.concat([excess_return_selected, selected_excess_rtn.rename(td)], axis=1)

    excess_return_raw.T.to_csv(dat_path + f'ExcessReturnTotal[{kind_b},{pc_num}].csv')
    excess_return_selected.T.to_csv(dat_path + f'ExcessReturnSelected[{kind_b},{pc_num}].csv')


# %%
def main():
    # %%
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    dat_path = conf['dat_path_pca']
    fbegin_date = '2012-01-01'
    fend_date = '2022-03-31'
    sample_size = 60

    idx_close = pd.read_csv(conf['idx_marketdata_close'], index_col=0, parse_dates=True, dtype=float)  # 股指
    idx_rtn_ctc = idx_close.pct_change().iloc[1:]
    kind_b, idx_rtn = '300', idx_rtn_ctc['000300.SH'].rename('baseline')  # CSI300
    # kind_b, idx_rtn = '500', idx_rtn_ctc['000905.SH'].rename('baseline')  # CSI500
    pc_num = 10

    # %%
    cal_exc_rtn_sel(conf, dat_path, fbegin_date, fend_date, sample_size, kind_b, idx_rtn, pc_num)


# %%
def func2(conf, dat_path, kind_b, idx_rtn_ctc, pc_num):
    # %%
    ers = pd.read_csv(dat_path + f'ExcessReturnSelected[{kind_b},{pc_num}].csv', index_col=0, parse_dates=True)
    ert = pd.read_csv(dat_path + f'ExcessReturnTotal[{kind_b},{pc_num}].csv', index_col=0, parse_dates=True)

    fund_net_val = pd.DataFrame(pd.read_hdf(conf['fund_net_value'], key='refactor_net_value'))[ers.columns]
    fund_return = fund_net_val.pct_change().iloc[1:]

    # %%
    fv = ers.shift(1).iloc[1:]
    fbegin_date = '2018-01-01'
    long_n = 5

    def fund_selected_cum_ret(fv, fbegin_date='2018-01-01', long_n=5):
        # %%
        fund_rtn = fund_return.loc[fv.index[0]: fv.index[-1]]
        baseline = fund_rtn.mean(axis=1).rename(f'baseline({fund_rtn.shape[1]})')
        exc_rtn_sel_ = fund_rtn[fv.rank(axis=1, ascending=False) <= long_n].mean(axis=1).rename(f'ERS_top{long_n}')
        idx_rtn = idx_rtn_ctc.loc[fv.index[0]: fv.index[-1], '000905.SH'].rename('Index')

        result_abs = pd.concat([exc_rtn_sel_, baseline, idx_rtn], axis=1)

        fund_rtn.count(axis=1).loc[fbegin_date:].plot(title='Asset Number')
        plt.show()

        result_abs = result_abs.dropna()
        result_abs.loc[fbegin_date:].cumsum().add(1).plot(linewidth=3, title=f'Absolute Result, {kind_b}, {pc_num}')
        plt.show()

        result_exc = result_abs.iloc[:, :-1] - result_abs.iloc[:, -1:].values.reshape(-1, 1)
        result_exc.loc[fbegin_date:].cumsum().add(1).plot(linewidth=3, title=f'Excess Result, {kind_b}, {pc_num}')
        plt.show()

    # %%
    fund_selected_cum_ret(ers.shift(1).iloc[1:])  # 日度选基
    fund_selected_cum_ret(ers.shift(5).iloc[5::5, :].reindex_like(ers).fillna(method='ffill'))  # 周度
    # fund_selected_cum_ret(ert.shift(1).iloc[1:])  # 日度选基
    # fund_selected_cum_ret(ert.shift(5).iloc[5::5, :].reindex_like(ers).fillna(method='ffill'))  # 周度


