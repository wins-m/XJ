"""(created by swmao on Feb 16th)"""

import os, yaml
import pandas as pd
import warnings
warnings.simplefilter("ignore")
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style("darkgrid")
plt.rc("figure", figsize=(16, 8))
plt.rc("font", size=12)
plt.rc("savefig", dpi=90)
# plt.rc("font", family="sans-serif")
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"


# %%
def max_drawdown_compare():
    pass


def half_year_stat_compare(idx: str, res_path: str, filenames: dict):
    panel = pd.DataFrame()
    for k, v in filenames.items():
        df = pd.read_csv(res_path + v + '/ResLongNC.csv', index_col=0)
        panel = pd.concat([panel, df[idx].rename(k)], axis=1)
    panel.plot.bar(title=f'Half Year {idx}')
    plt.show()


def plot_holding_status(csv_path, filenames):
    for k, v in filenames.items():
        if 'baseline' in k:
            continue
        df = pd.read_csv(csv_path + k + '.csv', index_col=0, parse_dates=True)
        df = df.sum(axis=1)

        for wlen in [1, 5, 20, 60]:
            df.rolling(wlen).mean().plot(alpha=.4 + wlen / 100, label=wlen)
        plt.legend()
        plt.title(f'{k}, mean: {df.mean():.3f}')
        plt.show()


def net_value_compare(conf, filenames=None):
    if filenames is None:
        filenames: dict = {
            '80-100': 'first_report_i5_R100CAR_3_0up_dur3_n_NA_ew_1g_ctc_1hd(0216_092825)',
            '60-100': 'first_report_i5_R80CAR_3_0up_dur3_n_NA_ew_1g_ctc_1hd(0216_092825)',
            '40-100': 'first_report_i5_R60CAR_3_0up_dur3_n_NA_ew_1g_ctc_1hd(0216_092825)',
            '20-100': 'first_report_i5_R40CAR_3_0up_dur3_n_NA_ew_1g_ctc_1hd(0216_092825)',
            '0-100': 'first_report_i5_R20CAR_3_0up_dur3_n_NA_ew_1g_ctc_1hd(0216_092825)',
        }

    res_path = conf['factorsres_path']
    ana_path = conf['resanalysis_path']

    panel_nc = pd.DataFrame()
    panel_wc = pd.DataFrame()
    for kw, filename in filenames.items():
        panel_long = pd.read_csv(res_path + filename + '/PanelLong.csv', index_col=0, parse_dates=True)
        panel_long.head(2)
        panel_nc[kw] = panel_long['Wealth(cumsum)']
        panel_wc[kw] = panel_long['Wealth_wc(cumsum)']

    for suffix, df in zip(['No Cost', 'With Cost'], [panel_nc, panel_wc]):
        df.plot(title='Wealth(cumsum) ' + suffix, linewidth=3)
        plt.show()
        if 'baseline' in df.columns:
            df1 = df - df[['baseline']].values.reshape(-1, 1)
            df1.plot(title='Excess Wealth(cumsum) ' + suffix, linewidth=3)
            plt.show()


# %%
if __name__ == '__main__':
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    savefig = True

    net_value_compare(conf)
