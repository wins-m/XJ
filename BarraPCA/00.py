"""
(created on June 16th)
查看个股日收益分布情况

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(9, 5))
# plt.rc("figure", figsize=(18, 10))
plt.rc("font", size=12)
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rc("savefig", dpi=90)
# plt.rc("font", family="sans-serif")
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"

from supporter.request import get_hold_return


def figure_cross_section_sd(df, msg):
    y = df.std(axis=1).rename('SD')
    y.plot(title=f'{msg} std dev, $\mu_\sigma={y.mean()*100:.3f}$%')
    plt.tight_layout()
    plt.show()


def figure_cross_section_mean(df, msg):
    y = df.mean(axis=1).rename('Mean')
    y.plot(title=f'{msg} mean, $\mu_\mu={y.mean()*100:.3f}$%')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    ret_kind = 'ctc'
    bd0 = '2015-12-01'
    bd = '2016-01-01'
    ed = '2021-12-31'
    stk_pool = 'NA'

    rtn = get_hold_return(conf, ret_kind, bd0, ed, stk_pool)
    rtn = rtn.dropna(how='all', axis=1).loc[bd:ed]
    # np.matrix(rtn > 0.11).sum()
    # np.matrix(rtn < -0.11).sum()

    figure_cross_section_sd(rtn, msg='ctc rtn')
    figure_cross_section_mean(rtn, msg='ctc rtn')
    rtn1 = rtn.copy()
    rtn1[rtn > 0.11] = 0.11
    rtn1[rtn < -0.11] = -0.11
    figure_cross_section_sd(rtn1, msg='ctc rtn (shrink updown)')
    figure_cross_section_mean(rtn1, msg='ctc rtn (shrink updown)')
    rtn2 = rtn.copy()
    rtn2[rtn > 0.11] = np.nan
    rtn2[rtn < -0.11] = np.nan
    figure_cross_section_sd(rtn2, msg='ctc rtn (no updown)')
    figure_cross_section_mean(rtn2, msg='ctc rtn (no updown)')


    np.nanmean(np.matrix(rtn))
    np.nanstd(np.matrix(rtn))

    alpha_path = '/mnt/c/Users/Winst/Documents/factors_csv/factor_FRtn5D(0.0,3.0).csv'
    alpha = pd.read_csv(alpha_path, index_col=0, parse_dates=True)
    np.nanmean(np.matrix(alpha))
    np.nanstd(np.matrix(alpha))
