import numpy as np
import pandas as pd
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn

seaborn.set_style("darkgrid")
plt.rc("figure", figsize=(10, 4))
plt.rc("font", size=10)
# plt.rc("figure", figsize=(20, 8))
plt.rc("savefig", dpi=90)
# plt.rc("font", family="sans-serif")
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"


def _get_signal(panel, idx, dur):
    signal = panel.pivot(index='tradingdate', columns='stockcode', values=idx)
    if dur > 1:
        signal = signal.replace(0, np.nan).fillna(method='ffill', limit=dur - 1)
    signal = signal.replace(np.nan, 0)
    return signal


def _get_weight(signal):
    return signal.apply(lambda s: s / s.abs().sum(), axis=1)


def factor_info(csv_path, fname, signal, descrip=''):
    signal.to_csv(csv_path + fname + '.csv')
    irow = {'IF_TEST': 1,
            'F_NAME': fname,
            'F_BEGIN': str(signal.index[0].date()),
            'F_END': str(signal.index[-1].date()),
            'UPDATE': datetime.today().strftime('%Y-%m-%d'),
            'DESCRIP': descrip}
    return irow


def write_f_info(excel_path, info_list):
    df = pd.read_excel(excel_path)
    df['IF_TEST'] = 0
    df = df.append(info_list, ignore_index=True)
    # 同名因子只保留UPDATE日期最新的
    mask = df.sort_values(['UPDATE', 'IF_TEST'], ascending=False)[['F_NAME']].drop_duplicates().index
    df = df.loc[mask].sort_index()
    df.astype('str').to_excel(excel_path, encoding='gbk', index=None)


def drop_return_t1t2(excel_path, csv_path, event_panel, idx, threshold, ishow=False, side='low'):
    fname = f'first_report_{side}{idx}100({threshold})'
    print(fname)

    panel = event_panel[event_panel.Tradeable].copy()
    panel['baseline'] = 1

    signal = _get_signal(panel, 'baseline', dur=3)
    weight = _get_weight(signal=signal)

    if 'low' in side:
        threshold = -threshold if threshold > 0 else threshold  # 阈值需要<=0
        panel['negR1'] = panel[f'{idx}1'] <= threshold
        panel['negR2'] = panel[f'{idx}2'] <= threshold
        mask_low_ar1_lag1 = (_get_signal(panel, 'negR1', 1).shift(1) == True)
        mask_low_ar1_lag2 = (_get_signal(panel, 'negR1', 1).shift(2) == True)
        mask_low_ar2_lag2 = (_get_signal(panel, 'negR2', 1).shift(2) == True)
        weight[mask_low_ar1_lag1] = 0
        weight[mask_low_ar1_lag2] = 0
        weight[mask_low_ar2_lag2] = 0

    if 'high' in side:
        threshold = -threshold if threshold < 0 else threshold  # 阈值需要>=0
        panel['posR1'] = panel[f'{idx}1'] >= threshold
        panel['posR2'] = panel[f'{idx}2'] >= threshold
        mask_low_ar1_lag1 = (_get_signal(panel, 'posR1', 1).shift(1) == True)
        mask_low_ar1_lag2 = (_get_signal(panel, 'posR1', 1).shift(2) == True)
        mask_low_ar2_lag2 = (_get_signal(panel, 'posR2', 1).shift(2) == True)
        weight[mask_low_ar1_lag1] = 0
        weight[mask_low_ar1_lag2] = 0
        weight[mask_low_ar2_lag2] = 0

    if ishow:
        tmp = weight.sum(axis=1)
        for rlen in [1, 5, 20, 60]:
            tmp.rolling(rlen).mean().plot(label=f'{rlen}', alpha=.4 + rlen / 100)
        plt.legend()
        plt.title('Position')
        plt.show()

    write_f_info(excel_path=excel_path,
                 info_list=factor_info(
                     csv_path=csv_path,
                     fname=fname,
                     signal=weight,
                     descrip=f'阈值{threshold}去除{side}_{idx}1, {side}_{idx}2'))


def drop_return_t2(excel_path, csv_path, event_panel, idx, threshold, ishow=False, side='low'):
    fname = f'first_report_{side}{idx}110({threshold})'
    print(fname)

    panel = event_panel[event_panel.Tradeable].copy()
    panel['baseline'] = 1

    signal = _get_signal(panel, 'baseline', dur=3)
    weight = _get_weight(signal=signal)

    if 'low' in side:
        threshold = -threshold if threshold > 0 else threshold  # 阈值需要<=0
        panel['negR2'] = panel[f'{idx}2'] <= threshold
        mask_low_ar2_lag2 = (_get_signal(panel, 'negR2', 1).shift(2) == True)
        weight[mask_low_ar2_lag2] = 0

    if 'high' in side:
        threshold = -threshold if threshold < 0 else threshold  # 阈值需要>=0
        panel['posR2'] = panel[f'{idx}2'] >= threshold
        mask_low_ar2_lag2 = (_get_signal(panel, 'posR2', 1).shift(2) == True)
        weight[mask_low_ar2_lag2] = 0

    if ishow:
        tmp = weight.sum(axis=1)
        for rlen in [1, 5, 20, 60]:
            tmp.rolling(rlen).mean().plot(label=f'{rlen}', alpha=.4 + rlen / 100)
        plt.legend()
        plt.title('Position')
        plt.show()

    write_f_info(excel_path=excel_path,
                 info_list=factor_info(
                     csv_path=csv_path,
                     fname=fname,
                     signal=weight,
                     descrip=f'阈值{threshold}去除{side}_{idx}2'))


def drop_return_t2_g(excel_path, csv_path, event_panel, idx, ishow=False, side='L'):
    fname = f'first_report_{side}_{idx}110'
    print(fname)

    panel = event_panel[event_panel.Tradeable].copy()
    panel['baseline'] = 1

    signal = _get_signal(panel, 'baseline', dur=3)
    weight = _get_weight(signal=signal)

    if side == 'L':
        mask_low_ar2_lag2 = (_get_signal(panel, f'L_{idx}2', 1).shift(2) == 1)
        weight[mask_low_ar2_lag2] = 0

    if side == 'H':
        mask_low_ar2_lag2 = (_get_signal(panel, f'H_{idx}2', 1).shift(2) ==  1)
        weight[mask_low_ar2_lag2] = 0

    if ishow:
        tmp = weight.sum(axis=1)
        for rlen in [1, 5, 20, 60]:
            tmp.rolling(rlen).mean().plot(label=f'{rlen}', alpha=.4 + rlen / 100)
        plt.legend()
        plt.title('Position')
        plt.show()

    write_f_info(excel_path=excel_path,
                 info_list=factor_info(
                     csv_path=csv_path,
                     fname=fname,
                     signal=weight,
                     descrip=f'日内二分去除{side}_{idx}2==1'))
