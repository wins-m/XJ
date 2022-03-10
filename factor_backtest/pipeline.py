"""
(created by swmao on March 7th)
 - 因子原始值
 - 计算原始值在全A的IC,保存  TODO: warnings
 - 计算CSI500内原始值IC,保存
 - 在CSI500内去极值,标准化,中性化(i和iv),计算处理后IC,保存
 - 保留CSI500成分股,检查缺失情况,确定如何补缺


"""
import os
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")

from supporter.backtester import *


def panel_2d_val_mean(df, kw):
    df = df.copy().replace(False, np.nan)
    return f'Coverage({kw}):\t{100*(1 - np.mean(df.isna().values)):.2f} %'


class Factor(object):

    def __init__(self, conf, file, kind='ctc'):
        self.conf: dict = conf
        self.begin_date: str = '2013-01-01'  # conf['begin_date']
        self.end_date: str = '2021-12-31'  # conf['end_date']
        print(f'TimeRange:\t[{self.begin_date},{self.end_date}]')

        self.fval_raw: pd.DataFrame = pd.read_csv(conf['factorscsv_path'] + file, index_col=0, parse_dates=True, dtype=float)
        self.fname: str = file.replace('.csv', '')
        self.path_: str = conf['factorsres_path'] + f'SGN_{self.fname}' + '/{}'
        os.makedirs(self.path_.format(''), exist_ok=True)

        self.kind: str = kind  # oto/ctc

        self.stk_pool: pd.DataFrame = pd.read_csv(conf['idx_constituent'].format('CSI500'), index_col=0, parse_dates=True).loc[self.begin_date: self.end_date]
        self.in_stk_pool: pd.DataFrame = (~self.stk_pool.isna()).replace(False, np.nan)

        self.tradeable_status: pd.DataFrame = pd.DataFrame()
        self.all_ret: pd.DataFrame = pd.DataFrame()

    def evaluate_signal_ic(self):
        if len(self.all_ret) == 0:
            self.get_all_ret()
        for chn in ['raw', f'500+raw', '500+winso+std', '500+neu_i', '500+neu_iv']:
            self._evaluate_signal(channel=chn)

    def _evaluate_signal(self, channel='raw') -> Dict[str, pd.DataFrame]:
        print(f'Evaluate {self.fname}::{channel} IC...')
        sgn: Signal = Signal(data=self.fval_raw.loc[self.begin_date: self.end_date].copy())
        sgn.shift_1d(d_shifted=1)
        sgn.keep_tradeable(self.tradeable_status)

        if channel[:4] == '500+':
            _channel = channel[4:]
            sgn.keep_tradeable(self.in_stk_pool)
        else:
            _channel = channel

        if _channel == 'raw':
            sgn.neu_status = 'raw'
        elif _channel == 'winso+std':
            sgn.neutralize_by('n', '', '')
        elif _channel == 'neu_i':
            sgn.neutralize_by('i', self.conf['ind_citic'], '')
        elif _channel == 'neu_iv':
            sgn.neutralize_by('iv', self.conf['ind_citic'], self.conf['marketvalue'])
        else:
            raise AttributeError

        print(panel_2d_val_mean(sgn.get_fv(), kw=channel))
        sgn.get_fv().to_csv(self.path_.format(f'{self.fname}[{channel}].csv'))

        sgn.cal_ic(self.all_ret)
        ic_sr = sgn.get_ic(ranked=False, path=self.path_.format(f'SeriesIC[{channel}].csv'))
        ic_rank_sr = sgn.get_ic(ranked=True, path=self.path_.format(f'SeriesRankIC[{channel}].csv'))
        sgn.cal_ic_statistics()
        ic_stat = sgn.get_ic_stat(path=self.path_.format(f'PanelStatIC[{channel}].csv'))
        sgn.cal_ic_decay(all_ret=self.all_ret, lag=10)
        ic_decay = sgn.get_ic_decay(path=self.path_.format(f'PanelDecayIC[{channel}].csv'))

        return {'fv': sgn.get_fv(), 'IC': ic_sr, 'Rank IC': ic_rank_sr, 'IC Stat': ic_stat, 'IC Decay': ic_decay}

    def get_tradeable_status(self):
        df = self._get_tradeable(k='ipo')
        for key in ['ipo60', 'suspend', 'updown']:
            print(f'Sift tradeable stocks via `{key}`')
            df &= self._get_tradeable(k=key)
        self.tradeable_status = df
        print(panel_2d_val_mean(df, kw='Tradeable'))

    def _get_tradeable(self, k: str) -> pd.DataFrame:
        df = pd.DataFrame(pd.read_hdf(self.conf['a_list_tradeable'], key=k, indx_col=0, parse_dates=True, dtype=bool))
        if self.kind == 'ctc':
            df = df.shift(1).iloc[1:]
        return df.loc[self.begin_date: self.end_date]

    def get_all_ret(self):
        if self.kind == 'ctc':
            price = pd.read_csv(self.conf['closeAdj'], index_col=0, parse_dates=True, dtype=float)
        elif self.kind == 'oto':
            price = pd.read_csv(self.conf['openAdj'], index_col=0, parse_dates=True, dtype=float)
            price = price.shift(-1)  # Return: long T+1 Open short T+2 Open for T0 Signal
        else:
            raise AttributeError
        rtn = price.pct_change().loc[self.begin_date: self.end_date]
        if len(self.tradeable_status) == 0:
            self.get_tradeable_status()
        rtn *= self.tradeable_status
        self.all_ret = rtn.astype(float)
        print(panel_2d_val_mean(rtn, kw=f'Return-{self.kind}'))


# %%
def main():
    # %%
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    # %%
    # filename = 'alpha_101.csv'
    for filename in os.listdir(conf['factorscsv_path']):
        if filename[:6] == 'alpha_' and filename[6:9] > '099':
            print(filename)
            fct = Factor(conf=conf, file=filename, kind='ctc')
            fct.evaluate_signal_ic()


# %%
if __name__ == '__main__':
    # %%
    main()
