"""
(created by swmao on March 7th)
- 因子原始值
- (depreciated)计算原始值在全A的Rank IC,CSI500内分别经过winso&std,neu_i,neu_iv的IC及IC Decay
- 仅保留CSI500, rescale into (0, 1] and save

"""
import os
import sys
# from multiprocessing import Pool

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")

from supporter.backtester import *


def panel_2d_val_mean(df, kw):
    df = df.copy().replace(False, np.nan)
    return f'Coverage({kw}):\t{100*(1 - np.mean(df.isna().values)):.2f} %'


class Factor(object):

    def __init__(self, conf, csv_file, kind='ctc', stk_pool='CSI500'):
        self.conf: dict = conf
        self.fname: str = csv_file.replace('.csv', '')
        print(f'\n{self.fname}')
        self.kind: str = kind  # oto/ctc
        self.stk_pool: str = stk_pool
        self.fval_raw: pd.DataFrame = pd.read_csv(conf['factorscsv_path'] + csv_file, index_col=0, parse_dates=True, dtype=float)
        self.begin_date: str = max(conf['begin_date'], self.fval_raw.index[0].strftime('%Y-%m-%d'))
        self.end_date: str = min(conf['end_date'], self.fval_raw.index[-1].strftime('%Y-%m_%d'))
        print(f'TimeRange:\t[{self.begin_date},{self.end_date}]')
        # from supporter.suffix import get_time_suffix
        # self.path_: str = f'{conf["factorsres_path"]}SGN_{self.fname}[{get_time_suffix()}]' + '/{}'
        self.path_: str = f'{conf["factorsres_path"]}SGN_{self.fname}' + '/{}'
        os.makedirs(self.path_.format(''), exist_ok=True)
        print(f'Result will be saved in {self.path_}')
        self.tradeable_status: pd.DataFrame = pd.DataFrame()
        self.all_ret: pd.DataFrame = pd.DataFrame()
        self.fval_in_pool: pd.DataFrame = pd.DataFrame()
        self.fval_ranked: pd.DataFrame = pd.DataFrame()

    def dump_config(self):
        """Add configuration file to save path."""
        pass

    def get_fval_ranked(self, save_table=False) -> pd.DataFrame:
        if len(self.fval_ranked) == 0:
            self._get_fval_ranked()
        if save_table:
            # save_path = self.path_.format(f'[{self.stk_pool}ranked]{self.fname}.csv')
            save_path = f'{self.conf["factorscsv_path"]}[{self.stk_pool}ranked]{self.fname}.csv'
            self.fval_ranked.to_csv(save_path)
            print(f'Rescaled factor value saved in {save_path}')
        return self.fval_ranked

    def evaluate_signal_ic(self):
        """计算不同预处理方式后的IC保存文件"""
        def _evaluate_signal(channel='raw') -> Dict[str, pd.DataFrame]:
            """原始因子值以不同channel预处理计算IC保存IC结果"""
            print(f'Evaluate {self.fname}::{channel} IC...')
            sgn: Signal = Signal(data=self.fval_raw.loc[self.begin_date: self.end_date].copy())
            sgn.shift_1d(d_shifted=1)
            sgn.keep_tradeable(self.tradeable_status)

            if channel[:4] == '500+':
                stk_pool: pd.DataFrame = pd.read_csv(self.conf['idx_constituent'].format('CSI500'), index_col=0,
                                                     parse_dates=True).loc[self.begin_date: self.end_date]
                in_stk_pool: pd.DataFrame = (~stk_pool.isna()).replace(False, np.nan)
                _channel = channel[4:]
                sgn.keep_tradeable(in_stk_pool)
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

        if len(self.all_ret) == 0:
            self._get_all_ret()

        for chn in ['raw', f'500+raw', '500+winso+std', '500+neu_i', '500+neu_iv']:
            _evaluate_signal(channel=chn)

    def conduct_pipeline(self):
        """依次执行"""
        self._get_tradeable_status()
        self._get_all_ret()
        self._get_fval_in_pool()
        self._get_fval_ranked()

    def _get_fval_in_pool(self):
        """截面因子值仅保留stk_pool成分股其余留空"""
        src_path = self.conf['idx_constituent'].format(self.stk_pool)
        stk_in_pool = ~pd.read_csv(src_path, index_col=0, parse_dates=True).loc[self.begin_date: self.end_date].isna()
        fv_in_pool = self.fval_raw.reindex_like(stk_in_pool) * stk_in_pool.replace(False, np.nan)
        self.fval_in_pool = fv_in_pool.dropna(how='all', axis=1)
        print(f'{self.stk_pool}: fval_raw{self.fval_raw.shape} -> fval_in_pool{self.fval_in_pool.shape}')

    def _get_fval_ranked(self):
        """Rescale cross-sectional fval into (0, 1] rank order."""
        if len(self.fval_in_pool) == 0:
            self._get_fval_in_pool()
        self.fval_ranked = self.fval_in_pool.rank(axis=1, pct=True)

    def _get_tradeable_status(self):
        """加载可交易情况self.tradeable_status"""

        def _get_tradeable(k: str) -> pd.DataFrame:
            """加载一层(key=k)可交易情况"""
            _df = pd.DataFrame(pd.read_hdf(self.conf['a_list_tradeable'], key=k, indx_col=-1, parse_dates=True, dtype=bool))
            if self.kind == 'ctc':
                _df = _df.shift(0).iloc[1:]
            return _df.loc[self.begin_date: self.end_date]

        df = _get_tradeable(k='ipo')
        for key in ['ipo60', 'suspend', 'updown']:
            print(f'Sift tradeable stocks via `{key}`')
            df &= _get_tradeable(k=key)
        self.tradeable_status = df
        print(panel_2d_val_mean(df, kw='Tradeable'))

    def _get_all_ret(self):
        """Load all tradeable (attainable) daily returns to self.all_ret."""

        if self.kind == 'ctc':
            price = pd.read_csv(self.conf['closeAdj'], index_col=0, parse_dates=True, dtype=float)
        elif self.kind == 'oto':
            price = pd.read_csv(self.conf['openAdj'], index_col=0, parse_dates=True, dtype=float)
            price = price.shift(-1)  # Return: long T+1 Open short T+2 Open for T0 Signal
        else:
            raise AttributeError

        rtn = price.pct_change().loc[self.begin_date: self.end_date]

        if len(self.tradeable_status) == 0:
            self._get_tradeable_status()

        rtn *= self.tradeable_status
        self.all_ret = rtn.astype(float)

        print(panel_2d_val_mean(rtn, kw=f'Return-{self.kind}'))


def main():
    # %%
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    csv_file, kind = 'alpha_086.csv', 'ctc'
    for csv_file in os.listdir(conf['factorscsv_path']):
        if csv_file[:6] == 'alpha_':
            fct = self = Factor(conf=conf, csv_file=csv_file, kind='ctc')
            fct.get_fval_ranked(save_table=True)
            # fct.evaluate_signal_ic()


# %%
# if __name__ == '__main__':
#     main()
