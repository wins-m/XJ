"""
(created by swmao on June 20th)
Class SimuAlpha to generate toy alphas

"""
import yaml
import numpy as np
import pandas as pd
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.transformer import get_winsorize_sr


class SimuAlpha(object):

    def __init__(self):
        self._fval = pd.DataFrame()
        self._name = None  # alpha name

    def get_fval(self) -> pd.DataFrame:
        if len(self._fval) == 0:
            raise Exception
        return self._fval

    def show_fval(self):
        print(self._fval)

    def save_fval(self, save_path):
        self.get_fval().to_csv(save_path + self._name + '.csv')
        print(f'Generate Alpha `{self._name}` saved in `{save_path}`')

    def adjust_fval(self, mtd='zscore', centre=0.0, scale=1.0):
        """"""
        if mtd == 'uniform':
            self._fval = self._fval.rank(pct=True, axis=1) * 2 - 1
        elif mtd == 'zscore':
            self._fval = self._fval.apply(lambda s: get_winsorize_sr(s), axis=1)
            self._fval = self._fval.apply(lambda s: (s - s.mean()) / s.std(), axis=1)
        elif mtd == 'reverse':
            self._fval = self._fval.apply(lambda s: get_winsorize_sr(s), axis=1)
            self._fval = self._fval.apply(lambda s: (s.mean() - s) / s.std(), axis=1)
        else:
            raise Exception(f'Alpha adjust method not in `zscore, uniform, reverse`')
        self._name += f"_{mtd}"

        if centre != 0:
            self._fval += centre
            self._name += f"_M({centre})"
        if scale != 1:
            self._fval *= scale
            self._name += f"_SD({scale})"

        print(f'Adjust alpha `{self._name}`')

    def alpha_future_return(self, closeAdj, dn=5, ms=(.0, .0), bd='2012-01-01', ed='2099-12-31'):
        """
        Future {dn} day return, with white noise mean-stddev {ms}
        :param closeAdj: str
            csv path of adjusted close price
        :param dn: int, optional
            1 if tomorrow return (long today's close, short tomorrow's close)
        :param ms: Tuple[float, float]
            (mean, std) of white noise
        :param bd: str
            fval begin date
        :param ed: str
            fval end date

        """
        self._name = f'FRtn{dn}D({ms[0]},{ms[1]})'

        close_adj = pd.read_csv(closeAdj, index_col=0, parse_dates=True)
        res = close_adj.pct_change(dn).shift(-dn).loc[bd: ed]
        if (ms[0] == 0) and (ms[1] == 0):
            res = res.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        else:
            res = res.apply(
                lambda x: (x - x.mean()) / x.std() + np.random.normal(loc=ms[0], scale=ms[1], size=len(x)),
                axis=1)
            print("future return + white noise", end='\n\t')
            print(f"cross-section mu={res.mean(axis=1).mean():.3f} sigma={res.std(axis=1).mean():.3f}")

        self._fval = res
        print(f'Simulate alpha `{self._name}`')

    def load_factor(self, csv_path, fname, bd='2012-01-01', ed='2099-12-31'):
        df = pd.read_csv(f"{csv_path}{fname}.csv", index_col=0, parse_dates=True)
        df = df.loc[bd:ed]
        self._fval = df
        self._name = fname


def main():
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    simu_alpha = SimuAlpha()
    # simu_alpha.alpha_future_return(conf['closeAdj'], dn=5, ms=(0.0, 3.0), bd='2016-01-01', ed='2022-03-31')  # FRtn5D(0.0,3.0)_zscore_SD(0.0225)
    simu_alpha.load_factor(conf['factorscsv_path'], 'factor_apm', bd='2016-02-01', ed='2022-03-31')  # factor_apm_zscore_SD(0.0225)
    simu_alpha.adjust_fval(mtd='zscore', centre=0, scale=2.25e-2)
    simu_alpha.show_fval()
    simu_alpha.save_fval(save_path=conf['factorscsv_path'])


if __name__ == '__main__':
    main()
