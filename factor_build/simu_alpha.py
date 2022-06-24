"""
(created by swmao on June 20th)
Class SimAlpha to generate toy alphas

"""
import yaml
import numpy as np
import pandas as pd
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.transformer import get_winsorize_sr


def cross_section_regress_adj(x: pd.DataFrame, y: pd.DataFrame, wl=1, intercept=True):
    """"""
    from tqdm import tqdm
    views = list(x.index)
    assets = list(x.columns)
    x_rescaled = pd.DataFrame(index=views, columns=assets)
    coefficient = pd.DataFrame(index=views, columns=['alpha', 'beta'])
    print('\nAdjust alpha with OLS ...')
    for i1 in tqdm(range(len(views))):
        i0 = max(0, i1 - wl)
        td0, td1 = views[i0], views[i1]
        x_train: pd.Series = x.loc[td0:td1].stack().dropna()
        y_train: pd.Series = y.loc[views].loc[td0:td1].stack().dropna()
        idx = x_train.index.intersection(y_train.index)
        x_train, y_train = x_train.loc[idx], y_train.loc[idx]
        xv, yv = np.matrix(x_train).reshape(-1, 1), np.matrix(y_train).reshape(-1, 1)
        if intercept:
            beta: float = (np.linalg.inv(xv.T @ xv) @ xv.T @ yv)[0, 0]
            alpha: float = yv.mean() - beta * xv.mean()
        else:
            alpha, beta = 0, yv.mean() / xv.mean()

        x_rescaled.loc[td1, :] = alpha + beta * x.loc[td1, :]
        coefficient.loc[td1, 'alpha'] = alpha
        coefficient.loc[td1, 'beta'] = beta

    return x_rescaled, coefficient


class SimAlpha(object):

    def __init__(self):
        self._fval = pd.DataFrame()
        self._name = None  # alpha name
        self._reg_coefficient = pd.DataFrame(columns=['alpha', 'beta'])

    def get_fval(self) -> pd.DataFrame:
        if len(self._fval) == 0:
            raise Exception
        return self._fval

    def show_fval(self):
        print(self._fval)

    def save_reg_coefficient(self, save_path: str):
        if len(self._reg_coefficient) == 0:
            raise Exception('blank regression coefficient')
        self._reg_coefficient.to_excel(save_path + self._name + '_reg_coefficient.xlsx')

    def save_fval(self, save_path):
        self.get_fval().to_csv(save_path + self._name + '.csv')
        print(f'Generate Alpha `{self._name}` saved in `{save_path}`')

    def adjust_fval(self, mtd='zscore', **kwargs):
        """
        adjust cross-section distribution of factor value
        :param mtd: str, method used
        :param kwargs: additional args config
        :return:
        """
        if mtd == 'uniform':
            self._fval = self._fval.rank(pct=True, axis=1) * 2 - 1
        elif mtd == 'zscore':
            self._fval = self._fval.apply(lambda s: get_winsorize_sr(s), axis=1)
            self._fval = self._fval.apply(lambda s: (s - s.mean()) / s.std(), axis=1)
        elif mtd == 'reverse':
            self._fval = self._fval.apply(lambda s: get_winsorize_sr(s), axis=1)
            self._fval = self._fval.apply(lambda s: (s.mean() - s) / s.std(), axis=1)
        elif mtd[:3] == 'reg':
            try:
                y = kwargs['y']
                wl = kwargs['wl']
                intercept = kwargs['intercept']
            except KeyError:
                raise Exception("**kwargs {'y': DataFrame, 'wl': int, 'intercept': bool} must be given for mtd 'reg'")
            self._fval, self._reg_coefficient = cross_section_regress_adj(x=self._fval, y=y, wl=wl, intercept=intercept)
        else:
            raise Exception(f'Alpha adjust method not in `zscore, uniform, reverse`')

        self._name += f"_{mtd}"
        if 'centre' in kwargs:
            self._fval += kwargs['centre']
            self._name += f"(m={kwargs['centre']})"
        if 'scale' in kwargs:
            self._fval *= kwargs['scale']
            self._name += f"(sd={kwargs['scale']})"
        if ('wl' in kwargs) and ('intercept' in kwargs):
            self._name += f"(wl={kwargs['wl']},i={'T' if kwargs['intercept'] else 'F'})"

        print(f'Adjust alpha `{self._name}`')

    def alpha_future_return(self, close_adj, dn=5, ms=(.0, .0), bd='2012-01-01', ed='2099-12-31'):
        """
        Future {dn} day return, with white noise mean-stddev {ms}
        :param close_adj: str
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

        close_adj = pd.read_csv(close_adj, index_col=0, parse_dates=True)
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


def simu_alpha(conf):
    begin_date_0 = '2015-12-01'
    begin_date = '2016-01-01'
    end_date = '2022-03-31'

    sim_alpha = SimAlpha()

    # # FRtn5D(0.0,3.0)_zscore_SD(0.0225)
    # sim_alpha.alpha_future_return(conf['closeAdj'], dn=5, ms=(0.0, 3.0), bd=begin_date, ed=end_date)
    # factor_apm_zscore_SD(0.0225)
    sim_alpha.load_factor(conf['factorscsv_path'], 'factor_apm', bd=begin_date, ed=end_date)

    # # Adjust Z-Score
    # sim_alpha.adjust_fval(mtd='zscore', centre=0, scale=2.25e-2)
    # Adjust Reg
    close_adj: pd.DataFrame = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True)
    close_adj = close_adj.loc[begin_date_0: end_date]
    d, wl = 1, 60
    rtn_next_view = close_adj.pct_change(periods=d).shift(-d).loc[begin_date:]
    sim_alpha.adjust_fval(mtd=f'reg{d}d', y=rtn_next_view, wl=wl, intercept=True)
    sim_alpha.save_reg_coefficient(save_path=conf['factorscsv_path'])

    sim_alpha.show_fval()
    sim_alpha.save_fval(save_path=conf['factorscsv_path'])


def main():
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    simu_alpha(conf)


if __name__ == '__main__':
    main()
