import sys


sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.factor_operator import *


class StkPool(object):

    def __init__(self, kind='A', w=None):
        self.kind = kind  # A, CSI300, CSI500, CSI800, CSI1000
        self.w_2d = w


class Portfolio(object):

    def __init__(self, w: pd.DataFrame):
        self.w_2d = w
        cols = ['Turnover', 'RtnNoCost', 'AbsoluteValueNoCost', 'ExcessValueNoCost',
                'RtnWithCost', 'AbsoluteValueWithCost', 'ExcessValueWithCost']
        self.result = pd.DataFrame(index=w.index, columns=cols)

    def cal_turnover_daily(self):
        self.result['Turnover'] = self.w_2d.diff().abs().sum(axis=1)

    def cal_return_daily(self, rtn_all: pd.DataFrame):
        self.result['RtnNoCost'] = (rtn_all.reindex_like(self.w_2d) * self.w_2d).sum(axis=1)
        self.result['RtnWithCost'] = self.result.RtnNoCost - self.result.Turnover

    def cal_absolute_result(self):
        pass

    def get_return_nc(self):
        return self.result.RtnNoCost.copy()

    def get_return_wc(self):
        return self.result.RtnWithCost.copy()

    def get_turnover(self):
        return self.result.Turnover.copy()

    def get_position(self):
        return self.w_2d.copy()

    def get_result(self):
        return self.result.copy()


class Signal(object):

    def __init__(self, data: pd.DataFrame, bd, ed):
        self.fv = data
        self.bd = bd
        self.ed = ed

    def shift_1d(self, T=1):
        self.fv = self.fv.shift(T).iloc[T:]

    def get_fv(self, bd=None, ed=None) -> pd.DataFrame:
        bd = self.bd if bd is None else bd
        ed = self.ed if ed is None else ed
        return self.fv.loc[bd: ed].copy()

    def keep_tradeable(self, mul: pd.DataFrame):
        self.fv = self.fv.reindex_like(mul.loc[self.bd: self.ed])
        self.fv = self.fv * mul.loc[self.bd: self.ed]
        self.fv = self.fv.astype(float)

    def neutralize_by(self, mtd, p_ind, p_mv):
        self.fv = factor_neutralization(self.fv, mtd, p_ind, p_mv)

    def get_long_short_group(self, ngroups: int) -> pd.DataFrame:
        return self.get_fv().rank(axis=1, pct=True).applymap(lambda x: x // (1 / ngroups))


class Strategy(object):

    def __init__(self):
        pass