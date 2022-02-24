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

    def __init__(self, data: pd.DataFrame, bd=None, ed=None, neu=None, ishow=False):
        self.fv = data
        self.bd = bd
        self.ed = ed
        self.__update_bd_ed__()
        self.neu_status = neu  # 中性化情况
        self.ic = None
        self.rank_ic = None
        self.ic_stat = None
        self.ic_decay = None
        self.ic_ir_cum = None

    def __update_bd_ed__(self):
        self.bd = self.fv.index[0] if self.bd is None else max(self.bd, self.fv.index[0])
        self.ed = self.fv.index[-1] if self.ed is None else min(self.ed, self.fv.index[-1])

    def shift_1d(self, d_shifted=1):
        self.fv = self.fv.shift(d_shifted).iloc[d_shifted:]
        self.__update_bd_ed__()

    def keep_tradeable(self, mul: pd.DataFrame):
        self.fv = self.fv.reindex_like(mul.loc[self.bd: self.ed])
        self.fv = self.fv * mul.loc[self.bd: self.ed]
        self.fv = self.fv.astype(float)

    def neutralize_by(self, mtd, p_ind, p_mv):
        self.fv = factor_neutralization(self.fv, mtd, p_ind, p_mv)
        self.neu_status = mtd  # 已做中性化

    def cal_ic(self, all_ret):
        if self.neu_status is None:
            print('Neutralize fval before IC calculation!')
        else:
            ret = all_ret.loc[self.bd: self.ed]
            self.ic = cal_ic(fv_l1=self.fv, ret=ret, ranked=False)
            self.rank_ic = cal_ic(fv_l1=self.fv, ret=ret, ranked=True)

    def cal_ic_statistics(self):
        if self.ic is None:
            raise AssertionError('Calculate IC before IC statistics!')
        ic_stat = pd.DataFrame()
        ic_stat['IC'] = cal_ic_stat(data=self.ic)
        ic_stat['Rank IC'] = cal_ic_stat(data=self.rank_ic)
        self.ic_stat = ic_stat.astype('float16')

    def cal_ic_decay(self, all_ret, lag):
        ret = all_ret.loc[self.bd: self.ed]
        self.ic_decay = cal_ic_decay(fval_neutralized=self.fv, ret=ret, maxlag=lag)

    def cal_ic_ir_cum(self):
        pass

    def plot_ic(self, ishow: bool, path_f: str = None):
        if path_f is not None:
            self.ic.plot.hist(figsize=(10, 5), bins=50, title='IC distribution')
            plt.savefig(path_f.format('IC.png'))
            if ishow:
                plt.show()
            else:
                plt.close()
            self.rank_ic.plot.hist(figsize=(10, 5), bins=50, title='IC distribution')
            plt.savefig(path_f.format('ICRank.png'))
            if ishow:
                plt.show()
            else:
                plt.close()

    def plot_ic_decay(self, ishow, path):
        self.ic_decay.plot.bar(figsize=(10, 5), title='IC Decay')
        plt.savefig(path)
        if ishow:
            plt.show()
        else:
            plt.close()

    def get_fv(self, bd=None, ed=None) -> pd.DataFrame:
        bd = self.bd if bd is None else bd
        ed = self.ed if ed is None else ed
        return self.fv.loc[bd: ed].copy()

    def get_fbegin(self):
        return self.bd

    def get_fend(self):
        return self.ed

    def get_ic_stat(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.ic_stat.to_csv(path)
        return self.ic_stat.copy()

    def get_ic_mean(self, ranked=True) -> float:
        if self.ic_stat is None:
            raise AssertionError('Calculate IC statistics before `get_ic_mean`')
        return self.ic_stat.loc['mean', ['IC', 'Rank IC'][ranked]]

    def get_ic_decay(self, path=None):
        if path is not None:
            self.ic_decay.to_csv(path)
        return self.ic


class Strategy(object):

    def __init__(self, sgn: Signal, ng: int, ishow=False):
        self.sgn = sgn  # factor value after preprocessing
        self.ng = ng  # number of long-short groups
        self.ishow = ishow
        self.ls_group = None
        self.ls_g_rtns = None

    def cal_long_short_group(self):
        self.ls_group = get_long_short_group(df=self.sgn.fv, ngroups=self.ng)

    def cal_group_returns(self, ret, idx_w):
        ret = ret.loc[self.sgn.get_fbegin(): self.sgn.get_fend()]  # ?冗余
        self.ls_g_rtns = cal_long_short_group_rtns(
            long_short_group=self.ls_group, ret=ret, idx_weight=idx_w, ngroups=self.ng)

    def plot_group_returns(self, path):
        plot_rtns_group(self.ls_g_rtns, self.ishow, path)

    def plot_group_returns_total(self, path):
        cal_total_ret_group(self.ls_g_rtns, self.ishow, path)

    def get_ls_group(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.ls_group.to_csv(path)
        return self.ls_group

    def get_group_returns(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.ls_g_rtns.to_csv(path)
        return self.ls_g_rtns
