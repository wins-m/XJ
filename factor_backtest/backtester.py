import sys
import time
from datetime import timedelta

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.factor_operator import *


def clip_backtest_conf(conf: dict):
    res = {
        'csv_path': conf['factorscsv_path'],
        'res_path': conf['factorsres_path'],
        'idx_constituent': conf['idx_constituent'],
        'tradeable_path': conf['a_list_tradeable'],
        'ind_citic_path': conf['ind_citic'],
        'marketvalue_path': conf['marketvalue'],
        'close_path': conf['closeAdj'],
        'open_path': conf['openAdj'],
        'test_mode': str(conf['test_mode']),
        'exclude_tradeable': conf['exclude_tradeable'],
        'neu_mtd': conf['neu_mtd'],
        'stk_pool': conf['stk_pool'],
        'stk_w': conf['stk_w'],
        'return_kind': conf['return_kind'],
        'ngroups': conf['ngroups'],
        'holddays': conf['holddays'],
        'cost_rate': float(conf['tc']),
        'begin_date': conf['begin_date'],
        'end_date': conf['end_date'],
        'save_tables': conf['save_tables'],
        'save_plots': conf['save_plots'],
        'ishow': conf['ishow'],
        'all_factornames': pd.read_excel(conf['factors_tested'], index_col=0).loc[1:1].iloc[:, 0].to_list(),
        'save_suffix': conf['save_suffix'] if conf['save_suffix'] != '' else time.strftime("%m%d_%H%M%S",
                                                                                           time.localtime()),
        'begin_date_nd60': (pd.to_datetime(conf['begin_date']) - timedelta(60)).strftime('%Y-%m-%d')
    }
    # res['fbegin_end'] = df[['F_NAME', 'F_BEGIN', 'F_END']].set_index('F_NAME').apply(lambda s: (s.iloc[0],
    # s.iloc[1]), axis=1).to_dict()

    return res


class StkPool(object):

    def __init__(self, kind='A', w=None):
        self.kind = kind  # A, CSI300, CSI500, CSI800, CSI1000
        self.w_2d = w


class Portfolio(object):

    def __init__(self, w: pd.DataFrame = None):
        self.w_2d = w
        self.panel = None
        self.cost_rate = None

    def cal_panel_result(self, cr, ret):
        self.cost_rate = cr
        self.panel = portfolio_statistics_from_weight(weight=self.w_2d, cost_rate=cr, all_ret=ret)

    def plot_turnover(self, ishow, path):
        if self.panel is None:
            raise AssertionError('Calculate panel result before plot turnover!')
        self.panel['Turnover'].plot(figsize=(10, 5), grid=True, title='Turnover')
        plt.savefig(path)
        if ishow:
            plt.show()
        else:
            plt.close()

    def get_position_weight(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.w_2d.to_csv(path)
        return self.w_2d.copy()

    def get_panel(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.panel.to_csv(path)
        return self.panel.copy()


class Signal(object):

    def __init__(self, data: pd.DataFrame, bd=None, ed=None, neu=None):
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

    def plot_ic(self, ishow: bool, path_f: str):
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

    def __init__(self, sgn: Signal, ng: int):
        self.sgn = sgn  # factor value after preprocessing
        self.ng = ng  # number of long-short groups
        self.ls_group = None
        self.ls_g_rtns = None
        self.holddays = None
        self.portfolio_l: Portfolio = Portfolio()
        self.portfolio_s: Portfolio = Portfolio()
        self.portfolio_ls: Portfolio = Portfolio()
        self.portfolio_baseline: Portfolio = Portfolio()
        self.all_panels = {}

    def cal_long_short_group(self):
        self.ls_group = get_long_short_group(df=self.sgn.fv, ngroups=self.ng)

    def cal_group_returns(self, ret, idx_w):
        ret = ret.loc[self.sgn.get_fbegin(): self.sgn.get_fend()]  # ?冗余
        self.ls_g_rtns = cal_long_short_group_rtns(
            long_short_group=self.ls_group, ret=ret, idx_weight=idx_w, ngroups=self.ng)

    def cal_long_short_panels(self, idx_w, hd, rvs, cr, ret):
        self.portfolio_l = self.get_holding_position(idx_w=idx_w, hd=hd, rvs=rvs, kind='long')
        self.portfolio_s = self.get_holding_position(idx_w=idx_w, hd=hd, rvs=rvs, kind='short')
        self.portfolio_ls = self.get_holding_position(idx_w=idx_w, hd=hd, rvs=rvs, kind='long_short')
        self.portfolio_baseline = self.get_holding_position(idx_w=idx_w, hd=hd, rvs=rvs, kind='baseline')
        self.portfolio_l.cal_panel_result(cr=cr, ret=ret)
        self.portfolio_s.cal_panel_result(cr=cr, ret=ret)
        self.portfolio_ls.cal_panel_result(cr=cr, ret=ret)
        self.portfolio_baseline.cal_panel_result(cr=cr, ret=ret)

    def plot_group_returns(self, ishow, path):
        plot_rtns_group(self.ls_g_rtns, ishow, path)

    def plot_group_returns_total(self, ishow, path):
        cal_total_ret_group(self.ls_g_rtns, ishow, path)

    def plot_turnover(self, ishow, path):
        long_short_turnover = pd.concat([df['Turnover'].rename(k) for k, df in self.all_panels.items()], axis=1)
        long_short_turnover[['long', 'short']].plot(figsize=(10, 5), grid=True, title='Turnover')
        plt.savefig(path)
        if ishow:
            plt.show()
        else:
            plt.close()

    def get_ls_group(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.ls_group.to_csv(path)
        return self.ls_group

    def get_group_returns(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.ls_g_rtns.to_csv(path)
        return self.ls_g_rtns

    def get_holding_position(self, idx_w, hd=1, rvs=False, kind='long') -> Portfolio:
        if (kind == 'long' and not rvs) or (kind == 'short' and rvs):
            _position = (self.ls_group == self.ng if self.ng == 1 else self.ng - 1)
        elif (kind == 'short' and not rvs) or (kind == 'long' and rvs):
            _position = (self.ls_group == 0)
        elif kind == 'long_short':
            _position = (self.ls_group == self.ng if self.ng == 1 else self.ng - 1) - (self.ls_group == 0)
        elif kind == 'baseline':
            return Portfolio(w=idx_w)
        else:
            raise ValueError(f'Invalid portfolio kind: `{kind}`')

        _position *= idx_w.loc[self.sgn.bd: self.sgn.ed]
        _position = _position.apply(lambda s: s / s.abs().sum(), axis=1)
        self.holddays = hd
        _position = _position.iloc[::hd].reindex_like(_position).fillna(method='ffill')  # 连续持仓
        assert round(_position.dropna(how='all').abs().sum(axis=1).prod(), 4) == 1

        return Portfolio(w=_position)

    def get_ls_panels(self, path_f: str = None) -> dict:
        if path_f is not None:
            self.portfolio_ls.get_panel(path_f.format('PanelLongShort.csv'))
            self.portfolio_l.get_panel(path_f.format('PanelLong.csv'))
            self.portfolio_s.get_panel(path_f.format('PanelShort.csv'))
        self.all_panels = {'long_short': self.portfolio_ls.get_panel(),
                           'long': self.portfolio_l.get_panel(),
                           'short': self.portfolio_s.get_panel(),
                           'baseline': self.portfolio_baseline.get_panel()}
        return self.all_panels
