import sys
import time
from datetime import timedelta
from typing import Dict

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
        self.panel: pd.DataFrame = pd.DataFrame(
            index=w.index, columns=['NStocks', 'Turnover', 'Return', 'Return_wc',
                                    'Wealth(cumsum)', 'Wealth_wc(cumsum)', 'Wealth(cumprod)', 'Wealth_wc(cumprod)'])
        self.cost_rate = None
        self.stat = {}
        self.mdd = {}

    def cal_panel_result(self, cr, ret):
        self.cost_rate = cr
        self.panel = portfolio_statistics_from_weight(weight=self.w_2d, cost_rate=cr, all_ret=ret)

    def cal_half_year_stat(self, wc=False):
        col = 'Return_wc' if wc else 'Return'
        self.stat[wc] = cal_result_stat(self.panel[[col]])

    def plot_turnover(self, ishow, path):
        if self.panel is None:
            raise AssertionError('Calculate panel result before plot turnover!')
        self.panel['Turnover'].plot(figsize=(10, 5), grid=True, title='Turnover')
        plt.savefig(path)
        if ishow:
            plt.show()
        else:
            plt.close()

    def plot_cumulative_returns(self, ishow, path, kind='cumsum', title=None):
        title = f'Portfolio Absolute Result ({kind})' if title is None else title
        self.panel[[f'Wealth({kind})', f'Wealth_wc({kind})']].plot(figsize=(10, 5), grid=True, title=title)
        plt.savefig(path)
        if ishow:
            plt.plot()
        else:
            plt.close()

    def plot_max_drawdown(self, ishow, path, wc=False, kind='cumsum', title=None):
        col = f'Wealth_wc({kind})' if wc else f'Wealth({kind})'
        title = f'MaxDrawdown {col}' if title is None else title
        df = self.panel[col].copy()
        df = df + 1 if df.iloc[0] < .6 else df
        cal_sr_max_drawdown(df=df, ishow=ishow, title=title, save_path=path, kind=kind)

    def get_position_weight(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.w_2d.to_csv(path)
        return self.w_2d.copy()

    def get_panel(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.panel.to_csv(path)
        return self.panel.copy()

    def get_stock_number(self) -> pd.Series:
        return self.panel['NStocks'].copy()

    def get_turnover(self) -> pd.Series:
        return self.panel['Turnover'].copy()

    def get_daily_ret(self, wc=False) -> pd.Series:
        return self.panel['Return_wc' if wc else 'Return'].copy()

    def get_wealth(self, wc=False, kind='cumprod') -> pd.Series:
        return self.panel[f'Wealth_wc({kind})' if wc else f'Wealth({kind})'].copy()

    def get_half_year_stat(self, wc=False, path=None) -> pd.DataFrame:
        if wc not in self.stat.keys():
            print('Calculate half-year statistics before get_stat...')
            self.cal_half_year_stat(wc=wc)
        if path is not None:
            self.stat[wc].to_csv(path)
        return self.stat[wc]


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
        self.portfolio: Dict[str, Portfolio] = {}
        # self.all_panels: Dict[str, pd.DataFrame] = {}

    def cal_long_short_group(self):
        self.ls_group = get_long_short_group(df=self.sgn.fv, ngroups=self.ng)

    def cal_group_returns(self, ret, idx_w):
        ret = ret.loc[self.sgn.get_fbegin(): self.sgn.get_fend()]  # ?冗余
        self.ls_g_rtns = cal_long_short_group_rtns(
            long_short_group=self.ls_group, ret=ret, idx_weight=idx_w, ngroups=self.ng)

    def cal_long_short_panels(self, idx_w, hd, rvs, cr, ret):
        """由self.ls_group获得long, short, long_short, baseline的Portfolio，并计算序列面板"""
        for kind in ['long_short', 'long', 'short', 'baseline']:
            self.portfolio[kind] = self.get_holding_position(idx_w=idx_w, hd=hd, rvs=rvs, kind=kind)
            self.portfolio[kind].cal_panel_result(cr=cr, ret=ret)
            # self.all_panels[kind] = self.portfolio[kind].get_panel()

    def plot_group_returns(self, ishow, path):
        plot_rtns_group(self.ls_g_rtns, ishow, path)

    def plot_group_returns_total(self, ishow, path):
        cal_total_ret_group(self.ls_g_rtns, ishow, path)

    def plot_turnover(self, ishow, path):
        long_short_turnover = pd.concat([self.portfolio[k].get_turnover().rename(k) for k in ['long', 'short']], axis=1)
        long_short_turnover.plot(figsize=(10, 5), grid=True, title='Turnover')
        plt.savefig(path)
        if ishow:
            plt.show()
        else:
            plt.close()

    def plot_cumulative_returns(self, ishow, path, wc=False, kind='cumsum', excess=False):
        df = pd.concat([self.portfolio[k].get_wealth(wc, kind).rename(k) for k in ['baseline', 'long_short', 'long', 'short']], axis=1)
        if excess:
            df[['long_short', 'long', 'short']] -= df['baseline'].values.reshape(-1, 1)
            df = df[['long_short', 'long', 'short']]
        title = f'Long-Short {["Absolute", "Excess"][excess]} Result({kind}) {["No Cost", "With Cost"][wc]}'
        df.plot(figsize=(10, 5), grid=True, title=title)
        plt.savefig(path)
        if ishow:
            plt.show()
        else:
            plt.close()

    def plot_annual_return_bars(self, ishow, path, wc=False):
        pass

    def plot_annual_sharpe(self, ishow, path, wc=False):
        pass

    def get_ls_group(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.ls_group.to_csv(path)
        return self.ls_group

    def get_group_returns(self, path=None) -> pd.DataFrame:
        if path is not None:
            self.ls_g_rtns.to_csv(path)
        return self.ls_g_rtns

    def get_holding_position(self, idx_w, hd=1, rvs=False, kind='long') -> Portfolio:
        """
        由2d分组序号self.ls_group获得持仓组合
        :param idx_w: 2d权重，日截面上各股权重配比，不要求行和为一
        :param hd: 持仓长度（日），调仓周期
        :param rvs: 是否取反(分组依据因子值越大表现越好)。若False，组序号最大long，组序号最小short
        :param kind: 支持long, short, long_short
        :return: Portfolio(weight)
        """
        if (kind == 'long' and not rvs) or (kind == 'short' and rvs):
            _position = (self.ls_group == self.ng if self.ng == 1 else self.ng - 1).astype(int)
            _position *= idx_w.loc[self.sgn.bd: self.sgn.ed]
        elif (kind == 'short' and not rvs) or (kind == 'long' and rvs):
            _position = (self.ls_group == 0).astype(int)
            _position *= idx_w.loc[self.sgn.bd: self.sgn.ed]
        elif kind == 'long_short':
            _position = (self.ls_group == self.ng if self.ng == 1 else self.ng - 1).astype(int) - \
                        (self.ls_group == 0).astype(int)
            _position = -_position if rvs else _position
            _position *= idx_w.loc[self.sgn.bd: self.sgn.ed]
        elif kind == 'baseline':
            _position = idx_w.loc[self.sgn.bd: self.sgn.ed].copy()
        else:
            raise ValueError(f'Invalid portfolio kind: `{kind}`')

        _position = _position.apply(lambda s: s / s.abs().sum(), axis=1)
        self.holddays = hd
        _position = _position.iloc[::hd].reindex_like(_position).fillna(method='ffill')  # 连续持仓
        assert round(_position.dropna(how='all').abs().sum(axis=1).prod(), 4) == 1

        return Portfolio(w=_position)

    def get_ls_panels(self, path_f: str = None) -> dict:
        if path_f is not None:
            self.portfolio['long_short'].get_panel(path_f.format('PanelLongShort.csv'))
            self.portfolio['long'].get_panel(path_f.format('PanelLong.csv'))
            self.portfolio['short'].get_panel(path_f.format('PanelShort.csv'))
        return {k: self.portfolio[k].get_panel() for k in self.portfolio.keys()}

    def get_portfolio_statistics(self, kind='long', wc=False, path_f=None):
        """获取半年表现面板"""
        _kind = kind.replace('long', 'Long').replace('short', 'Short') + ['NC', 'WC'][wc]
        path = None if path_f is None else path_f.format(f'Res{_kind}.csv')
        return self.portfolio[kind].get_half_year_stat(wc=wc, path=path)
