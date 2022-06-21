"""
(created by swmao on June 2nd)

"""
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import sys
from multiprocessing import Pool

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.bata_etf import info2suffix, get_index_constitution
from supporter.backtester import Portfolio

OPTIMIZE_TARGET = '/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/BarraPCA/optimize_target_v2.xlsx'


class OptRes(object):

    def __init__(self, ir1: pd.Series, conf: dict):
        self.info = ir1
        self.mkt_type = ir1['mkt_type']
        self.conf = conf
        self.closeAdj = conf['closeAdj']
        self.idx_cons = conf['idx_constituent']

        self.suffix: str = ''
        self.path: str = ''
        self.title: str = ''
        self.W: pd.DataFrame = pd.DataFrame()  # portfolio weight
        self.views: list = list()  # adjust dates
        self.opt_info: pd.DataFrame = pd.DataFrame()
        self.bd: str = ''
        self.ed: str = ''
        self.Return: pd.DataFrame = pd.DataFrame()  # returns next week
        self.Wealth: pd.DataFrame = pd.DataFrame()
        self._get_path()
        self._get_portfolio_weight()
        self._get_optimize_information()
        self.port: Portfolio = Portfolio(w=self.W)

    def tf_portfolio_weight(self):
        plt.hist(self.W.values.flatten(), bins=100)
        plt.title(self.title)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_weight_hist_' + self.suffix + '.png')
        plt.close()

        tmp = pd.DataFrame(self.W.count(axis=1).rename('W'))
        tmp['4W'] = tmp['W'].rolling(4).mean()
        tmp['26W'] = tmp['W'].rolling(26).mean()
        tmp.plot(title=self.title, linewidth=2)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_portfolio_size_' + self.suffix + '.png')
        plt.close()
        del tmp

        tmp = pd.DataFrame()
        tmp['w-MAX'] = self.W.max(axis=1)
        tmp['w-MEDIAN'] = self.W.median(axis=1)
        tmp['w-AVERAGE'] = self.W.mean(axis=1)
        tmp.plot(title=self.title, linewidth=2)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_portfolio_weight_' + self.suffix + '.png')
        plt.close()
        del tmp

    def tf_historical_result(self):
        close_adj = pd.read_csv(self.closeAdj, index_col=0, parse_dates=True)
        close_adj_w = close_adj.loc[self.views]
        rtn_next_week = close_adj_w.pct_change().shift(-1)
        # Half Year Statistics (without cost)
        self.port.cal_panel_result(cr=0, ret=rtn_next_week)
        self.port.cal_half_year_stat(wc=False)
        self.port.get_half_year_stat(wc=False, path=self.path + 'table_half_year_stat_' + self.suffix + '.xlsx')
        # Table Return  TODO: reuse methods in Portfolio
        self.Return['portfolio'] = (rtn_next_week.reindex_like(self.W) * self.W).sum(axis=1)
        ind_cons_w = get_index_constitution(self.idx_cons.format(self.mkt_type), self.bd, self.ed)
        ind_cons_w: pd.DataFrame = ind_cons_w.loc[self.views]
        self.Return[self.mkt_type] = (rtn_next_week.reindex_like(ind_cons_w) * ind_cons_w).sum(axis=1)
        self.Return.to_excel(self.path + 'table_return_' + self.suffix + '.xlsx')
        # Table & Graph Holding Weight (2021-12-31)
        tmp = self.W.loc['2021-12-31'].dropna().rename('port')
        tmp = pd.concat([tmp, ind_cons_w.loc['2021-12-31'].dropna().rename('cons')], axis=1)
        tmp = tmp.sort_values(['port', 'cons'], ascending=False)
        tmp['d'] = tmp.iloc[:, 0] - tmp.iloc[:, 1]
        tmp.to_excel(self.path + 'table_holding_diff_20211231' + self.suffix + '.xlsx')
        tmp['d'].dropna().reset_index(drop=True).plot(style='o', title=self.title)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_holding_diff_20211231_' + self.suffix + '.png')
        plt.close()
        tmp['port'].dropna().reset_index(drop=True).plot(title=self.title)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_holding_weight_20211231_' + self.suffix + '.png')
        plt.close()
        del tmp
        # Table & Graph Wealth
        self.Wealth = self.Return.cumsum()
        self.Wealth['Excess'] = self.Wealth.iloc[:, 0] - self.Wealth.iloc[:, 1]
        self.Wealth = self.Wealth.add(1)
        self.Wealth.to_excel(self.path + 'table_result_wealth_' + self.suffix + '.xlsx')
        self.Wealth.plot(title=self.title)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_result_wealth_' + self.suffix + '.png')
        plt.close()

    def tf_turnover(self):
        sr = self.opt_info['turnover']
        sr.plot(title=f'Turnover, M={sr.mean()*100:.2f}%')
        plt.tight_layout()
        plt.savefig(self.path + 'figure_turnover_' + self.suffix + '.png')
        plt.close()

    def tf_optimize_time(self):
        sr = self.opt_info['stime']
        sr.plot(title='Solve Time, M={sr.mean():.3f}s')
        plt.tight_layout()
        plt.savefig('figure_solve_time_' + self.suffix + '.png')
        plt.close()

    def tf_risk_and_result(self):
        df = self.opt_info[['risk', 'opt0']]
        df['alpha'] = df['risk'] + df['opt0']
        _title = f"Result({df['opt0'].mean():.3f}): Alpha({df['alpha'].mean():.3f}) - Risk({df['risk'].mean():.3f})"
        df[['risk', 'alpha']].plot(title=_title)
        plt.tight_layout()
        plt.savefig('figure_alpha_risk_' + self.suffix + '.png')
        plt.close()

    def _get_path(self):
        res_path = self.conf['factorsres_path']
        ir1 = self.info
        self.suffix = info2suffix(ir1)
        # self.suffix = f"{ir1['beta_suffix']}(B={ir1['B']},E={ir1['E']},D={ir1['D']},H0={ir1['H0']}" + \
        #               (')' if np.isnan(float(ir1['H1'])) else f",H1={ir1['H1']})")  # suffix for all result file
        self.path = res_path + f"OptResWeekly[{ir1['mkt_type']}]" + ir1['alpha_name'] + '/' + self.suffix + '/'
        self.title = self.info['alpha_name'] + ': ' + self.suffix

    def _get_portfolio_weight(self):
        self.W = pd.read_csv(self.path + f"portfolio_weight_{self.suffix}.csv",
                             index_col=0, parse_dates=True)
        self.bd = self.W.index[0].strftime('%Y-%m-%d')
        self.ed = self.W.index[-1].strftime('%Y-%m-%d')
        self.views = list(self.W.index)

    def _get_optimize_information(self):
        self.opt_info = pd.read_excel(self.path + f'opt_info_{self.suffix}.xlsx')


def func(args):
    ir1, conf, msg = args
    self = OptRes(ir1, conf)
    self.tf_historical_result()
    self.tf_portfolio_weight()
    self.tf_turnover()
    self.tf_optimize_time()
    self.tf_risk_and_result()
    print(self.title, msg)


def opt_res_ana():
    # %% Configs:
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    optimize_target: pd.DataFrame = pd.read_excel(OPTIMIZE_TARGET, index_col=0, dtype=object).loc[1:1]
    print(optimize_target)

    ir1 = optimize_target.iloc[0]
    # %%
    p = Pool(7)
    cnt = 0
    for ir in optimize_target.iterrows():
        cnt += 1
        msg = f'({cnt}/{len(optimize_target)})'
        p.apply_async(func, args=[(ir[1], conf, msg), ])
        # self = OptRes(ir[1], conf)
        # self.tf_historical_result()
        # self.tf_portfolio_weight()
    p.close()
    p.join()


# %%
if __name__ == '__main__':
    opt_res_ana()
