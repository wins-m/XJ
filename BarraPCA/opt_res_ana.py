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

    def __init__(self, ir1: pd.Series, close_adj, idx_cons, res_path, tc=None):
        self.closeAdj = close_adj  # conf['closeAdj']
        self.idx_cons = idx_cons  # conf['idx_constituent']
        self.res_path = res_path  # conf['factorsres_path']
        self.trade_cost = float(tc) if isinstance(tc, str) else tc  # conf['cr']
        self.alpha_name = ir1['alpha_name']
        self.mkt_type = ir1['mkt_type']
        self.suffix = info2suffix(ir1)
        self.path = f"{self.res_path}OptResWeekly[{self.mkt_type}]{self.alpha_name}/{self.suffix}/"

        self.W = pd.read_csv(self.path + f"portfolio_weight_{self.suffix}.csv", index_col=0, parse_dates=True)
        self.bd = self.W.index[0].strftime('%Y-%m-%d')
        self.ed = self.W.index[-1].strftime('%Y-%m-%d')
        self.views = list(self.W.index)
        self.port: Portfolio = Portfolio(w=self.W)
        self.opt_info = pd.read_excel(self.path + f'opt_info_{self.suffix}.xlsx', index_col=0, parse_dates=True)

        self.Return: pd.DataFrame = pd.DataFrame()  # returns next week
        self.Wealth: pd.DataFrame = pd.DataFrame()

    def figure_portfolio_weight(self):
        self.port.plot_weight_hist(
            path=self.path + 'figure_weight_hist_' + self.suffix + '.png')
        self.port.plot_port_asset_num(
            path=self.path + 'figure_portfolio_size_' + self.suffix + '.png',
            rw={'W': 1, '4W': 4, '52W': 52})
        self.port.plot_asset_weight(
            path=self.path + 'figure_portfolio_weight_' + self.suffix + '.png')

    def figure_historical_result(self):
        close_adj = pd.read_csv(self.closeAdj, index_col=0, parse_dates=True)
        close_adj_w = close_adj.loc[self.views]
        rtn_next_week = close_adj_w.pct_change().shift(-1)

        # Half Year Statistics (without cost)
        self.port.cal_panel_result(cr=0, ret=rtn_next_week)
        self.port.cal_half_year_stat(wc=False)
        self.port.get_half_year_stat(wc=False, path=self.path + 'table_half_year_stat_' + self.suffix + '.xlsx')
        if isinstance(self.trade_cost, float):
            self.port.cal_panel_result(cr=self.trade_cost, ret=rtn_next_week)
            self.port.cal_half_year_stat(wc=True)
            _path = self.path + 'table_half_year_stat_' + self.suffix + f'(cr={self.trade_cost}).xlsx'
            self.port.get_half_year_stat(wc=True, path=_path)

        # Table Return
        self.Return['portfolio'] = (rtn_next_week.reindex_like(self.W) * self.W).sum(axis=1)
        ind_cons_w = get_index_constitution(self.idx_cons.format(self.mkt_type), self.bd, self.ed)
        ind_cons_w: pd.DataFrame = ind_cons_w.loc[self.views]
        self.Return[self.mkt_type] = (rtn_next_week.reindex_like(ind_cons_w) * ind_cons_w).sum(axis=1)
        self.Return.to_excel(self.path + 'table_return_' + self.suffix + '.xlsx')

        # Table & Graph Holding Weight (2021-12-31)
        tmp = pd.concat(
            [
                self.W.loc['2021-12-31'].dropna().rename('port'),
                ind_cons_w.loc['2021-12-31'].dropna().rename('cons')
            ],
            axis=1)
        tmp = tmp.sort_values(['port', 'cons'], ascending=False)
        tmp['diff'] = tmp.iloc[:, 0] - tmp.iloc[:, 1]
        tmp.to_excel(self.path + 'table_holding_diff_20211231' + self.suffix + '.xlsx')
        tmp['diff'].dropna().reset_index(drop=True).plot(style='o', title=f'Weight Difference to {self.mkt_type}')
        plt.tight_layout()
        plt.savefig(self.path + 'figure_holding_diff_20211231_' + self.suffix + '.png')
        plt.close()
        tmp['port'].dropna().reset_index(drop=True).plot(title='Holding Weight')
        plt.tight_layout()
        plt.savefig(self.path + 'figure_holding_weight_20211231_' + self.suffix + '.png')
        plt.close()
        del tmp

        # Table & Graph Wealth
        self.Wealth = self.Return.cumsum()
        self.Wealth['Excess'] = self.Wealth.iloc[:, 0] - self.Wealth.iloc[:, 1]
        self.Wealth = self.Wealth.add(1)
        self.Wealth.to_excel(self.path + 'table_result_wealth_' + self.suffix + '.xlsx')
        self.Wealth.plot(title='Wealth (1 week ahead)')
        plt.tight_layout()
        plt.savefig(self.path + 'figure_result_wealth_' + self.suffix + '.png')
        plt.close()

    def figure_turnover(self):
        self.port.plot_turnover(
            path=self.path + 'figure_turnover_' + self.suffix + '.png',
            ishow=False)

    def figure_opt_time(self):
        sr = self.opt_info['stime']
        sr.plot(title=f'Solve Time, M={sr.mean():.3f}s')
        plt.tight_layout()
        plt.savefig(self.path + 'figure_solve_time_' + self.suffix + '.png')
        plt.close()

    def figure_risk_a_result(self):
        df = self.opt_info[['risk', 'opt0']]
        # df['risk'] = df['risk'].apply(lambda x: x if isinstance(x, float) else eval(x)[0][0])
        df['alpha'] = df['risk'] + df['opt0']
        _title = f"Result({df['opt0'].mean():.3f}): Alpha({df['alpha'].mean():.3f}) - Risk({df['risk'].mean():.3f})"
        df.rename(columns={'opt0': 'alpha - risk'}).plot(title=_title)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_alpha_risk_' + self.suffix + '.png')
        plt.close()


def func(args):
    ir1, conf, msg = args
    self = OptRes(ir1, conf['closeAdj'], conf['idx_constituent'], conf['factorsres_path'], conf['tc'])
    self.figure_historical_result()
    self.figure_portfolio_weight()
    self.figure_turnover()
    self.figure_opt_time()
    self.figure_risk_a_result()
    print(self.alpha_name + ': ' + self.suffix, msg)


def opt_res_ana(conf, test=False):
    # %% Configs:
    optimize_target: pd.DataFrame = pd.read_excel(OPTIMIZE_TARGET, index_col=0, dtype=object).loc[1:1]
    print(optimize_target)

    # %%
    if test:
        ir1 = optimize_target.iloc[0]
        func((ir1, conf, ''))
    else:
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
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    opt_res_ana(conf, test=False)
