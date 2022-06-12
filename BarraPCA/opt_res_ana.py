"""
(created by swmao on June 2nd)

"""
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import sys
from multiprocessing import Pool

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.bata_etf import *

OPTIMIZE_TARGET = '/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/BarraPCA/optimize_target_v2.xlsx'


class OptRes(object):

    def __init__(self, ir1: pd.Series, conf: dict):
        self.info = ir1
        self.conf = conf
        self.suffix: str = ''
        self.path: str = ''
        self.title: str = ''
        self.portfolio_weight: pd.DataFrame = pd.DataFrame()
        self.bd: str = ''
        self.ed: str = ''
        self.rtn: pd.DataFrame = pd.DataFrame()
        self.rtn_cumsum: pd.DataFrame = pd.DataFrame()

        self._get_path()
        self._get_portfolio_weight()

    def tf_portfolio_weight(self):
        plt.hist(self.portfolio_weight.values.flatten(), bins=100)
        plt.title(self.title)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_weight_hist_' + self.suffix + '.png')
        plt.close()

        tmp = pd.DataFrame(self.portfolio_weight.count(axis=1).rename('W'))
        tmp['4W'] = tmp['W'].rolling(4).mean()
        tmp['26W'] = tmp['W'].rolling(26).mean()
        tmp.plot(title=self.title, linewidth=2)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_portfolio_size_' + self.suffix + '.png')
        plt.close()
        del tmp

        tmp = pd.DataFrame()
        tmp['w-MAX'] = self.portfolio_weight.max(axis=1)
        tmp['w-MEDIAN'] = self.portfolio_weight.median(axis=1)
        tmp['w-AVERAGE'] = self.portfolio_weight.mean(axis=1)
        tmp.plot(title=self.title, linewidth=2)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_portfolio_weight_' + self.suffix + '.png')
        plt.close()
        del tmp

    def tf_historical_result(self):
        close_adj = pd.read_csv(self.conf['closeAdj'], index_col=0, parse_dates=True)
        rtn_w2w = close_adj.loc[self.portfolio_weight.index].pct_change().shift(-1)
        self.rtn['portfolio'] = (rtn_w2w.reindex_like(self.portfolio_weight) * self.portfolio_weight).sum(axis=1)
        ind_cons_w = get_index_constitution(self.conf['idx_constituent'].format(self.info['mkt_type']), self.bd,
                                            self.ed)
        ind_cons_w: pd.DataFrame = ind_cons_w.loc[self.portfolio_weight.index]
        self.rtn[self.info['mkt_type']] = (rtn_w2w.reindex_like(ind_cons_w) * ind_cons_w).sum(axis=1)
        self.rtn.to_excel(self.path + 'table_return_' + self.suffix + '.xlsx')

        tmp = self.portfolio_weight.loc['2021-12-31'].dropna().rename('port')
        tmp = pd.concat([tmp, ind_cons_w.loc['2021-12-31'].dropna().rename('cons')], axis=1)
        tmp = tmp.sort_values(['port', 'cons'], ascending=False)
        tmp['d'] = tmp.iloc[:, 0] - tmp.iloc[:, 1]
        tmp.to_excel(self.path + 'table_holding_diff_20211231' + self.suffix + '.xlsx')
        #
        tmp['d'].dropna().reset_index(drop=True).plot(style='o', title=self.title)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_holding_diff_20211231_' + self.suffix + '.png')
        plt.close()
        #
        tmp['port'].dropna().reset_index(drop=True).plot(title=self.title)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_holding_weight_20211231_' + self.suffix + '.png')
        plt.close()
        #
        del tmp

        self.rtn_cumsum = self.rtn.cumsum()
        self.rtn_cumsum['Excess'] = self.rtn_cumsum.iloc[:, 0] - self.rtn_cumsum.iloc[:, 1]
        self.rtn_cumsum = self.rtn_cumsum.add(1)
        self.rtn_cumsum.to_excel(self.path + 'table_result_wealth_' + self.suffix + '.xlsx')

        self.rtn_cumsum.plot(title=self.title)
        plt.tight_layout()
        plt.savefig(self.path + 'figure_result_wealth_' + self.suffix + '.png')
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
        self.portfolio_weight = pd.read_csv(self.path + f"portfolio_weight_{self.suffix}.csv",
                                            index_col=0, parse_dates=True)
        self.bd = self.portfolio_weight.index[0].strftime('%Y-%m-%d')
        self.ed = self.portfolio_weight.index[-1].strftime('%Y-%m-%d')


def func(args):
    ir1, conf, msg = args
    self = OptRes(ir1, conf)
    self.tf_historical_result()
    self.tf_portfolio_weight()
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
