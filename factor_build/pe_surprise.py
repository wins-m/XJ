"""
(created by swmao on Jan.12th)
计算一致预期PE中不能由其他若干指标解释的异质性部分。
(Jan. 17th)
log(P/T) 原始值 - 缩微自变量的拟合值（用拟合值估计系数）
---
# `pe_surprise.py`

计算一致预期PE中不能由其他若干指标解释的异质性部分。

## 计算：风格内日内截面回归，取残差作为因子
pe ~ avgroe + np_chg_6m + np_chg_lid + np_growth + surprise + instnum1 + instnum2 + instnum3 + mv1 + mv2 + mv3

- 被解释变量`pe`
  - 为一致预期P/E`factor_west_pe_180`对数化后，
  - ~~在风格大类内中心化（减风格均值），~~（在回归中多余）
  - 再在风格大类内缩尾
- 解释变量（除`np_growth`）均直接截面内缩尾（version2: 在风格大类内部缩尾），图像上均是肥尾
- 若该日的解释变量覆盖个股数量少于400，则不纳入回归 各日期参与回归的面板shape见`panel_size_{date0}_{date1}.csv`
- 计算ols拟合值，Log(P/E)原始值 - 拟合值 为残差，结果见`pe_residual_{date0}_{date1}.csv`

## 解释变量选择（version3）
- `np_chg_6m`, `np_chg_lid`, `surprise`重复信息
- ols_1 : `np_chg_6m`
  - pe ~ avgroe + np_chg_6m + np_growth + instnum1 + instnum2 + instnum3 + mv1 + mv2 + mv3
- ols_2 : `np_chg_lid`
  - pe ~ avgroe + np_chg_lid + np_growth + instnum1 + instnum2 + instnum3 + mv1 + mv2 + mv3
- ols_3 : `surprise`
  - pe ~ avgroe + np_growth + surprise + instnum1 + instnum2 + instnum3 + mv1 + mv2 + mv3
- 若三个解释变量只保留一个，则结果文件名中缀加入 `ols_{i}`

## 指标面板处理
```python
# 面板指标
variables = {
    'factor_west_pe_180': ('fv', 'tradingdate', 2),  # 预期PE
    'instnum_class': ('class', 'tradingdate', 1),  # 机构关注度分类
    'mv_class': ('class', 'tradingdate', 1),  # 市值分类
    'factor_west_avgroe_180': ('fv', 'tradingdate', 2),  # 预期ROE（%）
    'factor_west_netprofit_chg_180_6_1m': ('fv', 'tradingdate', 2),  # 预期动量6M
    'factor_west_netprofit_chg_lid': ('fv', 'tradingdate', 2),  # 最新财报信息日后预期变化（？）
    'factor_west_netprofit_growth_180': ('fv', 'tradingdate', 2),  # 预期净利润增速
    'stk_west_surprise': ('surprise', 'update_date', 2),  # 报告年化超预期
    'ci_sector_constituent': ('industry', 'tradingdate', 0),  # 风格大类（四类）
}
```
所有指标的时间轴对齐到`factor_west_pe_180`
- 风格大类`ci_sector_constituent`自2020年6月向前补全；
- `instnum_class`和`mv_class`向前补全空值
- `stk_west_surprise`向后补全空值
- 其他指标保留空值

## 指标说明
| 衍生库factordatabase                               | 说明              | 开始日期       | 备注                          |
|-------------------------------------------------|-----------------|------------|-----------------------------|
| factor_west_avgroe_180 (avgroe)                 | 预期ROE（%）        | 2018-07-27 |                             |
| ~~factor_west_growth_avg~~                      | 预期综合增速          | 2016-03-26 | 不存在该表                       |
| factor_west_netprofit_chg_180_6_1m (np_chg_6m)  | 预期动量6M          | 2013-07-01 |                             |
| factor_west_netprofit_chg_lid (np_chg_lid)      | 最新财报信息日后预期变化（？） | 2013-09-01 | (2019-02-25及以后为PIT数据)(处理极值) |
| factor_west_netprofit_growth_180 (np_growth)    | 预期净利润增速         | 2013-01-01 | 处理极值                        |
| stk_west_surprise (surprise)                    | 报告年化超预期         | 2015-03-31 |                             |
| factor_west_pe_180 (pe)                         | 预期PE            | 2013-01-01 |                             |
| instnum_class (instn<br/>um1,instnum2,instnum3) | 机构关注度分类         | 2013-09-02 | 1,2,3                       |
| mv_class (mv1,mv2,mv3)                          | 市值分类            | 2013-09-02 | 1,2,3                       |

"""
import os
import pandas as pd
import numpy as np
import time
import statsmodels.formula.api as sm
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.neu import get_winsorize_sr
# from matplotlib import pyplot as plt


def cal_pe_surprise_g(begin_date: str, end_date: str, data_path: str, factorscsv_path: str, group: str, save_panel):
    """一致P/E预期异常"""

    def ols_residual(sub_df, fm, saying=False):
        """在DataFrame内依据公式回归返回残差"""
        ols_res = sm.ols(formula=fm, data=sub_df).fit()
        if saying:
            print(ols_res.summary())
        return ols_res.resid

    def ols_yhat(sub_df, fm, saying=False):
        """在DataFrame内依据公式回归返回预测值"""
        ols_res = sm.ols(formula=fm, data=sub_df).fit()
        if saying:
            print(ols_res.summary())
        return ols_res.predict(sub_df.iloc[:, 1:])

    save_filename = f"""pe_residual_{begin_date.replace('-','')}_{end_date.replace('-', '')}.csv"""
    if group != 'all':
        save_filename = save_filename.replace('pe_residual', f'pe_residual_{group}')
    factor_west_pe_180 = pd.read_csv(data_path + 'factor_west_pe_180.csv', index_col=0, parse_dates=True)
    ci_sector_constituent = pd.read_csv(data_path + 'ci_sector_constituent.csv', index_col=0, parse_dates=True)
    ci_sector_constituent = ci_sector_constituent.reindex_like(factor_west_pe_180).fillna(method='backfill')  # 补全20年6月前
    instnum_class = pd.read_csv(data_path + 'instnum_class.csv', index_col=0, parse_dates=True)
    instnum_class = instnum_class.reindex_like(factor_west_pe_180).fillna(method='backfill')
    mv_class = pd.read_csv(data_path + 'mv_class.csv', index_col=0, parse_dates=True)
    mv_class = mv_class.reindex_like(factor_west_pe_180).fillna(method='backfill')

    factor_west_netprofit_chg_180_6_1m, factor_west_netprofit_chg_lid, stk_west_surprise = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if group == 'all':
        factor_west_netprofit_chg_180_6_1m = pd.read_csv(data_path + 'factor_west_netprofit_chg_180_6_1m.csv',
                                                         index_col=0, parse_dates=True)
        factor_west_netprofit_chg_180_6_1m = factor_west_netprofit_chg_180_6_1m.reindex_like(factor_west_pe_180)
        factor_west_netprofit_chg_lid = pd.read_csv(data_path + 'factor_west_netprofit_chg_lid.csv', index_col=0,
                                                    parse_dates=True)
        factor_west_netprofit_chg_lid = factor_west_netprofit_chg_lid.reindex_like(factor_west_pe_180)
        stk_west_surprise = pd.read_csv(data_path + 'stk_west_surprise.csv', index_col=0, parse_dates=True)
        stk_west_surprise = stk_west_surprise.reindex_like(factor_west_pe_180)  # 一些日期可能无公告，需要补全
        stk_west_surprise = stk_west_surprise.fillna(method='ffill', limit=120)  # 向前填充，即无新报告公布，沿用最后报告值；最多回看120个交易日
    elif group == 'ols_1':
        factor_west_netprofit_chg_180_6_1m = pd.read_csv(data_path + 'factor_west_netprofit_chg_180_6_1m.csv',
                                                         index_col=0, parse_dates=True)
        factor_west_netprofit_chg_180_6_1m = factor_west_netprofit_chg_180_6_1m.reindex_like(factor_west_pe_180)
    elif group == 'ols_2':
        factor_west_netprofit_chg_lid = pd.read_csv(data_path + 'factor_west_netprofit_chg_lid.csv', index_col=0,
                                                    parse_dates=True)
        factor_west_netprofit_chg_lid = factor_west_netprofit_chg_lid.reindex_like(factor_west_pe_180)
    elif group == 'ols_3':
        stk_west_surprise = pd.read_csv(data_path + 'stk_west_surprise.csv', index_col=0, parse_dates=True)
        stk_west_surprise = stk_west_surprise.reindex_like(factor_west_pe_180)  # 一些日期可能无公告，需要补全
        stk_west_surprise = stk_west_surprise.fillna(method='ffill', limit=120)  # 向前填充，即无新报告公布，沿用最后报告值；最多回看120个交易日

    factor_west_avgroe_180 = pd.read_csv(data_path + 'factor_west_avgroe_180.csv', index_col=0, parse_dates=True)
    factor_west_avgroe_180 = factor_west_avgroe_180.reindex_like(factor_west_pe_180)
    factor_west_netprofit_growth_180 = pd.read_csv(data_path + 'factor_west_netprofit_growth_180.csv', index_col=0,
                                                   parse_dates=True)
    factor_west_netprofit_growth_180 = factor_west_netprofit_growth_180.reindex_like(factor_west_pe_180)

    trade_dates = factor_west_pe_180.index.to_list()
    #
    ind0 = trade_dates.index(pd.to_datetime(begin_date))
    ind1 = trade_dates.index(pd.to_datetime(end_date))
    factor_val = pd.DataFrame()
    panel_size = pd.DataFrame()
    lst_time = time_loop_start = time.time()
    td_i = ind1
    for td_i in range(ind0, ind1 + 1):
        td = trade_dates[td_i]
        td_str = td.strftime('%Y-%m-%d')
        print('DATE:', td_str, end='\t')
        pe = factor_west_pe_180.loc[td]
        # 查看pe的分布，确定取对数
        pe_log = pe.apply(np.log)
        panel = pe_log.rename('pe')
        # """pe.loc[td].hist(); plt.show()
        # np.log(pe.loc[td]).hist(); plt.show()
        # get_winsorize_sr(np.log(pe)).loc[td].hist(); plt.show()
        # get_winsorize_sr(pe).loc[td].hist(); plt.show()
        # (1/pe.loc[td]).hist(); plt.show()
        # pe_log = np.log(pe)
        # pe_log_winsorized = get_winsorize_sr(pe_log).loc[td]
        # pe_log = pe_log.loc[td]"""
        # 风格（四类，主要比较1和2）
        sector = ci_sector_constituent.loc[td]
        # pe_log_sector_avg = pe_log.groupby(sector).apply(lambda s: get_winsorize_sr(s).mean())
        # """# 不中心化
        # pe_log_sector_avg = pe_log.groupby(sector).mean()  # 分布对称，不需要缩尾
        # sector_pe = sector.apply(lambda x: pe_log_sector_avg[x] if not np.isnan(x) else np.nan)
        # pe_log_d_sector = pe_log - sector_pe  # 中心化，若该日期的该个股无ci行业风格标签，则空值
        # pe_log_d_sector_winso = pe_log_d_sector.groupby(sector).apply(lambda s: get_winsorize_sr(s))  # 分大类去极值，可简化
        # panel = pe_log_d_sector_winso.rename('pe')"""
        pe_log_winso = pe_log.groupby(sector).apply(lambda s: get_winsorize_sr(s))
        panel_winso = pe_log_winso.rename('pe')
        # panel_winso.groupby(sector).hist(legend=True); plt.show()  # 查看分布
        # 解释变量
        # factor_west_avgroe_180
        col_name = 'avgroe'
        df = factor_west_avgroe_180.loc[td]
        df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
        panel = pd.concat([panel, df.rename(col_name)], axis=1)
        panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
        # df.plot(kind='hist', title='factor_west_avgroe_180'); plt.show()
        # df1.T.plot(kind='hist', title='factor_west_avgroe_180, winsorized(nsigma=3)'); plt.show()
        #
        # factor_west_netprofit_growth_180
        col_name = 'np_growth'
        df = factor_west_netprofit_growth_180.loc[td]
        df1 = df  # 已经去过极值 df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
        panel = pd.concat([panel, df.rename(col_name)], axis=1)
        panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
        if group == 'all':
            # factor_west_netprofit_chg_180_6_1m
            col_name = 'np_chg_6m'
            df = factor_west_netprofit_chg_180_6_1m.loc[td]
            df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
            panel = pd.concat([panel, df.rename(col_name)], axis=1)
            panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
            # factor_west_netprofit_chg_lid
            col_name = 'np_chg_lid'
            df = factor_west_netprofit_chg_lid.loc[td]
            df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
            panel = pd.concat([panel, df.rename(col_name)], axis=1)
            panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
            # stk_west_surprise
            col_name = 'surprise'
            df = stk_west_surprise.loc[td]
            df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
            panel = pd.concat([panel, df.rename(col_name)], axis=1)
            panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
        elif group == 'ols_1':
            # factor_west_netprofit_chg_180_6_1m
            col_name = 'np_chg_6m'
            df = factor_west_netprofit_chg_180_6_1m.loc[td]
            df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
            panel = pd.concat([panel, df.rename(col_name)], axis=1)
            panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
        elif group == 'ols_2':
            # factor_west_netprofit_chg_lid
            col_name = 'np_chg_lid'
            df = factor_west_netprofit_chg_lid.loc[td]
            df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
            panel = pd.concat([panel, df.rename(col_name)], axis=1)
            panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
        elif group == 'ols_3':
            # stk_west_surprise
            col_name = 'np_chg_6m'
            df = stk_west_surprise.loc[td]
            df1 = df.groupby(sector).apply(lambda s: get_winsorize_sr(s))
            panel = pd.concat([panel, df.rename(col_name)], axis=1)
            panel_winso = pd.concat([panel_winso, df1.rename(col_name)], axis=1)
        # instnum_class
        lhs = pd.get_dummies(instnum_class.loc[td]).rename(columns={1: 'instnum1', 2: 'instnum2', 3: 'instnum3'})
        panel = pd.concat([panel, lhs], axis=1)
        panel_winso = pd.concat([panel_winso, lhs], axis=1)
        # mv_class
        lhs = pd.get_dummies(mv_class.loc[td]).rename(columns={1: 'mv1', 2: 'mv2', 3: 'mv3'})
        panel = pd.concat([panel, lhs], axis=1)
        panel_winso = pd.concat([panel_winso, lhs], axis=1)
        # ci_sector_constituent
        lhs = sector.rename('sector')
        panel = pd.concat([panel, lhs], axis=1)
        panel_winso = pd.concat([panel_winso, lhs], axis=1)
        # 不再统一去极值
        # panel_winso = panel.copy()
        # for col in panel.columns[:-7]:
        #     panel_winso[col] = panel[col].groupby(sector).apply(lambda s: get_winsorize_sr(s))
        #     #
        # 存一次原始面板
        if save_panel:
            os.makedirs(factorscsv_path + save_filename.replace('.csv', ''), exist_ok=True)
            panel.to_csv(factorscsv_path + save_filename.replace('.csv', f'/{td_str}_raw.csv'))
            panel_winso.to_csv(factorscsv_path + save_filename.replace('.csv', f'/{td_str}_reg.csv'))
        # 面板去空值
        panel_ind_cnt = panel_winso.count()
        panel_nonna = panel_winso[panel_ind_cnt[(panel_ind_cnt >= 400)].index]  # 解释变量覆盖个股数量少于400则忽略该变量
        # panel_nonna = panel.dropna(how='all', axis=1)  # 若解释变量全部缺失，则剔除该解释变量（因此因子值时间段前后不可比较）
        panel_nonna = panel_nonna.dropna(how='any', axis=0)
        print('PANEL:', panel_nonna.shape, end='\t')
        panel_size[td_str] = panel_nonna.shape
        # 回归
        if len(panel_nonna) > 0:
            var_list = panel_nonna.columns.to_list()
            fm = var_list[0] + ' ~ ' + ' + '.join(var_list[1:-1])  # 回归公式
            # 使用原始log(P/E) - 极值缩尾后的拟合 而非 缩尾后 - 缩尾后拟合
            # fv = panel_nonna.groupby('sector').apply(lambda s: ols_residual(s, fm))  # , saying=True))  # 分行业回归
            # fv = fv.rename(td_str)
            # factor_val = pd.concat((factor_val, fv1), axis=1)
            fv2 = panel.groupby('sector').apply(lambda s: s['pe']) - panel_nonna.groupby('sector').apply(lambda s: ols_yhat(s, fm))
            fv2 = fv2.dropna().rename(td_str)
            # (fv - fv2).sort_values()
            factor_val = pd.concat((factor_val, fv2), axis=1)
        else:
            factor_val[td_str] = np.nan
        # notice
        cur_time = time.time()
        print(f'loop time {(cur_time - lst_time):.3f} s')
        lst_time = cur_time
        if td_i % 100 == 0:  # 每100天存一次
            print('newest date:', td_str, "save in", factorscsv_path + save_filename)
            factor_val.to_csv(factorscsv_path + save_filename)
            panel_size.T.to_csv(factorscsv_path + save_filename.replace('pe_residual', 'panel_size'))
            #
    print(f'LOOP FINISHED, cost time {(time.time() - time_loop_start):.3f} s\n')
    factor_val.to_csv(factorscsv_path + save_filename)
    panel_size.T.to_csv(factorscsv_path + save_filename.replace('pe_residual', 'panel_size'))


def cal_pe_surprise(conf):
    """main"""
    begin_date = conf['begin_date']
    end_date = conf['end_date']
    data_path = conf['data_path']
    factorscsv_path = conf['factorscsv_path']
    group = conf['pe_ols_group']
    save_panel = conf['save_panel']
    if group == 'loop3':
        for ols_g in ['ols_1', 'ols_2', 'ols_3']:
            cal_pe_surprise_g(begin_date, end_date, data_path, factorscsv_path, ols_g, save_panel)
            #
    else:
        cal_pe_surprise_g(begin_date, end_date, data_path, factorscsv_path, group, save_panel)


if __name__ == '__main__':
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
    #
    cal_pe_surprise(conf)
