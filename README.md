# `get_data.py`
下载单项指标（主要为因子）的面板数据，long转wide格式存入本地；
- 所用表格在[access_target](./data/access_target.xlsx)中指定


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

---


# `save_remote.py`
将本地表格上传，注意在config文件中指定表格文件名列表
- 表格由本地的wide转成long
- 增加在服务器建表（指定格式），建表函数写在`supporter.mysql`
- 列名主要为`tradingdate`, `stockcode`, `fv`, `id`
  - （可精简）对于`pe_residual*.csv`，包括`industry`，由于同一股票在时间段内风格大类可能变化
- 去除空值后上传
