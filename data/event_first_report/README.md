# 首次研报 Event First Report

1. `event_analysis.py`: 生成平均AR-CAR、相关性矩阵（后改为再ipynb中分析）等图像
2. `prepare_panel.py`: 准备事件面板，市场情况（涨跌停、上市/停牌日期、是否交易日）；计算前后若干日超额收益、累计超额收益；添加分组标签
3. `event_condition.py`: 配合ipynotebook的几个module
4. `*explore_panel.ipynb`: 分析比较不同筛选条件下回测结果
5. `weight2efr.py`: 维护数据库`efr_*_dur3` (`* := baseline, lcar8, har0, lcar8_har0`)，下载最近的EFR持仓到本地

### 目标

- ***对“首次研报”事件，确定筛选条件，s.t. 事件后X日累计超额收益(CAR_X) 期望最大***
- 方案一：筛选 以最大化 CAR3，T+0 close 购入，持有3日（T+3 close 平仓） 
- 方案二：筛选 以最大化 (AR1 AR2 AR3)，T+0 close 购入；确定 T+i 日行情筛选条件 T+i+1 open 卖出部分，T+4日平 
- 方案三：预测 CAR3 （持仓权重），T+1 open 购入，持有3日卖出 
- 方案四：预测 (AR1, AR2, AR3) ，由此确定 (T+1, T+2, T+3) close 持仓


### 步骤

1. `eventid`
2. Calculate R, R_mkt, AR, CAR before & after events, using `close` (time range -120~120)
3. `maxupdown` for T+0, T+1, T+2, T+3, & Days after last
4. `suspend` days after last
5. `instnum`


### 目录`factors_res/event_first_report/`下文件

```
event_first_report
├── AR_CAR_120.png
├── AR_CAR_15.png
├── AR_CAR_240.png
├── AR_CAR_30.png
├── AR_CAR_60.png
├── ARbox.png
├── ar_cumsum_corr(pearson).xlsx
├── ar_cumsum_corr(spearman).xlsx
├── corr120_pearson.png
├── corr120_spearman.png
├── corr25_pearson.png
├── corr25_spearman.png
├── event_abnormal_returns.csv
├── event_first_report_0209.zip
├── one_day_AR_after_event_2d.hdf
├── violin20.png
├── violin5.png
└── violin60.png
```