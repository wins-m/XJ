解释变量

| 表名                               |                          | 数据开始日期 | 数据结束日期 | 更新频率   |
| ---------------------------------- | ------------------------ | ------------ | ------------ | ---------- |
| factor_west_avgroe_180             | 预期ROE(%)               | 2018-07-27   | T-1          | 日度自然日 |
| factor_west_growth_avg             | 预期综合增速             | 2016-03-26   | T-1          | 日度自然日 |
| factor_west_netprofit_chg_180_6_1m | 预期动量6M               | 2013-07-01   | T-1          | 日度自然日 |
| factor_west_netprofit_chg_lid      | 最新财报信息日后预期变化 | 2013-09-01   | T-1          | 日度自然日 |
| factor_west_netprofit_growth_180   | 预期净利润增速(处理极值) | 2013-01-01   | T-1          | 日度自然日 |
| stk_west_surprise                  | 报告年化超预期           | 2015-03-31   | T-1          | 日度交易日 |
| factor_west_pe_180                 | 预期PE                   | 2013-01-01   | T-1          | 日度自然日 |
| instnum_class                      | 机构关注度分类           | 2013-09-02   | T-1          | 日度交易日 |
| mv_class                           | 市值分类                 | 2013-09-02   | T-1          | 日度交易日 |

- 市值 | size | log(market capitalization)
- 盈利能力
    - 每股红利 dividend payout ratio | DP | 股利/股本 
    - 净利润增长率 | profit | 同比增长
    - 净资产收益率 | roe | 扣除非经常损益后每股收益/每股净资产
    - 净收益营运指数 | Income Quality | 扣除非经常损益后的净利润/净利润
- 成长能力
    - EPS增长率 | eps, g
- 营运能力
    - 总资产周转率 | TAT | 销售收入净额/平均资产总额
- 研发能力
    - 研发投入增长率 | R&D | 同比增长
- 偿债能力
    - 资产负债率 | Leverage | 负债总额/资产总额
    - 流动比率 | liquidity | 流动资产/流动负债
- 市盈率
    - 发行市盈率 | ipo_pe
    - 行业平均市盈率 | pe_industry
    - EP | earnings per share / market value per share
    - ROE | 利润/账面价值
- 市场因素
    - 一年期贷款利率 | ir
    - beta系数 | beta
    - 换手率 | vol
    - 相对指数涨跌幅 | index
    - 价格动量 | mom
- 股权结构
    - 持股比例（第一大股东） | Construction
    - 持股比例（前十大股东） | ISR

---

