### Database

 [相聚数据库说明文档intern.xlsx](相聚数据库说明文档intern.xlsx) 

#### MySQL客户端

MySQL Workbench 8.0CE

IP地址是192.168.1.104 端口默认3306

数据库账号：intern01 密码：rh35th

swmao自用 intern02：fh840t

```python
from sqlalchemy import create_engine


def conn_mysql(dbname):
    """
    连接本地MySQL数据库
    dbname:数据库名称，string类型
    """	
    engine = create_engine('mysql+pymysql://intern01:rh35th@192.168.1.104:3306/'+dbname+'?charset=UTF8MB4')
    print('MySQL连接成功..')
    return engine
```

```python
Data = pd.read_sql_query("select tradingdate from jeffdatabase.tdays_d order by tradingdate desc limit 0,1", engine)
```

#### 基础库**jeffdatabase**

| 股池列表                 |                                                              |      |
| ------------------------ | ------------------------------------------------------------ | ---- |
| a_list_etf               | 上市ETF列表                                                  |      |
| a_list_suspendsymbol     | A股停牌股列表                                                |      |
| a_list_symbol            | 全部A股列表（含科创版）                                      |      |
| a_list_symbol_shn        | 陆股通股票列表                                               |      |
| stk_ipo_date             | A股股票上市/退市日期                                         |      |
| stk_maxupordown          | A股个股涨跌停状态（1涨停；-1跌停）                           |      |
| idx_constituent          | 指数成分股                                                   |      |
|                          |                                                              |      |
| zzdel_a_list_symbol      | 全部A股列表(+2007-01-04)(+2007-12-28)(+2010-2015,-12-31)(+2010-2016,-04-30)(+2010-2016,-08-31)(+2010-2016,-10-31) |      |
| zzdel_a_list_symbol_star | 科创板股票列表                                               |      |

| 日期           |                      |      |
| -------------- | -------------------- | ---- |
| alldays_d      | 日度自然日日期       |      |
| alldays_h      | 半年度自然日日期     |      |
| alldays_m      | 月度自然日日期       |      |
| alldays_q      | 季度自然日日期       |      |
| alldays_y      | 年度自然日日期       |      |
| tdays_d        | 日度交易日日期       |      |
| tdays_d_shn    | 陆股通日度交易日日期 |      |
| tdays_m        | 月度交易日日期       |      |
| tdays_w        | 周度交易日日期       |      |
| tdays_d_offset | ？                   |      |

| 行业                    |                                                              |      |
| ----------------------- | ------------------------------------------------------------ | ---- |
| ind_list_citic          | 中信行业列表                                                 |      |
| ind_list_sw             | 申万行业列表                                                 |      |
| ind_citic_constituent   | 个股中信行业分类（含科创板）                                 |      |
| ind_sw_constituent      | 个股申万行业分类(一级行业从2014-01-02开始)、2018-09-28、2018-06-29、2007-01-04读取完成 |      |
| ci_sector_constituent   | 中信行业分类风格分类(1-成长 2-消费 3-周期 4-金融 5-综合)     |      |
| ind_sw_constituent_star | 科创板申万一级行业分类                                       |      |
| stk_concept             | A股市场个股所属WIND概念板块(2020-04-15及之后为PIT数据)(含科创板2020-04-15及之后数据) |      |

| 行情                     |                                                              |      |
| ------------------------ | ------------------------------------------------------------ | ---- |
| stk_marketdata           | A股市场个股行情(含科创板)（dealnum开始日期详见Chengjiaobishu.m） |      |
| stk_marketvalue          | A股市场个股市值                                              |      |
| fx_hkdcnyset             | 年度自然日日期中信行业分类陆股通结算汇率                     |      |
| etf_marketdata           | 中国上市ETF行情数据                                          |      |
| fut_marketdata           | 股指期货行情数据                                             |      |
| idx_marketdata           | A股市场指数行情数据                                          |      |
| zzdel_idx_marketdata_min | A股市场指数分钟线行情(000016.SH、000300.SH、000905.SH)       |      |

| 大市               |                                          |      |
| ------------------ | ---------------------------------------- | ---- |
| edb_bond_rate      | 国债到期收益率：1年                      |      |
| edb_bond_rate      | 金融机构人民币贷款加权平均利率：一般贷款 |      |
| edb_margin_trading | EDB-A股市场融资业务数据                  |      |
| edb_risk_track_org | EDB-风险跟踪指标原值                     |      |
| edb_shn_trading    | EDB-陆股通成交金额数据                   |      |
| fx_hkdcnyset       | 陆股通结算汇率                           |      |

| 持股                      |                                    |      |
| ------------------------- | ---------------------------------- | ---- |
| shn_stockholdings         | 陆股通持股明细                     |      |
| stk_holder_pctbyfund_q    | 基金持仓比例                       |      |
| stk_majorholderdealrecord | 全部A股-重要股东二级市场交易(明细) |      |

| 基本面                             |                                                              |      |
| ---------------------------------- | ------------------------------------------------------------ | ---- |
| stk_deductedprofit_yoy             | 单季度.扣除非经常性损益后的净利润同比增长率                  |      |
| stk_pb_lf                          | 市净率PB                                                     |      |
| stk_pcf_ocf_ttm                    | 市现率PCF(总市值/经营现金净流量TTM)                          |      |
| stk_pe_ttm                         | 市盈率TTM                                                    |      |
| stk_performanceexpress_date        | 个股业绩快报披露日期(2019-02-22及以后update_date为真实值)    |      |
| stk_profitnotice                   | 个股业绩预告明细(2020-08-20及以后update_date为真实值)        |      |
| stk_ps_ttm                         | 市销率TTM                                                    |      |
| stk_stm_issuingdate                | 个股定期报告披露日期(2019-02-22及以后update_date为真实值)    |      |
| stk_stockwest                      | 个股盈利预测明细(2013-2016年数据来自GoGoal，2020-08-26及之后update_date为真实值) |      |
| stk_west_avgroe_180                | 万得一致预期ROE(%)(+2013-2018,-04-30)(+2013-2017,-08-31)(+2013-2017,-10-31) |      |
| stk_west_instnum_180               | 万得一致预期预测机构家数                                     |      |
| stk_west_netprofit_180             | 万得一致预期净利润                                           |      |
| stk_west_netprofit_changegrade_180 | 万得一致预期净利润上下调家数(Y+1年)                          |      |
| stk_west_sales_180                 | 万得一致预期营业收入                                         |      |
| zzdel_stk_profitnotice             | 个股业绩预告明细                                             |      |



| factordatabase                                 | 说明                                                         | 开始日期   | 备注       |
| ---------------------------------------------- | ------------------------------------------------------------ | ---------- | ---------- |
| Intern02 : factordatabase                      |                                                              |            |            |
| factor_west_avgroe_180 (avgroe)                | 预期ROE（%）                                                 | 2018-07-27 |            |
| ~~factor_west_growth_avg~~                     | 预期综合增速                                                 | 2016-03-26 | 不存在该表 |
| factor_west_netprofit_chg_180_6_1m (np_chg_6m) | 预期动量6M                                                   | 2013-07-01 |            |
| factor_west_netprofit_chg_lid (np_chg_lid)     | 最新财报信息日后预期变化(2019-02-25及以后为PIT数据)(处理极值) | 2013-09-01 |            |
| factor_west_netprofit_growth_180 (np_growth)   | 预期净利润增速（处理极值）                                   | 2013-01-01 |            |
| stk_west_surprise (surprise)                   | 报告年化超预期                                               | 2015-03-31 |            |
| factor_west_pe_180 (pe)                        | 预期PE                                                       | 2013-01-01 |            |
| Intern01 : factordatabase                      |                                                              |            |            |
| instnum_class (instnum1,instnum2,instnum3)     | 机构关注度分类                                               | 2013-09-02 | 1,2,3      |
| mv_class (mv1,mv2,mv3)                         | 市值分类                                                     | 2013-09-02 | 1,2,3      |

---

