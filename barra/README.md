## BARRA模型

假设资产收益由共同因子驱动：

- $r_n$可以由国家因子$f_c$，行业因子$f_i$，风格因子$f_s$表出；

- 其余为各资产特质收益，互不相关；

- 在截面内，资产收益率可以由下式表示
    $$
    r_n =
    	f_c + \sum_{i=1}^{P}{X_{ni} f_{i}} + \sum_{s=1}^{Q}{X_{ns} f_{s}} + \varepsilon_n \\
    \text{i.e.} \quad
    \bold{Y} = \bold{F} \bold{X} + \bold{\Epsilon}
    $$

    > - $\bold{X}$：N维资产在K维因子上的因子暴露
    > - $\bold{Y}$：N维资产在时间段内的收益
    > - $\bold{F}$：待求解的纯因子收益

- 由于特质收益率$\bold{\Epsilon}$方差不一定相同，根据实证经验，方差与市值平方根成反比，因此用加权最小二乘（WLS）估计，权重为市值平方根占比
    $$
    \min_\bold{F} {
    	(\bold{Y} - \bold{F} \bold{X})^T \bold{V} (\bold{Y} - \bold{F} \bold{X})
    } \\
    \text{where} \quad
    \bold{V} = \left[
    \begin{matrix}
    v_1 & 0 & \dots & 0 \\ 
    0 & v_2 & \dots & 0 \\
    \vdots & \vdots & & \vdots \\
    0 & 0 & \dots & v_{N} 
    \end{matrix}
    \right]
    ,\quad
    v_n = { \sqrt{s_n} \over \sum_{n=1}^{N}{\sqrt{s_n}}}
    $$

- 国家因子和风格因子存在共线性，因此需要约束条件
    $$
    \sum_{i=1}^P {S_{I_i} f_{I_i}} = 0 \\
    \text{i.e.} \quad
    \bold{F}_{K\times 1} = \bold{R}_{K \times (K-1)} \bold{F}'_{(K-1) \times 1} + \bold{0} 
    \\
    \text{i.e.} \quad
    \left[\begin{matrix}
    f_C \\
    f_{I_1} \\
    \vdots \\
    f_{I_P} \\
    f_{S_1} \\
    \vdots \\
    f_{S_Q} 
    \end{matrix}\right]  
    =
    \left[
    \begin{matrix}
    1 & 0 & 0  & \dots & 0 & 0 & \dots & 0 \\
    0 & 1 & 0  & \dots & 0 & 0 & \dots & 0 \\
    \vdots & \vdots & \vdots &  & \vdots & \vdots & & \vdots \\
    0 & -{S_{I_1}\over S_{I_P}} & -{S_{I_2} \over S_{I_P}} & \dots & -{S_{I_{P-1}} \over S_{I_P}} & 0 & \dots & 0 \\
    0 & 0 & 0  & \dots & 0 & 1 & \dots & 0 \\
    \vdots & \vdots & \vdots &  & \vdots & \vdots & & \vdots \\
    0 & 0 & 0  & \dots & 0 & 0 & \dots & 1 \\
    \end{matrix}
    \right]
    
    \left[
    \begin{matrix}
    f_C \\
    f_{I_1} \\
    \vdots \\
    f_{I_{P-1}} \\
    f_{S_1} \\
    \vdots \\
    f_{S_Q}
    \end{matrix}
    \right]
    + 
    \left[\begin{matrix}
    0 \\ 0 \\ \vdots \\ 0 \\ 0 \\ \vdots \\ 0
    \end{matrix}
    \right]
    $$

    - 约束条件中 $S_{I_i}$ 为行业 $I_i$ 整体市值之和

- 求解带约束的WLS得到纯因子组合的资产权重矩阵 $\bold{\Omega}$（Menchero & Lee, 2015)
    $$
    \bold{\Omega} = \bold{R} (\bold{R}^T \bold{X}^T \bold{V} 
    \bold{X} \bold{R})^{-1} \bold{R}^T \bold{X}^T \bold{V}
    $$

- K维纯因子收益率
    $$
    \bold{F}_{K \times 1} = \bold{\Omega}_{K \times N} \bold{Y}_{N \times 1}
    $$

### 风格因子`get_barra.py`

获取数据库中的因子 来源： [聚宽说明](https://www.joinquant.com/help/api/help#JQData:ALPHA%E7%89%B9%E8%89%B2%E5%9B%A0%E5%AD%90) BARRA CNE5 已进行去极值、标准化

| 因子 code             | 因子名称  | 简介                               |
|:--------------------|:------|:---------------------------------|
| size                | 市值    | 捕捉大盘股和小盘股之间的收益差异                 |
| beta                | 贝塔    | 表征股票相对于市场的波动敏感度                  |
| momentum            | 动量    | 描述了过去两年里相对强势的股票与弱势股票之间的差异        |
| residual_volatility | 残差波动率 | 解释了剥离了市场风险后的波动率高低产生的收益率差异        |
| non_linear_size     | 非线性市值 | 描述了无法由规模因子解释的但与规模有关的收益差异，通常代表中盘股 |
| book_to_price_ratio | 账面市值比 | 描述了股票估值高低不同而产生的收益差异, 即价值因子       |
| liquidity           | 流动性   | 解释了由股票相对的交易活跃度不同而产生的收益率差异        |
| earnings_yield      | 盈利能力  | 描述了由盈利收益导致的收益差异                  |
| growth              | 成长    | 描述了对销售或盈利增长预期不同而产生的收益差异          |
| leverage            | 杠杆    | 描述了高杠杆股票与低杠杆股票之间的收益差异            |

**聚宽因子数据处理说明**

对描述因子和风格因子的数据分别进行正规化的处理，步骤如下：

- 对描述因子分别进行去极值和标准化
    去极值为将2.5倍标准差之外的值，赋值成2.5倍标准差的边界值
    标准化为市值加权标准化
    x=(x- mean(x))/(std(x))
    其中，均值的计算使用股票的市值加权，标准差为正常标准差。
- 对描述因子按照权重加权求和
    按照公式给出的权重对描述因子加权求和。如果某个因子的值为nan，则对不为nan的因子加权求和，同时权重重新归一化；如果所有因子都为nan，则结果为nan。
- 对风格因子市值加权标准化
- 缺失值填充
    按照聚宽一级行业分行业，以不缺失的股票因子值相对于市值的对数进行回归，对缺失值进行填充
- 对风格因子去极值，去极值方法同上面去极值描述


### 纯因子收益率计算（国家+风格+行业）`cal_factor_return.py`

- 计算纯因子收益率：T-1期结束时的风格暴露，对应T期的资产收益，得到T-1期的纯因子收益（解释T期的资产）

- 时间范围：2012-01-01 ~ 2022-03-31

- 缓存/结果地址：
  - `/mnt/c/Users/Winst/Documents/data_local/BARRA/`
  
- Y: 全市场 ctc收益率（昨日Close买，今日Close卖）
  - 去除：上市 120 交易日内；昨日、今日停牌；昨日涨停、今日跌停
  
- X: 国家(1) 风格(10) 行业(29/30)
  - 风格因子，历年原始值存在`barra_exposure.h5`（key形如 `y2022`）
  - 风格因子日截面，对行业正交（size）or 对行业和size正交（其他9个）
  - 行业选用中信一级`indus_citic`29或30个 2019.12.2开始30个（新增“综合金融”）
  - 正交化后的面板存在`barra_panel.h5`（key形如 `y2022`）
  
- WLS（Menchero & Lee, 2015)
    $$
    \bold{\Omega} = \bold{R} (\bold{R}^T \bold{X}^T \bold{V} 
    \bold{X} \bold{R})^{-1} \bold{R}^T \bold{X}^T \bold{V} \\
    
    \bold{F}_{K \times 1} = \bold{\Omega}_{K \times N} \bold{Y}_{N \times 1}
    $$
    
    - 纯因子构成存在`barra_omega.h5`（key形如`d20220101`）
    - 纯因子收益率存在`barra_fval.h5`（key形如 `y2022`）
    - 历年合并为`barra_fval_20120104_20220331.csv`

**运行记录**

```zsh
(base) swmao:PyCharmProject/ (master✗) $ python barra/cal_factor_return.py                  [8:55:13]

 2022
Industry Missing 0.09 %
Return Missing 1.74 %
before (269146, 42)
after missing-drop (264066, 42)
100%|█████████████████████████████████████████████████████████████████| 57/57 [00:20<00:00,  2.78it/s]

 2021
Industry Missing 0.02 %
Return Missing 0.00 %
before (1062384, 42)
after missing-drop (1062273, 42)
100%|███████████████████████████████████████████████████████████████| 243/243 [01:49<00:00,  2.22it/s]

 2020
Industry Missing 0.42 %
Return Missing 0.00 %
before (951992, 42)
after missing-drop (951276, 42)
100%|███████████████████████████████████████████████████████████████| 243/243 [01:47<00:00,  2.26it/s]

 2019
Industry Missing 0.48 %
Return Missing 0.00 %
before (890251, 42)
after missing-drop (889682, 42)
100%|███████████████████████████████████████████████████████████████| 244/244 [01:34<00:00,  2.57it/s]

 2018
Industry Missing 0.40 %
Return Missing 0.03 %
before (857076, 41)
after missing-drop (856683, 41)
100%|███████████████████████████████████████████████████████████████| 243/243 [01:34<00:00,  2.56it/s]

 2017
Industry Missing 1.87 %
Return Missing 0.06 %
before (799553, 41)
after missing-drop (798213, 41)
100%|███████████████████████████████████████████████████████████████| 244/244 [01:30<00:00,  2.69it/s]

 2016
Industry Missing 1.34 %
Return Missing 0.07 %
before (704632, 41)
after missing-drop (703988, 41)
100%|███████████████████████████████████████████████████████████████| 244/244 [01:21<00:00,  2.98it/s]

 2015
Industry Missing 0.38 %
Return Missing 0.07 %
before (665670, 41)
after missing-drop (665179, 41)
100%|███████████████████████████████████████████████████████████████| 244/244 [01:20<00:00,  3.03it/s]

 2014
Industry Missing 0.31 %
Return Missing 0.08 %
before (620558, 41)
after missing-drop (619576, 41)
100%|███████████████████████████████████████████████████████████████| 245/245 [01:19<00:00,  3.07it/s]

 2013
Industry Missing 0.12 %
Return Missing 0.08 %
before (588055, 41)
after missing-drop (587435, 41)
100%|███████████████████████████████████████████████████████████████| 238/238 [01:16<00:00,  3.12it/s]

 2012
Industry Missing 0.41 %
Return Missing 0.08 %
before (587190, 41)
after missing-drop (586492, 41)
100%|███████████████████████████████████████████████████████████████| 243/243 [01:19<00:00,  3.06it/s]
```

## 全市场风格解析

- 目标年份2021

- 因子暴露共线性检验
    $$
    RSI_{AB} = {mean(Corr^{AB}_t) \over std(Corr^{AB}_t)}, \quad t=1,2,\dots,T
    $$

    - ![各风格因子暴露相关强度RSI](https://s2.loli.net/2022/04/08/AcKHLSMBqmUkV2J.png)
    - ![各风格因子暴露相关性系数均值](https://s2.loli.net/2022/04/08/epLnlbv35NDItjO.png)
    - ![各风格因子暴露相关性系数标准差](https://s2.loli.net/2022/04/08/gSNIjCqhA9PreBX.png)

- 国家因子与全市场市值加权收益率：两者理论上应该一致

    - ![截距项国家因子与全市场日度收益市值加权（2021年度）](https://s2.loli.net/2022/04/08/cEi8Cw6rBZ5LqQa.png)
    - ![国家因子与全市场日度收益累积（2021年度）](https://s2.loli.net/2022/04/08/3j2vXq7GydYfH6l.png)

- 纯净行业因子与相应行业实际累计超额收益

    - ![纯净行业因子与相应行业实际累计超额收益（2021年度）](https://s2.loli.net/2022/04/08/6DRpACUPYtIb4Ok.png)

- 纯因子净值走势

    - ![各纯净因子净值走势（2021年度）](https://s2.loli.net/2022/04/08/hVMLjX6WN2kS47g.png)
    - ![各纯净因子净值走势（201201-202203）](https://s2.loli.net/2022/04/08/QB6zaWquj83UGcv.png)
    - ![各纯净因子每年收益（2012-2022）](https://s2.loli.net/2022/04/08/e4IJlTtgqOFdEuV.png)
    - ![各纯因子组合日收益历年信息比率（201201-202203）](https://s2.loli.net/2022/04/08/5rE6gzjIy7NktQP.png)

- 沪深300成分股因子暴露 模拟收益vs实际收益

    - ![沪深300投资组合因子暴露度百分位（201201-202203）](https://s2.loli.net/2022/04/08/HVTwb359hcK6QvN.png)
    - ![多因子模拟组合vs.沪深300日度收益（201201-202203）](https://s2.loli.net/2022/04/08/YjWFSJxgKc6Q8Os.png)
    - ![模拟组合vs.沪深300累积收益（201201-202203）](https://s2.loli.net/2022/04/08/EBAzUwvx6aoZsFc.png)

## 参考资料

Menchero, J., & Lee, J.-H. (2015). EFFICIENTLY COMBINING MULTIPLE SOURCES OF ALPHA. *Journal of Investment Management*, *Vol. 13*(No. 4), 71–86.

韩振国. (2018). *Barra模型初探：A股市场风格解析* (“星火”多因子系列（一）) [金融工程研究]. 方正证券.

