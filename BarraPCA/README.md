[toc]

## BARRA模型

### 模型：截面WLS

假设资产收益由共同因子驱动：

- $r_n$可以由国家因子$f_c$，行业因子$f_i$，风格因子$f_s$表出；

- 其余为各资产特质收益，互不相关；

- 在截面内，资产收益率可以由下式表示
    $$
    r_{n, t} =
    	f_{c, t} + \sum_{i=1}^{P}{X_{ni,t-1} f_{i,t}} + \sum_{s=1}^{Q}{X_{ns,t-1} f_{s,t}} + \varepsilon_{n,t} \\
    \text{i.e.} \quad
    \bold{Y}_{t} = \bold{F}_{t} \bold{X}_{t-1} + \bold{\Epsilon}_{t}
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

### 数据：风格因子 `get_barra.py`

获取数据库中的因子 来源： [聚宽说明](https://www.joinquant.com/help/api/help#JQData:ALPHA%E7%89%B9%E8%89%B2%E5%9B%A0%E5%AD%90) BARRA CNE5 已进行去极值、标准化

- 更新时间：2005年至今，下一自然日5:00、8:00更新
- 复权方式：后复权

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
- 计算纯因子收益率：T-1期结束时的风格暴露，对应T期的资产收益，得到T期的纯因子收益预测（解释T期的资产收益）

- 时间范围：2012-01-01 ~ 2022-03-31

- 缓存/结果地址：
    - `/mnt/c/Users/Winst/Documents/data_local/BARRA/`
    
- Y: 全市场 ctc收益率（昨日Close买，今日Close卖）
    - 去除：上市 120 交易日内；昨日、今日停牌；~~昨日涨停、今日跌停~~
    
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
(base) swmao:PyCharmProject/ (master✗) $ python BarraPCA/cal_factor_return.py                  [8:55:13]

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

0607: 修改为，先填充Barra暴露（q=0.75，w=10），后计算纯因子收益（即存在相邻日期重复）

```sh
(base) swmao:PyCharmProject/ (main✗) $ python BarraPCA/cal_factor_return.py

 2022
Industry Missing 0.09 %
Return Missing 4.54 %
before (269146, 42)
after missing-drop (256915, 42)
  0%|████████████████████████████████████████████████████████████████| 58/58 [00:29<00:00,  1.96it/s]

 2021
Industry Missing 0.02 %
Return Missing 5.78 %
before (1062384, 42)
after missing-drop (1000945, 42)
100%|██████████████████████████████████████████████████████████████| 243/243 [02:27<00:00,  1.65it/s]

 2020
Industry Missing 0.42 %
Return Missing 4.60 %
before (951992, 42)
after missing-drop (908176, 42)
100%|██████████████████████████████████████████████████████████████| 243/243 [02:11<00:00,  1.85it/s]

 2019
Industry Missing 0.48 %
Return Missing 2.56 %
before (890251, 42)
after missing-drop (867473, 42)
100%|██████████████████████████████████████████████████████████████| 244/244 [02:09<00:00,  1.89it/s]

 2018
Industry Missing 0.40 %
Return Missing 7.15 %
before (857076, 41)
after missing-drop (795753, 41)
100%|██████████████████████████████████████████████████████████████| 243/243 [01:53<00:00,  2.15it/s]

 2017
Industry Missing 1.87 %
Return Missing 13.73 %
before (799553, 41)
after missing-drop (689720, 41)
100%|██████████████████████████████████████████████████████████████| 244/244 [01:41<00:00,  2.41it/s]

 2016
Industry Missing 1.34 %
Return Missing 11.88 %
before (704632, 41)
after missing-drop (620939, 41)
100%|██████████████████████████████████████████████████████████████| 244/244 [01:32<00:00,  2.64it/s]

 2015
Industry Missing 0.38 %
Return Missing 18.91 %
before (665670, 41)
after missing-drop (539775, 41)
100%|██████████████████████████████████████████████████████████████| 244/244 [01:25<00:00,  2.84it/s]

 2014
Industry Missing 0.31 %
Return Missing 10.12 %
before (620558, 41)
after missing-drop (557647, 41)
100%|██████████████████████████████████████████████████████████████| 245/245 [01:28<00:00,  2.78it/s]

 2013
Industry Missing 0.12 %
Return Missing 4.56 %
before (588055, 41)
after missing-drop (561144, 41)
100%|██████████████████████████████████████████████████████████████| 238/238 [01:24<00:00,  2.82it/s]

 2012
Industry Missing 0.41 %
Return Missing 8.39 %
before (587190, 41)
after missing-drop (537913, 41)
100%|██████████████████████████████████████████████████████████████| 243/243 [01:22<00:00,  2.93it/s]
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

## 指增基金风格归因

### `attribution.py`

**数据来源：公募基金**

- `jqdata.fund_main_info`是基金基础信息
- `jqdata.fund_net_value`是净值
- 说明：[聚宽公募基金数据(净值等)](https://www.joinquant.com/help/api/help#JQData:%E5%85%AC%E5%8B%9F%E5%9F%BA%E9%87%91%E6%95%B0%E6%8D%AE%E5%87%80%E5%80%BC%E7%AD%89)
- 由`get_barra.py`获取基金净值、基金信息（5003：指增）
    - `fund_refactor_net_value[5003].pkl`
    - `fund_main_info[5003].pkl` 

**收益归因**

- 调仓日，回看60日，用T-1纯因子解释T日收益率（T-1纯因子由T的风格暴露算得）
- 指数和指增对风格因子的暴露，指增超额暴露，超额暴露获取的超额收益，60日求和
- 指增相对指数的超额收益，60日求和
- 总超额 - 超额暴露获得超额 = 选股超额

### `backtest.py`

- 选股超额 调仓日最高X只指增 相对全体同类产品 相对指数 表现对比

## PCA全市场分解

###  `daily_pca.py`

> 时间说明：为得到t期的主成分收益/主成分暴露，需要用到截至t期的个股收益

`d120,pc60,ipo60`

- 股池：全市场

- 上市：去除新上市60日

- 日度迭代

    - 过去120天

    - 存在完整收益序列的股池

    - 日收益超过0.1，缩到0.1

    - 日收益协方差矩阵，PCA分解（SVD）
        $$
        X: D \times N  = 120 \times N , \quad 
        \bold{1}^T X = \bold{0} \\
        X^T = U \Sigma V^T \\
        \begin{cases}
        e^{D \times 1} = diag(\Sigma)^2 / 120 \\
        u^{N \times pn} = \text{First pn columns of } U \\
        \end{cases}
        $$
        主成分权重，获得主成分120日收益
        $$
        R^{D \times pn} = X u^2
        $$

    - 个股收益对主成分收益回归，得到日度截面因子暴露 $F^{N \times pn}$
        $$
        F^T = (R^T R)^{-1} R^T X
        $$

        - 对最近D期收益$X$均可用最近D期的因子收益$R$反推出资产的因子暴露$F$

- `data_local/PCA/`

    - `pc00X.csv`: 纯因子00X在个股上的暴露，截面使用
    - `PricipalCompnent/`: 所有纯因子，单日在个股上的暴露（回归系数）
    - `PrincipalFactorReturn/`: 所有纯因子，单日过去150日，因子收益
    - `PrincipalStockWeight/`: 所有纯因子，单日计算的特征向量，（平方后是）在个股上的权重

### `fof.py`

用PCA选择指增

- 结果

## 组合优化

alpha: 未来因子 收益vs约束

- T+1收盘到T+6收盘为alpha
- IC > 0 for delay = 1..5
- alpha和因子暴露无缺失作为可选股池

Beta: Barra.style & PCA因子暴露，截面标准化

**version 1**
$$
\max_{w \ge 0}{ 
	\alpha^T w - \lambda \left[
        x^T X_f F X_f^T x + x^T D x
    \right]
}, \text{ where } x = w-w_b 
\\
\text{s.t.}
\begin{cases}
\begin{gather}
& \sum{w} \le 1  \tag{1}\\
& \left(\bold{1}_{\{\text{in bench}\}}\right)^T w \ge B \tag{2}\\
& | w - w_b | \le E \tag{3}\\
& \left| X_{f=style/barra} (w - w_b) \right| \le H_0 \tag{4}\\
& \left| X_{f=indus} (w - w_b) \right| \le H_1 \tag{5}\\
& ||w_t - w_{t-1}|| \le D \tag{6}\\
& (w - w_b)^T \Sigma (w - w_b) \le S \tag{7}\\
%& \bold{1}^T \bold{1}_{\{w > 0\}} \le N_{max} \tag{8}\\
\end{gather}
\end{cases}

\\
\text{where }
\begin{cases}
\alpha : \text{alpha因子} = \text{FRtn5D(0.0,3.0)} \\
B : \text{成分占比} = {0\%} \text{ or } {80\%} \\
E : \text{权重偏离} = \max{\{1\%,\ w_b/2\}} \\
H_0 : \text{风格偏离} = {0.20} \\
H_1 : \text{行业偏离} = {0.02} \\
D : \text{换手限制} = {2} \\
\lambda : \text{风险厌恶} = 0 \\
S : \text{特异波动} = {+\infin} \\
\end{cases}
$$

**version 0**
$$
\max_{w} {
	\sum_{i} {\alpha w} - {1\over2} \gamma w' \Sigma w 
},
\\
\text{s.t.} \quad
\begin{cases}
L \le {X_{f} (w - w_{b}) } \le H \\
\sum_{i}{|w_{i,t} - w_{i, t-1}|} \le D \\
0 \le w_i \le K \\
\sum{ w_i } = 1 - \text{(weight overflow)} \\
\end{cases}
, 
\quad  \text{Let} \quad
\begin{cases}
\alpha \in \set{\text{FRtn5D}, \text{APM}} \\
\gamma=0 \\
(-)L, H \in \set{0.0, 0.05, 0.1, 0.2} \\
K \in \set{0.2\%, 0.5\%, 5\%} \\
D = 2 \\
\end{cases}
$$

### `cvxpy`求解

```python
prob.solve(verbose=False, solver='ECOS', abstol='1e-6')
```

求解器 [ECOS](https://github.com/embotech/ecos/wiki) [see.this.paper](https://web.stanford.edu/~boyd/papers/ecos.html)

优化器无解

- 原因1：约束过紧。具体为，组合内个股最大权重$K$小于指数内成分股权重，无法模拟指数
    - 个股最大权重对指数膨胀，取为指数成分股权重最大值（向上取整%）
- 原因2：迭代次数不够。ECOS到达最大迭代次数`max_iters=100`；改为，最大1000次，若未最优，则10000次
- 原因3：因子暴露缺失严重，覆盖指数成分股不足（不影响“未来因子”）

> - cvxpy 包中有多种优化器可以选择，包括内部和外部的:例如(括号内是算法详
>     细解释)
>     ECOS(https://forces.embotech.com/Documentation/), ECOS_BB(https://github.com/embotech/ecos#mixed-integer-socps-ecos_bb), OSQP(https://osqp.org/docs/solver/index.html#algorithm), SCS(https://github.com/cvxgrp/scs)，
> - 尝试去 debug 之前同事做的优化器，但是没有成功，所以自己写了优化器，目前优化器 主要有以下优化目标可以选择:
>     (1) Maximize(return)
>     (2) Minimize(gamma*risk)
>     (3) Maximum(return-gamma*risk)
>     其中 return 为预期收益率，gamma 是风险厌恶系数，risk 是风险矩阵 有以下限制条件可以选择:
>     (1) 权重之和为 1
>     (2) 个股权重不大于某固定值，如 0.5
>     (3) 所选股票组合的风格或行业因子偏离基准标准差倍数
>     (4) 所选股票组合的风格或行业因子绝对数值范围
>     (5) 换手率限制
>
> - 我自己用 python 的 cvxpy 优化包写了一个包含基准行业中性和风格中性的优化方
>     法，优化的目标函数是 Maximum(return-gamma*risk)，其中 gamma 是风险厌恶系 数，默认为 1
>     1 行业中性是指，多头组合的行业配置与对冲基准的行业配置相一致。行业中性
>     配置的目的在于剔除行业因子对策略收益的影响
>     2 风格因子中性是指，多头组合的风格因子较之对冲基准的风险暴露为0。风格
>     因子中性的意义在于，将多头组合的风格特征完全与对冲基准相匹配，使得组
>     合的超额收益不来自于某类风格
>     3 除此之外还有单只个股的权重非负且小于0.5的限制
>     在 hs300 上 2019-12-25 做优化，目标函数是 Maximum(return-risk)，用的是当日收 盘价对上一日收盘价的收益求风险矩阵来做优化，权重大于 0 共有 37 只股票


## 风险矩阵估计

$$
r_n = f_c + \sum_{i}{ X_{n,i} f_i} + \sum_{s}{X_{n, s} f_s} + \sum_{p}{X_{n, p} f_p} + u_n
$$

最小化投资组合风险；合理估计股票收益协方差矩阵
$$
Risk(Portfolio) = w^T V w \\
V = X^T F X + \Delta
$$
对T期，只能用T-1期及以前的纯因子收益估计。

### Step1：共同因子协方差矩阵 $F$

```python
mod = MFM(fr=pure_factor_return)
mod.newey_west_adj_by_time()
mod.eigen_risk_adj_by_time()
mod.vol_regime_adj_by_time()
mod.save_factor_covariance(conf['dat_path_barra'])
```

半衰指数权平均EWMA：过去252日的因子收益协方差
$$
F^{Raw}_{k,l} 
= cov(f_k, f_l)_t 
= {
    \sum_{s=t-h+1}^{t} {
        \lambda^{t-s} 
        (f_{k,s} - \bar{f_k})
        (f_{l,s} - \bar{f_l})
    }  \over 
    \sum_{s=t-h}^{t} {\lambda^{t-s}}
} \\
\text{where} \quad 
\lambda = 0.5^{1/90}, \  h=252
$$

#### 风险Newey-West调整

$$
F^{NW} =
\left[ 
    F^{Raw} 
    + \sum_{\Delta=1}^{D} {
        \left(1 - {\Delta \over D+1}\right)
        \left(C_{+\Delta}^{(d)} + C_{-\Delta}^{(d)}\right)
    }
\right] \\
C_{kl, +\Delta}^{(d)} 
= cov(f_{k, t-\Delta}, f_{l, t}) 
= {
    \sum_{s=t-h+\Delta}^{t} {
        \lambda^{t-s} 
        (f_{k,s-\Delta} - \bar{f_k})
        (f_{l,s} - \bar{f_l})
    }  \over 
    \sum_{s=t-h+\Delta}^{t} {\lambda^{t-s}}
} 
%\\
= {C_{kl, -\Delta}^{(d)}}^T
%= cov(f_{k, t}, f_{l, t-\Delta})
%= {
%    \sum_{s=t-h+\Delta}^{t} {
%        \lambda^{t-s} 
%        (f_{k,s} - \bar{f_k})
%        (f_{l,s-\Delta} - \bar{f_l})
%    }  \over 
%    \sum_{s=t-h+\Delta}^{t} {\lambda^{t-s}}
%} 
\\

\text{where} \quad D=2,\ h=252,\ \lambda = 0.5^{1/90}
$$

#### 特征值调整 Eigenfactor Risk Adjustment

$$
U_0 D_0 U_0^T = F^{NW}  \\
F^{Eigen} = U_0 \widetilde{D}_0 U_0^T
$$

- $M=10000$次蒙特卡洛模拟，第$m$次：

1. 生成 模拟特征因子收益 $b_m: N\times T$：均值$0$，方差$D_0$
2. 计算 模拟因子收益 $r_m = U_0 b_m$
3. 计算 模拟因子收益协方差（？？）$F_m^{MC}=cov(r_m, r_m)$ 【满足$E[F_m^{MC}] = F^{NW}$】
4. 执行 模拟协方差特征值分解 $U_m D_m U_m^T = F_m^{MC}$
5. 计算 模拟特征因子“真实”协方差 $\widetilde{D}_m = U_m^T F^{NW} U_m$
6. 记录 $D_m, \widetilde{D}_m$ 对角元素

- 计算 第$k$个特征的模拟风险偏差 $\lambda(k) = \sqrt{{1 \over M} \sum_{m=1}^M{\widetilde{D}_m(k) \over D_m(k)}}$
- 实际因子收益“尖峰厚尾”调整 $\gamma(k) = a[\lambda(k) -1] + 1, \text{let}\ a=1.2$
- 特征因子协方差“去偏” $\widetilde{D}_0 = \gamma^2 D_0$
- 因子协方差“去偏”调整 $F^{Eigen} = U_0 \widetilde{D}_0 U_0^T$

#### 波动率偏误调整 Volatility Regime Adjustment

$$
F^{VRA} = \lambda_F^2 F^{Eigen}
$$

- 因子波动率乘数  $\lambda_F = \sqrt{ \sum_{s=t-h+1}^{t}{w^{t-s} (B_s^F)^2}},\ w=0.5^{1/42},\ h=252$
- 日风险预测即时偏差 $B_t^F = \sqrt{{1\over K} \sum_k{\left({f_{k,t} \over \sigma_{k,t}}\right)^2}}$
- 样本外标准化收益 $b_{t,q} = {r_{t+q} / \sigma_{t}}$
    - $r_{t+q}$：时刻$t$至$t+q$时间段（去21天）内资产收益率
    - $\sigma_t$：当前时刻$t$的预测风险

### Step2：特异风险方差矩阵 $\Delta$

```python
self = SRR(fr=factor_return, sr=asset_return, expo=factor_exposure, mv=market_value)
Ret_U = self.specific_return_by_time()  # Specific Risk
Raw_var, Newey_West_adj_var = self.newey_west_adj_by_time()  # New-West Adjustment
Gamma_STR, Sigma_STR = self.struct_mod_adj_by_time()  # Structural Model Adjustment
Sigma_Shrink = self.bayesian_shrink_by_time()  # Bayesian Shrinkage Adjustment
Lambda_VRA, Sigma_VRA = self.vol_regime_adj_by_time()  # Volatility Regime Adjustment
self.save_vol_regime_adj_risk(conf['dat_path_barra'])
```

横截面回归，不由公共因子解释的残差序列
$$
\{u_{nt}\}:\ T \times N \\
u_{nt} = r_{nt} - \sum_{k}{X_{nkt} f_{kt}}
$$
#### 特质风险（收益）Newey-West 方差

EWM方差进行 Newey-West 调整（h=252, tau=90, d=5）
$$
(\sigma^{Raw}_{n})^2
= cov(u_n)_t 
= {
    \sum_{s=t-h+1}^{t} {
        \lambda^{t-s} 
        (u_{n, s} - \bar{u}_n)^2
    }  \over 
    \sum_{s=t-h}^{t} {\lambda^{t-s}}
} \\
\text{where} \quad 
\lambda = 0.5^{1/90}, \  h=252
\\
\\
(\sigma_{u}^{NW})^2 = \left[ 
    \sigma^{Raw} 
    + \sum_{\Delta=1}^{D} {
        \left(1 - {\Delta \over D+1}\right)
        \left(C_{+\Delta}^{(d)} + C_{-\Delta}^{(d)}\right)
    }
\right] \\
C_{n, +\Delta}^{(d)} 
= cov(u_{k, t-\Delta}, u_{l, t}) 
= {
    \sum_{s=t-h+\Delta}^{t} {
        \lambda^{t-s} 
        (u_{n,s-\Delta} - \bar{u}_n)
        (u_{n,s} - \bar{u}_n)
    }  \over 
    \sum_{s=t-h+\Delta}^{t} {\lambda^{t-s}}
} \\
= C_{n, -\Delta}^{(d)} 
= cov(u_{k, t}, u_{l, t-\Delta})
= {
    \sum_{s=t-h+\Delta}^{t} {
        \lambda^{t-s} 
        (u_{n,s} - \bar{u}_n)
        (u_{n,s-\Delta} - \bar{u}_n)
    }  \over 
    \sum_{s=t-h+\Delta}^{t} {\lambda^{t-s}}
} \\

\text{where} \quad D=5,\ h=252,\ \lambda = 0.5^{1/90}
$$

#### 结构化模型调整 Structural Model

个股特质收益异常值：具有相同特征的股S票可能具有相同的特质波动
$$
\hat{\sigma}_{u} = \gamma \sigma_{u}^{NW} + (1 - \gamma) \sigma_{u}^{STR} \\
\\
\sigma^{STR}_n = E_0 \times \exp{\left(\sum_{k}{X_{nk} b_{k}}\right)}\\
	\begin{aligned}
        \text{where} \quad
        & E_{0}: \text{去除残差项的指数次幂带来的偏误，取1.05}  \\
        & X_{nk}: \text{因子暴露} \\
        & b_{k}: \text{以下WLS回归拟合系数} \\
	\end{aligned} \\
\\
b_{k} \text{: WLS Reg on stocks whose } \lambda = 1 \\
\ln{\sigma_n^{NW}} = \sum_{k} X_{nk} b_{k} + \varepsilon_{n} \\
\\
\gamma = 
	\min{ \left( 1, \max{\left(0, {h-60\over120} \right)} \right)} 
	\times \min{(1, \max{(0, \exp{(1 - Z_{u})})})} \\
\begin{aligned}
	\text{where} \quad
	& h = 252, \quad Z_u = \left| {\sigma_{u, eq} - \tilde{\sigma}_{u} \over \tilde{\sigma}_{u} } \right| \\
	& \tilde{\sigma}_{u} \text{ : 特异收益稳健标准差, } \tilde{\sigma}_{u} = {1 \over 1.35} (Q_3 - Q_1) \\
	& Q_1, Q_3 \text{ : 特异收益h=252日的1/4和3/4分位数} \\
	& \sigma_{u, eq} \text{ : 特异收益}[-10 \tilde{\sigma}_{u}, 10\tilde{\sigma}_{u}]{内等权重样本标准差} \\
\end{aligned}
$$
![image-20220509112732331](https://s2.loli.net/2022/05/09/DQrZaeb8j6RXitu.png)

<center>图：特异收益数据质量较优的股票比率</center>

> 非常接近1，似乎无需进行调整？

#### 贝叶斯压缩调整 Bayesian Shrinkage

时序处理得到特异风险，高估历史高波动率股票未来风险，低估历史低波动率股票未来风险
$$
\sigma_{n}^{SH} = v_{n} \bar{\sigma}(S_n) + (1 - v_n) \hat{\sigma}_n \\
\\
\begin{aligned}
	\text{where} \quad 
	& v_{n} = { 
            q \left| \hat{\sigma}_n - \bar{\sigma}(S_n)\right|  \over 
            \Delta_{\sigma}{(S_n)} + q \left| \hat{\sigma}_n - \bar{\sigma}(S_n) \right| 
        }, \text{ where } q = 1 \\
	& \bar{\sigma}(S_n) = \sum_{n \subset S_n} {w_{n} \hat{\sigma}_n} \quad
		\text{所在市值分组（10分组）市值加权的平均风险} \\
	& \Delta_{\sigma}{(S_n)} = \sqrt{{1 \over N(S_n)} \sum_{n \subset S_n}{(\hat{\sigma}_n - \bar{\sigma}(S_n))^2}}
\end{aligned}
$$
[]

<center>图：不同波动率分组下偏误统计量</center>

#### 波动率偏误调整  Volatility Regime Adjustment

$$
\sigma_{n}^{VRA} =  \lambda_S \sigma^{SH}_{n} \\

\lambda_S = \sqrt{ \sum_{t=T-h+1}^{T}{w^{T-t} (B_t^S)^2}},\ w=0.5^{1/42},\ h=252 \\ 
B_t^S = \sqrt{ \sum_n{w_{nt}\left( u_{nt} \over \sigma_{nt} \right)^2 }},\ w_{nt} \text{ : t期股票n的市值权重}
$$

**准确性评价**

$$
CSV_t^S = \sqrt{ \sum_{n}{w_{nt} u_{nt}^2} }
$$

[CSV, lambda]

<center>图：特异风险波动乘数$\lambda_S$ V.S. 横截面波动$CSV^S$</center>

> ...

![image-20220512132509149](README.assets/image-20220512132509149.png)

<center>特异风险偏差统计量 12 个月滚动平均</center>

> $u_{nt} / \sigma_{nt}$截面等权均值21日波动；调整后应该更接近1

## 参考资料

Menchero, J., & Lee, J.-H. (2015). EFFICIENTLY COMBINING MULTIPLE SOURCES OF ALPHA. *Journal of Investment Management*, *Vol. 13*(No. 4), 71–86.

韩振国. (2018). *Barra模型初探：A股市场风格解析* (“星火”多因子系列（一）) [金融工程研究]. 方正证券.

Menchero, J., Orr, D. J., & Wang, J. (2011). *The Barra US Equity Model (USE4)*. 44.

Orr, D. J., Mashtaler, I., & Nagy, A. (2012). *The Barra China Equity Model (CNE5)*. 59.

*Research Notes: The Barra Europe Equity Model (EUE3)*. (2009).

*Barra系列（三）：风险模型之共同风险*. (不详). 知乎专栏. 取读于 2022年4月28日, 从 https://zhuanlan.zhihu.com/p/69497933

*Barra系列（四）：风险模型之异质风险*. (不详). 知乎专栏. 取读于 2022年4月28日, 从 https://zhuanlan.zhihu.com/p/73794358

#### Single-period Portfolio Selection

Markowitz's mean-variance portfolio selection

#### Online Portfolio Selection

Li, B., & Hoi, S. C. H. (2014). Online portfolio selection: A survey. *ACM Computing Surveys*, *46*(3), 35:1-35:36. https://doi.org/10.1145/2512962

#### Deep Portfolio Selection

Heaton, J. B., Polson, N. G., & Witte, J. H. (2018). *Deep Portfolio Theory* (arXiv:1605.07230). arXiv. https://doi.org/10.48550/arXiv.1605.07230

- Follow the Winner
- Follow the Loser
- Pattern Matching-based
- Meta-Learning Algorithms (MLAs)

[量化攻城狮-组合优化专题](https://mp.weixin.qq.com/s/xcT71gfa924bYe5ETNOTdQ)

[量化攻城狮-再聊组合优化的约束条件](https://mp.weixin.qq.com/s/KKbmLkOgdSPi0UaqHVDiRw)

*不同条件下的组合优化模型结果分析*. (2020). 渤海证券.

[toc]
