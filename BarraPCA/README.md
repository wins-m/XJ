[toc]

## BARRA模型

### 纯因子收益：截面WLS回归

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

**数据来源**

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

## 风险矩阵估计 $V = X^T F X + \Delta$

$$
r_n = f_c + \sum_{i}{ X_{n,i} f_i} + \sum_{s}{X_{n, s} f_s} + \sum_{p}{X_{n, p} f_p} + u_n
$$

最小化投资组合风险
$$
Risk(Portfolio) = w^T V w
$$
合理估计股票收益协方差矩阵
$$
V = X^T F X + \Delta
$$
对T期，只能用T-1期及以前的纯因子收益估计。

#### Step1：共同因子协方差矩阵 $F$

```python
mod = MFM(fr=pure_factor_return)
mod.newey_west_adj_by_time()
mod.eigen_risk_adj_by_time()
mod.vol_regime_adj_by_time()
mod.save_vol_regime_adj_cov(conf['dat_path_barra'])
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

##### 风险Newey-West调整

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
} \\
= C_{kl, -\Delta}^{(d)} 
= cov(f_{k, t}, f_{l, t-\Delta})
= {
    \sum_{s=t-h+\Delta}^{t} {
        \lambda^{t-s} 
        (f_{k,s} - \bar{f_k})
        (f_{l,s-\Delta} - \bar{f_l})
    }  \over 
    \sum_{s=t-h+\Delta}^{t} {\lambda^{t-s}}
} \\

\text{where} \quad D=2,\ h=252,\ \lambda = 0.5^{1/90}
$$

> 只能调整成月度？——不×21

##### 特征值调整 Eigenfactor Risk Adjustment

$$
U_0 D_0 U_0^T = F^{NW}
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

##### 波动率偏误调整 Volatility Regime Adjustment

$$
F^{VRA} = \lambda_F^2 F^{Eigen}
$$

- 因子波动率乘数  $\lambda_F = \sqrt{ \sum_{s=t-h+1}^{t}{w^{t-s} (B_s^F)^2}},\ w=0.5^{1/42},\ h=252$
- 日风险预测即时偏差 $B_t^F = \sqrt{{1\over K} \sum_k{\left({f_{k,t} \over \sigma_{k,t}}\right)^2}}$
- 样本外标准化收益 $b_{t,q} = {r_{t+q} / \sigma_{t}}$
    - $r_{t+q}$：时刻$t$至$t+q$时间段（去21天）内资产收益率
    - $\sigma_t$：当前时刻$t$的预测风险

#### Step2：特异风险方差矩阵 $\Delta$

横截面回归，不由公共因子解释的残差序列
$$
\{u_{nt}\}:\ T \times N \\
u_{nt} = r_{nt} - \sum_{k}{X_{nkt} f_{kt}}
$$

##### 特质风险（收益）Newey-West 方差

EWM方差进行Newey-West 调整（h=252, tau=90, d=5）
$$
\sigma^{Raw}_{n} 
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
\sigma_{u}^{NW} = \left[ 
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

##### 结构化模型调整 Structural Model

个股特质收益异常值：具有相同特征的股票可能具有相同的特质波动
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
	& Q_3, Q_1 \text{ : 特异收益h=252日的1/4和3/4分位数} \\
	& \sigma_{u, eq} \text{ : 特异收益}[-10 \tilde{\sigma}_{u}, 10\tilde{\sigma}_{u}]{内等权重样本标准差} \\
\end{aligned}
$$
图：特异收益数据质量较优（$\gamma = 1$）股票比率

[]

均值？

##### 贝叶斯压缩调整 Bayesian Shrinkage

时序处理得到特异风险，高估历史高波动率股票，低估历史低波动率股票
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
图：不同波动率分组下偏误统计量

[Raw, NW, SM, Shrink, VRA]

##### 波动率偏误调整  Volatility Regime Adjustment

$$
\sigma_{n}^{VRA} =  \lambda_S \sigma^{SH}_{n} \\

\lambda_S = \sqrt{ \sum_{t=T-h+1}^{T}{w^{T-t} (B_t^S)^2}},\ w=0.5^{1/42},\ h=252 \\ 
B_t^S = \sqrt{ \sum_n{w_{nt}\left( u_{nt} \over \sigma_{nt} \right)^2 }},\ w_{nt} \text{ : t期股票n的市值权重}
$$

图：特异风险波动乘数$\lambda_S$ V.S. 横截面波动$CSV^S$

[CSV, lambda]
$$
CSV_t^S = \sqrt{ \sum_{n}{w_{nt} u_{nt}^2} }
$$
图：特质风险偏差统计量12个月滚动平均

[Shrink, VRA]

## 组合优化

$$
\max_{w} {
	\sum_{i} {\alpha w} - {1\over2} \gamma w' \Sigma w 
},\ \gamma=0  \\
\text{s.t.} \quad
\begin{cases}
F_l \le {X_{f} (w - w_{b}) } \le F_h \\
H_l \le {H (w - w_{b}) } \le H_h \\
P_l \le {P(w - w_b)} \le P_h \\
\sum_{i}{|w_{i,t} - w_{i, t-1}|} < d \\
0 \le w_i \le k \\
\sum{ w_i } = 1 \\
\end{cases}
$$



## 参考资料

Menchero, J., & Lee, J.-H. (2015). EFFICIENTLY COMBINING MULTIPLE SOURCES OF ALPHA. *Journal of Investment Management*, *Vol. 13*(No. 4), 71–86.

韩振国. (2018). *Barra模型初探：A股市场风格解析* (“星火”多因子系列（一）) [金融工程研究]. 方正证券.

[toc]
