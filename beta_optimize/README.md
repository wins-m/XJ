# Barra归因、风险调整、组合优化

**运行方式**

- 每日10:00后，数据库中Barra风格因子暴露更新后，运行`barra_factor_model.py`，增量计算上一日（多日）的
    - 正交后的Barra风格暴露面板：上传到`barra_exposure_orthogonal`
    - 至上一日为止的纯因子收益率（日度）：上传到`barra_pure_factor_return`
    - 调整后的纯因子协方差：上传到`barra_factor_cov_nw_eigen_vra`
    - 调整后的个股异质性波动（标准差）：上传到`barra_specific_risk_nw_sm_sh_vra`
- 每周末，先通过`barra_factor_model.py`获得截止到周五的纯因子收益和风险项
    - 在`optimize_target_v2.xlsx`中指定组合优化的alpha、约束条件
    - 运行`opt.py`
    - 查看`_PATH`所指定目录下`factor_res`内的组合优化结果，包括各期持仓、优化信息

[toc]

## `barra_factor_model.py`

- `access_target.xlsx` 需获取的Barra公共因子数据库表格信息

### 模型：Barra风格归因

假设资产收益由共同因子驱动：

- $r_n$可以由国家因子$f_c$，行业因子$f_i$，风格因子$f_s$表出；

    - 行业因子

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

### 数据：（日度）因子暴露、收盘收益率、市值

获取数据库中的因子 

- 来源： [聚宽说明](https://www.joinquant.com/help/api/help#JQData:ALPHA%E7%89%B9%E8%89%B2%E5%9B%A0%E5%AD%90) BARRA CNE5 已进行去极值、标准化
- 更新时间：2005年至今，下一自然日5:00、8:00更新
- 复权方式：后复权

| 因子 code           | 因子名称   | 简介                                                         |
| :------------------ | :--------- | :----------------------------------------------------------- |
| size                | 市值       | 捕捉大盘股和小盘股之间的收益差异                             |
| beta                | 贝塔       | 表征股票相对于市场的波动敏感度                               |
| momentum            | 动量       | 描述了过去两年里相对强势的股票与弱势股票之间的差异           |
| residual_volatility | 残差波动率 | 解释了剥离了市场风险后的波动率高低产生的收益率差异           |
| non_linear_size     | 非线性市值 | 描述了无法由规模因子解释的但与规模有关的收益差异，通常代表中盘股 |
| book_to_price_ratio | 账面市值比 | 描述了股票估值高低不同而产生的收益差异, 即价值因子           |
| liquidity           | 流动性     | 解释了由股票相对的交易活跃度不同而产生的收益率差异           |
| earnings_yield      | 盈利能力   | 描述了由盈利收益导致的收益差异                               |
| growth              | 成长       | 描述了对销售或盈利增长预期不同而产生的收益差异               |
| leverage            | 杠杆       | 描述了高杠杆股票与低杠杆股票之间的收益差异                   |

**聚宽因子数据处理说明（摘自聚宽）**

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

### 运行：纯因子收益率

- 计算纯因子收益率：T-1期结束时的风格暴露，对应T期的资产收益，得到T期的纯因子收益预测（解释T期的资产收益）

- Y: 全市场 ctc收益率（昨日Close买，今日Close卖）

    - ~~去除：上市 120 交易日内；昨日、今日停牌；昨日涨停、今日跌停~~
    - 超过11%和-11%缩尾到11%
    - 可在函数`load_asset_return`中做其他调整

- X: 国家(1) 风格(10) 行业(29 or 30)

    - 风格因子日截面，对行业正交（size）or 对行业和size正交（其他9个）
    - 行业选用中信一级`indus_citic`29或30个 2019.12.2开始30个（新增“综合金融”）
    - 正交化后的面板上传到数据库`barra_exposure_orthogonal`

- WLS（Menchero & Lee, 2015)

    - $$
        \bold{\Omega} = \bold{R} (\bold{R}^T \bold{X}^T \bold{V}
        \bold{X} \bold{R})^{-1} \bold{R}^T \bold{X}^T \bold{V} \\
        
        \bold{F}_{K \times 1} = \bold{\Omega}_{K \times N} \bold{Y}_{N \times 1}
        $$

- 纯因子收益率计算结果上传到`barra_pure_factor_return`

### 模型：风险矩阵估计

$$
r_n = f_c + \sum_{i}{ X_{n,i} f_i} + \sum_{s}{X_{n, s} f_s} + \sum_{p}{X_{n, p} f_p} + u_n
$$

最小化投资组合风险；合理估计股票收益协方差矩阵：由共同因子协方差及个股特质性波动表示
$$
Risk(Portfolio) = w^T V w \\
V = X^T F X + \Delta
$$
对T期，只能用T-1期及以前的纯因子收益估计。

#### Step1：共同因子协方差矩阵 $F$

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

##### 特征值调整 Eigenfactor Risk Adjustment

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

##### 结构化模型调整 Structural Model

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

##### 贝叶斯压缩调整 Bayesian Shrinkage

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

##### 波动率偏误调整  Volatility Regime Adjustment

$$
\sigma_{n}^{VRA} =  \lambda_S \sigma^{SH}_{n} \\

\lambda_S = \sqrt{ \sum_{t=T-h+1}^{T}{w^{T-t} (B_t^S)^2}},\ w=0.5^{1/42},\ h=252 \\ 
B_t^S = \sqrt{ \sum_n{w_{nt}\left( u_{nt} \over \sigma_{nt} \right)^2 }},\ w_{nt} \text{ : t期股票n的市值权重}
$$

### 运行：纯因子协方差和个股特质性波动

- Barra CNE5 模型参数，参考了：韩振国. (2018). *Barra模型进阶：多因子模型风险预测* (金融工程研究 （二）; “星火”多因子系列). 方正证券.
- 代码中，类`MFM`和`SRR`完成了上述调整
- 结果上传到数据库表格`barra_factor_cov_nw_eigen_vra`, `barra_specific_risk_nw_sm_sh_vra`
    - 若需改动表名：1）全局搜索修改；2）组合优化`opt.py`中注意作相应表名修改
    - 若数据库从`intern`换到别处，首次运行时，`MFM().upload_adjusted_covariance  (` 和 `SRR().upload_asset_specific_risk(`中改动`eng`和`how`参数

## `opt.py`

### 优化问题

> - $\alpha$: 截面因子值，分布将会调整为$N(0, (2.25\%)^2)$
> - $w$: 权重，在资产上总和为1
> - $\lambda$: 风险厌恶系数，实测需要较大数量级（1e4）使得风险项发挥作用4
> - $X_f$: 风格暴露
> - $F$: 纯因子收益率协方差
> - $D$: 特异性风险矩阵

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
B : \text{成分占比} = {0\%} \\
E : \text{权重偏离} = \max{\{0.5\%\text{ or }1.5\%,\ w_b/2\}} \\
H_0 : \text{风格偏离} = {0.20} \\
H_1 : \text{行业偏离} = {0.02} \\
D : \text{换手限制} = {2} \\
\lambda : \text{风险厌恶} \in {1.0e4, 2.5e4} \\
S : \text{特异波动} = {+\infin} \\
\end{cases}
$$

### 在`optimize_target_v2.xlsx`中设置优化边界条件

- | run  | alpha_name           | mkt_type | beta_kind | beta_suffix | beta_args                        | H0   | H1   | B    | E    | D    | G    | S    | N    | wei_tole | begin_date | end_date | opt_verbose |
    | ---- | -------------------- | -------- | --------- | ----------- | -------------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | -------- | ---------- | -------- | ----------- |
    | 1    | factor_apm           | CSI500   | Barra     | barra3      | (['size', 'beta', 'momentum'],)  | 0.20 | 0.02 | 0    | 0.5  | 2    | 2.5  | inf  | inf  | 1e-3     | 2020-02-01 | NA       | FALSE       |
    | 1    | factor_apm           | CSI300   | Barra     | barra3      | (['size', 'beta',  'momentum'],) | 0.20 | 0.02 | 0    | 1.5  | 2    | 2.5  | inf  | inf  | 1e-3     | 2020-02-01 | NA       | FALSE       |
    | 0    | 数据库表格（因子）名 | CSI500   | Barra     | barra3      | (['size', 'beta', 'momentum'],)  | 0.20 | 0.02 | 0    | 0.5  | 2    | 2.5  | inf  | inf  | 1e-3     | 2020-02-01 | NA       | FALSE       |

    - run：1则进行改行的组合优化，0不运行
    - alpha_name：需同数据库因子名一致，会从数据库获取因子值整理成二维表格
    - mkt_type: 支持CSI500或CSI300，其他市场需要去代码中增加if-else分支
    - beta_kind: 只支持Barra
    - beta_suffix: 指定不同的后缀，防止覆盖之前的同名结果
    - beta_args: 对于Barra，可以增加style factor，只对此处指定的超额暴露设定上下阈值
    - H0: 超出指数的风格偏离上下界。全A股截面风格暴露分布$N(0,1)$
    - H1: 超出指数的行业偏离上下界。29/30个行业dummies取 0 or 1
    - B: 指数成分股在选股组合中的权重占比
    - E: 相对指数的个股成分偏离
    - D: 相对上期的换手率限制（双边），最大为2
    - G: 风险厌恶系数$\lambda$，单位1e4
    - S: 特异波动限制，当优化目标含有风险惩罚项时，取$\infin$
    - N: 可选股池量，每次优化，在alpha排前N的个股中选择
    - wei_tole: 最小持仓权重，少于该值的个股权重将被化为0
    - begin_date: 优化开始日期
    - end_date: 设为FALSE，优化到最新日期（最后的周五for周度）
    - opt_verbose: 运行时是否print历次优化器过程

### 运行与代码内配置

- `_PATH` 项目根目录，包括其下的优化参数设置表格`optimize_target_v2.xlsx` ；自动生成alpha面板目录`factors_csv`和运行结果目录`factors_res`
- `main() - conf` 运行配置，具体见代码注释
- `main() `内最后一行`optimize(` 内参数 `process_num`，需计算多组优化参数时（对应xlsx中多行），可指定进程数量>1

## 参考资料

Menchero, J., & Lee, J.-H. (2015). EFFICIENTLY COMBINING MULTIPLE SOURCES OF ALPHA. *Journal of Investment Management*, *Vol. 13*(No. 4), 71–86.

韩振国. (2018). *Barra模型初探：A股市场风格解析* (“星火”多因子系列（一）) [金融工程研究]. 方正证券.

Menchero, J., Orr, D. J., & Wang, J. (2011). *The Barra US Equity Model (USE4)*. 44.

Orr, D. J., Mashtaler, I., & Nagy, A. (2012). *The Barra China Equity Model (CNE5)*. 59.

*Research Notes: The Barra Europe Equity Model (EUE3)*. (2009).

*Barra系列（三）：风险模型之共同风险*. (不详). 知乎专栏. 取读于 2022年4月28日, 从 https://zhuanlan.zhihu.com/p/69497933

*Barra系列（四）：风险模型之异质风险*. (不详). 知乎专栏. 取读于 2022年4月28日, 从 https://zhuanlan.zhihu.com/p/73794358

*不同条件下的组合优化模型结果分析*. (2020). 渤海证券.

---

> swmao@XJ, 2022.8