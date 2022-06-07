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

---

[量化攻城狮-组合优化专题](https://mp.weixin.qq.com/s/xcT71gfa924bYe5ETNOTdQ)

[量化攻城狮-再聊组合优化的约束条件](https://mp.weixin.qq.com/s/KKbmLkOgdSPi0UaqHVDiRw)

---

#### *不同条件下的组合优化模型结果分析*. (2020). 渤海证券.

$$
\max_{w\ge0}{ \alpha^T w 
- \lambda w^T \Sigma w
} \\
\text{s.t.}
\begin{cases}
\begin{gather}
& \sum{w} \le 1  \tag{1}\\
& \left(\bold{1}_{\{\text{in bench}\}}\right)^T w \ge B \tag{2}\\
& | w - w_b | \le E \tag{3}\\
& \left| X_{style/barra} (w - w_b) \right| \le H_0 \tag{4}\\
& \left| X_{indus} (w - w_b) \right| \le H_1 \tag{5}\\
& ||w_t - w_{t-1}|| \le D \tag{6}\\
& (w - w_b)^T \Sigma (w - w_b) \le S \tag{7}\\
%& \bold{1}^T \bold{1}_{\{w > 0\}} \le N_{max} \tag{8}\\
\end{gather}
\end{cases}

\\
\text{where }
\begin{cases}
B : \text{成分股占比} = {0\%} \\
E : \text{个股权重对指数的偏离} = \max{\{1\%,\ w_b/2\}} \\
H_0 : \text{style/pca因子暴露对指数的偏离} = {0.20} \\
H_1 : \text{indus行业暴露对指数的偏离} = {0.02} \\
D : \text{周度调仓时最大换手率} = {2} \\
\lambda : \text{风险厌恶水平} = 0 \\
S : \text{特异性波动率限制} \in \set{+\infin} \\
\end{cases}
$$

基本限制条件为：

1. lambda 系数为 10
2. 不设置换手率限制
3. 全部在成份股中进行选择
4. 权重偏离度1.5%
5. 行业中性
6. 市值因子中性
7. 不设置跟踪误差限制
8. 以沪深 300 为 基准时，最大股票数量为 150；以中证 500 为基准时，最大股票数量为 300。

若测试相应的限制条件，则在基本规则基础上修改相应的限制条件。

> 印花税为 0.1%，佣金为 0.02%，买入冲击成 本为 0.15%，卖出冲击成本为 0.3%

- 个股权重偏离度：(T)成分股权重上下1% // (L)成分股行业权重
- 累积收益；月度胜率（对基准）；年度收益；年化超额收益；跟踪误差；换手率
- lambda = [0, 20, 40, 60, 80, 100] 
- 成分股权重限制 [1., .95, .90, .85, .80]
- 行业风险敞口 [.03, .06, .09]
- 换手率 无 // 月换手小于30%
- 市值中性 // -1 ~ 1
- 跟踪误差？？
- 股票数量：0.1%最小 $$x - y \le 0,\ \text{sum}(y) \le n_{max}$$ 两个变量同时优化
- 