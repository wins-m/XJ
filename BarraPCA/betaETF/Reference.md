#### Single-period Portfolio Selection



#### Online Portfolio Selection

Li, B., & Hoi, S. C. H. (2014). Online portfolio selection: A survey. *ACM Computing Surveys*, *46*(3), 35:1-35:36. https://doi.org/10.1145/2512962

#### Deep Portfolio Selection

Heaton, J. B., Polson, N. G., & Witte, J. H. (2018). *Deep Portfolio Theory* (arXiv:1605.07230). arXiv. https://doi.org/10.48550/arXiv.1605.07230

##### Follow the Winner

##### Follow the Loser

##### Pattern Matching-based

##### Meta-Learning Algorithms (MLAs)

---

[量化攻城狮-组合优化专题](https://mp.weixin.qq.com/s/xcT71gfa924bYe5ETNOTdQ)

[量化攻城狮-再聊组合优化的约束条件](https://mp.weixin.qq.com/s/KKbmLkOgdSPi0UaqHVDiRw)

---

*不同条件下的组合优化模型结果分析*. (2020). 渤海证券.
$$
\max_{w}{ \alpha^T w - \lambda w^T \Sigma w} \\
\text{s.t.}
\begin{cases}
\begin{gather}
& 0 \le w  \le K  \tag{1}\\
& \sum{|w_t - w_{t-1}|} \le \delta \tag{2}\\
& \left(\bold{1}_{\{\text{in bench}\}}\right)^T w \ge B \tag{3}\\
& -E \le w - w_b \le E \tag{4}\\
& L \le X_{f} (w - w_b) \le H \tag{5}\\
& (w - w_b)^T \Sigma (w - w_b) \le \sigma^2 \tag{6}\\
& \bold{1}^T \bold{1}_{\{w > 0\}} \le N_{max} \tag{7}\\
\end{gather}
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