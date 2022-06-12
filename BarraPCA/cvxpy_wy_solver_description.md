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
