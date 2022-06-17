- ***根目录（在wsl的ubuntu环境）***
- `README.md` 目录说明文档（即此文档）
- `config.yaml` 参数、目录配置
- `config2.yaml` swmao macOS下的参数、目录配置，由./setup/config_transfer.py转换
- `main.py` 供外部访问的执行脚本


- ***文件夹./AllSample/***
- `argparse.example.py` 终端传参示例
- `conn_mysql.py` 从mysql服务器获取dataframe
- `create_table_rating_avg.py` 建表存表到数据库
- `imap.py` 多进程进度条
- `resultDemo.jpg` 回测结果示例
- `template.py` 项目module模板
- `tensorflow.py` 张量菜鸟语法
- `update_idx_price.py` 更新数据库表的示例


- ***文件夹./BarraPCA/***
- `00.py` 统计收益率分布
- `README.md` Barra和PCA收益分解、风格归因、组合优化文档
- `Reference.md` 组合优化参考（部分）
- `access_barra.xlsx` 数据库代获取的barra风格暴露
- `attribution.py` 指增产品收益归因
- `backtest.py` 指增选择的回测
- `cal_factor_return.py` 计算纯因子收益
- `cov_adjust.py` 方差矩阵估计
- `daily_pca.py` 收益PCA分解
- `fof.py` PCA选择指增
- `get_barra.py` 从数据库获取风格暴露
- `opt_res_ana.py` 由持仓分析组合表现
- `optimize.py` 组合优化
- `optimize_target_v2.xlsx` 组合优化目标
- ***子文件夹./HoldingStatistics/***
- `figures/` 指增持股权重分布
- `src/` Wind持股明细导出
- `tgt/` 持股权重，与指数对比
- `config.yaml` 配置
- `main.py` 批量分析指增持股
- `stat.py` 成分股持有量统计结果


- ***文件夹./alpha101/***
- `RidgeReg.py` 因子组合选股
- `prepare_input_local.py` 准备训练数据
- `single_train.py` 空，见ipynb


- ***文件夹./data/***
- `README.md` 数据库说明
- `access_target.xlsx` 所需获取的服务器表格信息
- `get_data.py` 下载指标转wide存本地，在`access_target.xlsx`中指定
- `hold_returns.py` 
- `tradeable.py` 获取可否交易的标签，去除新上市、停牌、涨跌停
- ***子文件夹./event_first_report/***
- `README.md` 事件研究说明（2.9后未补充）
- `event_analysis.py` 最初的事件研究，依赖于“首次研报”，生成AR-CAR图和相关性渲染图
- `event_condition.py` 由因子面板进行筛选形成信号因子更新回测目标xlsx的相关函数
- `prepare_panel.py` 从数据库的原始文件，准备事件面板
- `regAR3~x.md` CAR3归因的回归结果
- ***子文件夹./level2/***
- `level2_description.jpg` lvl2快照说明
- `lv2_csv2hdf.py` lvl2数据csv到hdf格式转化，处理2021前4月，地址在config.level2_path；现使用config.lvl2.path (by zyt)
- ***子文件夹./pe_residual/***
- `auto_update.py` 增量更新pe_residual到数据库，镜像到文件夹./setup
- `pe_surprise.py` 全量计算一致预期PE中不能由其他指标解释的异质性部分
- `save_remote.py` 将本地表格wide2long后上传，在[config](./config.yaml)文件中指定表格文件名列表


- ***文件夹./factor_backtest/***
- `barbybar.py` （作废）
- `event_test.py` 计算胜率
- `factor_tested.xlsx` 回测目标（因子等）指定
- `single_test.py` 现行单因子多空分组回测


- ***文件夹./factor_build/***
- `factor_reformat.py` 各类因子的预处理，做成可进入回测的2D面板
- `feature_extraction_level2.py` 计算level2价量因子
- `feature_level2.xlsx` 所需更新的level2价量因子
- `turnover.py` 生成换手率（5日）因子


- ***文件夹./res_analysis/***
- `net_value_compare.py` 回测结果的比较，生成图像
- `res_long_compare.py` 几组回测的半年指标合并对比


- ***文件夹./setup/***
- `auto_update_pe_residual.py` 自动更新
- `config_transfer.py` 配置文件转换
- `weight2efr.py` 更新首次研报持仓（服务器+本地）


- ***文件夹./supporter/***
- `alpha.py` 
- `backtester.py` 回测所需的class头文件
- `beta_etf.py` 
- `cov_a.py` 
- `factor_operator.py` 因子统一处理，读取、中性化等
- `io.py` 磁盘文件交互
- `mysql.py` 与mysql服务器交互的支持
- `pipeline.py` Factor类，处理因子，计算IC
- `request.py` 
- `stk_pool.py` 
- `transformer.py` 转换器（中性化模块）


---
