- ***根目录（在wsl的ubuntu环境）***
- `config.yaml` 参数、目录配置
- `config2.yaml` swmao macOS下的参数、目录配置，由./setup/config_transfer.py转换
- `main.py` 供外部访问的执行脚本
- `README.md` 目录说明文档（即此文档）


- ***文件夹./supporter/***
- `factor_operator.py` 因子统一处理，读取、中性化等
- `io.py` 磁盘文件交互
- `mysql.py` 与mysql服务器交互的支持
- `transformer.py` 转换器（中性化模块）


- ***文件夹./setup/***
- `auto_update_pe_residual.py` 自动更新
- `config_transfer.py` 配置文件转换


- ***文件夹./res_analysis/***
- `net_value_compare.py` 回测结果的比较，生成图像
- `res_long_compare.py` 几组回测的半年指标合并对比


- ***文件夹./factor_build/***
- `factor_reformat.py` 各类因子的预处理，做成可进入回测的2D面板
- `feature_extraction_level2.py` 计算level2价量因子
- `feature_level2.xlsx` 所需更新的level2价量因子
- `pe_surprise.py` 计算一致预期PE中不能由其他指标解释的异质性部分
- `turnover.py` 生成换手率（5日）因子


- ***文件夹./factor_backtest/***
- `backtester.py` 回测所需的class头文件
- `barbybar.py` （作废）
- `factor_tested.xlsx` 回测目标（因子等）指定
- `single_test.py` 现行单因子多空分组回测


- ***文件夹./Docs/***
- `all.notes.md` 周度工作记录
- `database_info.md` 数据库说明（部分）
- `explanatory_variables.md` pe_residual的解释变量说明


- ***文件夹./data/***
- `access_target.xlsx` 所需获取的服务器表格信息
- `auto_update.py` 增量更新pe_residual到数据库，镜像到文件夹./setup
- `get_data.py` 下载单项指标面板转wide格式存入本地，在[access_target](./data/access_target.xlsx)中指定
- `level2_description.jpg` lvl2快照说明
- `lv2_csv2hdf.py` lvl2数据csv到hdf格式转化，处理2021前4月，地址在config.level2_path；现使用config.lvl2.path (by zyt)
- `save_remote.py` 将本地表格long格式上传，在[config](./config.yaml)文件中指定表格文件名列表
- `tradeable.py` 获取可否交易的标签，去除新上市、停牌、涨跌停
- ***子文件夹./event_first_report***
- `event_analysis.py` 最初的事件研究，依赖于“首次研报”，生成AR-CAR图和相关性渲染图
- `event_condition.py` 由因子面板进行筛选形成信号因子更新回测目标xlsx的相关函数
- `prepare_panel.py` 从数据库的原始文件，准备事件面板
- `README.md` 事件研究说明（2.9后未补充）
- `regAR3~x.md` CAR3归因的回归结果


- ***文件夹./AllSample/***
- `conn_mysql.py` 从mysql服务器获取dataframe
- `create_table_rating_avg.py` 建表存表到数据库
- `resultDemo.jpg` 回测结果示例
- `template.py` 项目module模板
- `update_idx_price.py` 更新数据库表的示例

---