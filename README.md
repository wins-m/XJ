- ***根目录（在wsl的ubuntu环境）***
- `config.yaml` 参数、目录配置
- `main.py` 供外部访问的执行脚本
- `README.md` 说明文档


- ***文件夹./supporter/***
- `backtester.py`（未完成） 回测所需的class头文件
- `factor_operator.py` 因子统一处理，读取、中性化等
- `io.py` 磁盘文件交互
- `mysql.py` 与mysql服务器交互的支持
- `transformer.py` 转换器（中性化模块）


- ***文件夹./factor_build/***
- `factor_reformat.py` 各类因子的预处理，做成可进入回测的2D面板
- `feature_extraction_level2.py` 计算level2价量因子
- `feature_level2.xlsx` 所需更新的level2价量因子
- `pe_surprise.py` 计算一致预期PE中不能由其他指标解释的异质性部分


- ***文件夹./factor_backtest/***
- `barbybar.py` （待完成）由持仓权重回测
- `single_test.py` 现行单因子多空分组回测


- ***文件夹./data/***
- `access_target.xlsx` 所需获取的服务器表格信息
- `event_analysis.py` 事件研究，依赖于“首次研报”
- `get_data.py` 下载单项指标面板转wide格式存入本地，在[access_target](./data/access_target.xlsx)中指定
- `level2_description.jpg` lvl2快照说明
- `lv2_csv2hdf.py` lvl2数据csv到hdf格式转化，处理2021前4月，地址在config.level2_path；现使用config.lvl2.path (by zyt)
- `save_remote.py` 将本地表格long格式上传，在[config](./config.yaml)文件中指定表格文件名列表
- `tradeable.py` 获取可否交易的标签，去除新上市、停牌、涨跌停

---