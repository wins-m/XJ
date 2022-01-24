- ***根目录（在wsl的ubuntu环境）***
- `config.yaml`
- 参数、目录配置
- `README.md`
- 说明文档
- `main.py`
- 供外部访问的执行脚本


- ***文件夹./data/***
- `access_target.xlsx`
- 所需获取的服务器表格信息
- `get_data.py`
- 下载单项指标面板转wide格式存入本地，在[access_target](./data/access_target.xlsx)中指定
- `pe_surprise.py`
- 计算一致预期PE中不能由其他若干指标解释的异质性部分
- `save_remote.py`
- 将本地表格long格式上传，在[config](./config.yaml)文件中指定表格文件名列表
- `tradeable.py`
- 获取可否交易的标签，去除新上市、停牌、涨跌停


- ***文件夹./factor_backtest/***
- `single_test.py`
- 单因子多空分组回测
- `barbybar.py`
- （待完成）由持仓权重回测


- ***文件夹./supporter/***
- `factor_operator.py`
- 因子统一处理，读取、中性化等
- `factor_reformat.py`
- 各类因子的预处理，做成可进入回测的2D面板
- `mysql.py`
- 与mysql服务器交互的支持
- `neu.py`
- 中性化模块
