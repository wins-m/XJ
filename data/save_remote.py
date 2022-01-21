"""
(created by swmao on Jan. 14th)
# `save_remote.py`
将本地表格上传，注意在config文件中指定表格文件名列表
- 表格由本地的wide转成long
- 增加在服务器建表（指定格式），建表函数写在`supporter.mysql`
- 列名主要为`tradingdate`, `stockcode`, `fv`, `id`
  - （可精简）对于`pe_residual*.csv`，包括`industry`，由于同一股票在时间段内风格大类可能变化
- 去除空值后上传

"""
import pandas as pd
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.mysql import conn_mysql


def add_id_column(df: pd.DataFrame, col0: str = 'tradingdate', col1: str = 'stockcode') -> pd.DataFrame:
    """由 col0: %Y-%m-%d, col1: 123456.XY 生成行id"""
    df['id'] = df[col0].apply(lambda s: s.replace('-', '')) + df[col1].apply(lambda s: s[:6])
    return df.sort_values('id').reset_index()


def transfer_pe_residual_table(df: pd.DataFrame) -> pd.DataFrame:
    """处理pe_residual*.csv"""
    df1 = df.T.unstack().reset_index().rename(columns={'level_0': 'industry', 'level_1': 'stockcode',
                                                       'level_2': 'tradingdate', 0: 'fv'})
    df1 = df1.dropna()
    df1['industry'] = df1['industry'].astype(int)
    df1 = add_id_column(df1)
    return df1[['tradingdate', 'stockcode', 'industry', 'fv', 'id']]


def upload_all_tables(factorscsv_path: str, tables: dict, engine):
    """
    文件名指定的本地csv表格依次上传（注意内存压力，速度取决于网络）
    - 注意在服务器建表的函需要import

    """

    if ('pe_tables' in tables) and (tables['pe_tables'][0]):

        from supporter.mysql import create_table_pe_residual

        pe_tables = tables['pe_tables'][1:]
        tb = pe_tables[0]
        for tb in pe_tables:
            print(tb)
            df = pd.read_csv(factorscsv_path + tb, index_col=[0, 1])
            print('LOADED, TRANSFERRING...', end='\t')
            df = transfer_pe_residual_table(df)
            print('TRANSFER, UPLOADING...', end='\t')
            tname = tb.replace('.csv', '')
            dtypedict = create_table_pe_residual(tname, engine)
            print('NEW TABLE CREATED', end='\t')
            df.to_sql(tname, con=engine, if_exists='replace', index=False, dtype=dtypedict)
            print('UPLOADED')


def save_remote(conf):
    # engine
    eng_info = conf['mysql_engine']['engine3']  # 上传使用engine3的配置，即存在intern库
    engine = conn_mysql(eng_info)
    # csv path
    factorscsv_path = conf['factorscsv_path']
    tables = conf['tables']
    upload_all_tables(factorscsv_path, tables, engine)


if __name__ == '__main__':
    # conf
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, 'r', encoding='utf-8'))
    save_remote(conf)
