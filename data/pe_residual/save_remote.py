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
from data.pe_residual.auto_update import transfer_pe_residual_table


def main():
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, 'r', encoding='utf-8'))

    save_remote(conf)


def save_remote(conf):
    # engine
    eng_info = conf['mysql_engine']['engine3']  # 上传使用engine3的配置，即存在intern库
    engine = conn_mysql(eng_info)
    # csv path
    factorscsv_path = conf['factorscsv_path']
    tables = conf['tables']

    upload_all_tables(factorscsv_path, tables, engine)


def upload_all_tables(factorscsv_path: str, tables: dict, engine):
    """
    文件名指定的本地csv表格依次上传（注意内存压力，速度取决于网络）
    - 注意在服务器建表的函需要import

    """

    # pe_tables 系列的上传
    if ('pe_tables' in tables) and (tables['pe_tables'][0]):

        from supporter.mysql import create_table_pe_residual

        pe_tables = tables['pe_tables'][1:]
        # tb = pe_tables[0]
        for tb in pe_tables:
            print(tb)
            df = pd.read_csv(factorscsv_path + tb, index_col=[0, 1])
            print('TRANSFERRING...')
            df = transfer_pe_residual_table(df)
            print('UPLOADING...')
            tname = tb.replace('.csv', '')
            dtypedict = create_table_pe_residual(tname, engine)
            print('NEW TABLE CREATED')
            df.to_sql(tname, con=engine, if_exists='replace', index=False, dtype=dtypedict)
            print('UPLOADED')

    if ('efr_tables' in tables) and (tables['efr_tables'][0]):

        from supporter.mysql import create_table_efr

        all_tables = tables['efr_tables'][1:]
        for tb in all_tables:
            print(tb)
            df = pd.read_csv(factorscsv_path + tb)
            tname = tb.replace('.csv', '').lower()
            dtypedict = create_table_efr(tname, engine)
            print('UPLOADING...')
            df.to_sql(tname, con=engine, if_exists='replace', index=False, dtype=dtypedict)
            print('UPLOADED')


# %%
if __name__ == '__main__':
    main()
