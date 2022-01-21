"""
(created by swmao on Jan. 14th)
与mysql服务器交互的支持，参考AllSample内conn_mysql.py, create_table_rating_avg.py

"""

from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy.dialects.mysql import \
        BIGINT, BINARY, BIT, BLOB, BOOLEAN, CHAR, DATE, \
        DATETIME, DECIMAL, DECIMAL, DOUBLE, ENUM, FLOAT, INTEGER, \
        LONGBLOB, LONGTEXT, MEDIUMBLOB, MEDIUMINT, MEDIUMTEXT, NCHAR, \
        NUMERIC, NVARCHAR, REAL, SET, SMALLINT, TEXT, TIME, TIMESTAMP, \
        TINYBLOB, TINYINT, TINYTEXT, VARBINARY, VARCHAR, YEAR
import pandas as pd


def conn_mysql(eng: dict):
    """根据dict中的服务器信息，连接mysql"""
    user = eng['user']
    password = eng['password']
    host = eng['host']
    port = eng['port']
    dbname = eng['dbname']
    engine = create_engine(f'mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}?charset=UTF8MB4')
    return engine


def mysql_query(query, engine):
    """mysql接口，返回DataFrame"""
    return pd.read_sql_query(query, engine)


def create_table_pe_residual(tname='test', engine=None) -> dict:
    """Create P/E residual table, named `tname`"""
    metadata = MetaData(engine)
    new_table = Table(tname, metadata,
                      Column('id', BIGINT(20, unsigned=True), primary_key=True, nullable=False, unique=True),
                      Column('fv', DOUBLE()),
                      Column('industry', INTEGER()),
                      Column('stockcode', VARCHAR(20), nullable=False),
                      Column('tradingdate', DATE(), nullable=False),
                      )
    metadata.create_all(engine)
    dtypedict = {
        'id': BIGINT(20, unsigned=True),
        'fv': DOUBLE(),
        'industry': INTEGER(),
        'stockcode': VARCHAR(20),
        'tradingdate': DATE()
    }
    return dtypedict
