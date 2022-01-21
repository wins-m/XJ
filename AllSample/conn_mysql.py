import pandas as pd
from sqlalchemy import create_engine


def conn_mysql(dbname):
    """
    连接本地MySQL数据库
    dbname:数据库名称，string类型
    """
    engine = create_engine('mysql+pymysql://intern01:rh35th@192.168.1.104:3306/'+dbname+'?charset=UTF8MB4')
    print('MySQL连接成功..')
    return engine


Data = pd.read_sql_query("select tradingdate from jeffdatabase.tdays_d order by tradingdate desc limit 0,1", engine)