from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy.dialects.mysql import \
        BIGINT, BINARY, BIT, BLOB, BOOLEAN, CHAR, DATE, \
        DATETIME, DECIMAL, DECIMAL, DOUBLE, ENUM, FLOAT, INTEGER, \
        LONGBLOB, LONGTEXT, MEDIUMBLOB, MEDIUMINT, MEDIUMTEXT, NCHAR, \
        NUMERIC, NVARCHAR, REAL, SET, SMALLINT, TEXT, TIME, TIMESTAMP, \
        TINYBLOB, TINYINT, TINYTEXT, VARBINARY, VARCHAR, YEAR


def create_table_rating_avg(username, password, dbname, tname):
    engine = create_engine(
        'mysql+pymysql://{}:{}@{}/{}?charset=UTF8MB4'.format(Const.username.value, Const.password.value,
                                                             Const.url.value, dbname))

    metadata = MetaData(engine)

    new_table = Table(tname, metadata,
                      Column('id', BIGINT(20, unsigned=True), primary_key=True),
                      Column('fv', DOUBLE()),
                      Column('stockcode', VARCHAR(20), nullable=False),
                      Column('stockname', VARCHAR(20)),
                      Column('tradingdate', DATE(), nullable=False),
                      )
    metadata.create_all(engine)
