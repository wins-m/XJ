"""
(created by swmao on March 30th)

"""
import sys, os
import pandas as pd
import yaml

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
conf = yaml.safe_load(open(conf_path, encoding='utf-8'))


def get_barra():
    from data.get_data import transfer_data

    mysql_engine = {
        'engine0': {'user': 'intern01',
                    'password': 'rh35th',
                    'host': '192.168.1.104',
                    'port': '3306',
                    'dbname': 'alphas_jqdata'}
    }
    force_update = False
    data_pat = '/mnt/c/Users/Winst/Documents/data_local/BARRA/'
    os.makedirs(data_pat, exist_ok=True)
    access_target = '/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/barra/access_barra.xlsx'

    transfer_data(mysql_engine, data_pat, access_target, force_update)


def get_fund():
    """获取基金信息（5003：增强指数型）"""

    try:
        df = pd.read_pickle(conf['dat_path_barra'] + 'fund_main_info[5003].pkl')  # 增强指数型 公募基金 详情
        net_val = pd.DataFrame(pd.read_hdf(conf['data_path'] + 'fund_net_value.h5', key='refactor_net_value'))  # 全部公募基金 累计复权净值 复权单位净值＝单计净值＋成立以来每份累计分红派息的金额（1+涨跌幅）
    except FileNotFoundError:
        from supporter.mysql import conn_mysql, mysql_query
        engine = conn_mysql(eng={'user': 'intern01',
                                 'password': 'rh35th',
                                 'host': '192.168.1.104',
                                 'port': '3306',
                                 'dbname': 'jqdata'})

        query = f"""SELECT * FROM jqdata.fund_main_info WHERE invest_style_id=5003"""
        df = mysql_query(query, engine)
        # df.to_csv(conf['data_path'] + 'fund_main_info[5003].csv')
        df.to_pickle(conf['dat_path_barra'] + 'fund_main_info[5003].pkl')

        query = f"""SELECT code,day,refactor_net_value FROM jqdata.fund_net_value WHERE day>='2012-01-01' ORDER BY day,code"""
        net_val = mysql_query(query, engine)
        net_val = net_val.pivot('day', 'code', 'refactor_net_value')
        net_val.index = pd.to_datetime(net_val.index)
        # net_val.to_csv(conf['data_path'] + 'net_value_refactor.csv')
        # net_val.to_pickle(conf['dat_path_barra'] + 'fund_net_value.pkl')
        net_val.to_hdf(conf['data_path'] + 'fund_net_value.h5', key='refactor_net_value')

    code_intersect = net_val.columns.intersection(df.main_code)
    net_val[code_intersect].to_pickle(conf['dat_path_barra'] + 'fund_refactor_net_value[5003].pkl')


if __name__ == '__main__':
    # get_barra()
    get_fund()
