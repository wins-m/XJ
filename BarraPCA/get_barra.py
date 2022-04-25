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
    access_target = '/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/BarraPCA/access_barra.xlsx'

    transfer_data(mysql_engine, data_pat, access_target, force_update)


def get_fund():
    """获取基金信息（5003：增强指数型）"""
    try:
        df: pd.DataFrame = pd.read_pickle(conf['dat_path_barra'] + 'fund_main_info[5003].pkl')  # 增强指数型 公募基金 详情
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
    df[df.main_code.apply(lambda x: x in code_intersect)].astype(object).to_excel(
        conf['dat_path_barra'] + 'fund_5003_info_intersect.xlsx', index=None, encoding='GBK')


def split():
    fund_info = pd.read_pickle(conf['fund_info_5003'])  # 基金信息
    fund_val = pd.read_pickle(conf['refactor_net_value_5003'])  # 资产收益
    fund_rtn: pd.DataFrame = fund_val.pct_change()  # 累计复权净值，盘前9:00更新

    # 基金属性-基准划分
    desc = fund_rtn.describe().astype(object).T
    desc['main_code'] = desc.index
    desc = desc.merge(fund_info, on='main_code', how='left')
    desc.to_excel(conf['dat_path_barra'] + 'fund_stat.xlsx', index=None)
    attr_300 = desc[desc['name'].apply(lambda x: '300' in x)]
    attr_300.to_excel(conf['dat_path_barra'] + 'fund_stat[300].xlsx', index=None)
    attr_500 = desc[desc['name'].apply(lambda x: '500' in x)]
    attr_500.to_excel(conf['dat_path_barra'] + 'fund_stat[500].xlsx', index=None)
    attr_1000 = desc[desc['name'].apply(lambda x: '1000' in x)]
    attr_1000.to_excel(conf['dat_path_barra'] + 'fund_stat[1000].xlsx', index=None)


if __name__ == '__main__':
    # get_barra()
    # get_fund()
    # split()
    pass
