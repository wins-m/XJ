import sys, os
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
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
