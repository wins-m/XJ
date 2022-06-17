"""
(created by swmao)
生成一些收益率

"""
import sys

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.request import get_hold_return

import yaml

conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

kind = 'ot5c'
for kind in ['otc']:
    rtn = get_hold_return(conf, ret_kind=kind, bd=None, ed=None, stk_pool='NA')
    rtn.to_csv(conf['rtnFormat'].format(kind))
