import pandas as pd
import numpy as np
import os, sys, time
from tqdm import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.factor_operator import read_single_factor

# %%
if __name__ == '__main__':
    # %%
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    csv_path = conf['factorscsv_path']

    # %%
    fv_turnover = read_single_factor(conf['turnover'])
    fv_turnover5D = (fv_turnover * -1).rolling(5).mean()
    fv_turnover5D.to_csv(csv_path + 'turnover5D.csv')
