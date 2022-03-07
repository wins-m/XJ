import pandas as pd
import numpy as np
import os, sys, time
from tqdm import tqdm
from multiprocessing import Pool
from matplotlib import pyplot as plt

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")

import warnings

warnings.simplefilter("ignore")

# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style("darkgrid")
# plt.rc("figure", figsize=(16, 6))
plt.rc("figure", figsize=(8, 3))
plt.rc("savefig", dpi=90)
# plt.rc("font", family="sans-serif")
# plt.rc("font", size=12)
plt.rc("font", size=10)

plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"

# %%
if __name__ == '__main__':
    # %%
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))
