```python
import pandas as pd
import numpy as np
import os, sys, time
from tqdm import tqdm
from multiprocessing import Pool

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")

import warnings
warnings.simplefilter("ignore")

# %matplotlib inline
import seaborn
seaborn.set_style("darkgrid")

import matplotlib.pyplot as plt
plt.rc("figure", figsize=(9, 5))
# plt.rc("figure", figsize=(18, 10))
plt.rc("font", size=12)
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rc("savefig", dpi=90)
# plt.rc("font", family="sans-serif")
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"

if __name__ == '__main__':
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

```