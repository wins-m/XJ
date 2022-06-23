# %matplotlib inline
import warnings
warnings.simplefilter("ignore")
import seaborn
seaborn.set_style("darkgrid")
import matplotlib.pyplot as plt
plt.rc("figure", figsize=(9, 5))
plt.rc("font", size=12)
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rc("savefig", dpi=90)
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"
