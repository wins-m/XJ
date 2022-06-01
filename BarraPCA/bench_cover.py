import pandas as pd


file0 = "/mnt/c/Users/Winst/Documents/data_local/idx_constituent_CSI500.csv"
file1 = f"""/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/BarraPCA/betaETF/holding_161017.csv"""
#file1 = f"""/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/BarraPCA/betaETF/holding1_161017.csv"""

df0 = pd.read_csv(
        file0,
        index_col=0,
        parse_dates=True).loc[
                '2021-06-30'
                #'2020-12-31'
                ].dropna()
df1 = pd.read_csv(file1).set_index('stockcode')['weight']
print(df1.head())
print(df0.head())
print(df1.loc[df1.index.intersection(df0.index)].sum())


pd.DataFrame().to_csv(file1.replace('holding', 'holding1'))
