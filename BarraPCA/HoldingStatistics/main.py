import yaml
import pandas as pd
import os
from matplotlib import pyplot as plt
plt.rc("figure", figsize=(8, 5))
plt.rc("font", size=15)
plt.rcParams['axes.autolimit_mode'] = 'round_numbers'
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
plt.rc("savefig", dpi=90)

conf = yaml.safe_load(open('./config.yaml'))

os.makedirs(conf['tgt_path'], exist_ok=True)
os.makedirs('figures', exist_ok=True)
# if os.path.exists(conf['tgt_file']):
#     stat_res = pd.read_csv(conf['tgt_file'], encoding=conf['coding'])
# else:
#     stat_res = pd.DataFrame()
stat_res = pd.DataFrame()

# filename = conf['src_file'][0]
report_date = conf['report_date']
# for report_date in ['2021-06-30', '2021-12-31']:
for filename in sorted(os.listdir(conf['src_path'])):
    print('\n', filename)
    res = {'src': filename, 'code': '', 'report_date': '', 'mkt_type': '',
           'cons_w': -1., 'cons_n': -1, 'ncons_n': -1}

    src_code = filename.split('(')[1].split(')')[0]
    res['code'] = src_code

    res['report_date'] = report_date

    if '中证500' in filename:
        mkt_type = 'CSI500'
    elif '沪深300' in filename:
        mkt_type = 'CSI300'
    else:
        raise Exception(filename)
    res['mkt_type'] = mkt_type

    file0 = conf['idx_constituent'].format(mkt_type)
    df0 = pd.read_csv(file0, index_col=0, parse_dates=True)
    df0 = df0.loc[report_date].dropna()
    # print(df0.head())

    file1 = conf['src_path'] + filename
    df1 = pd.read_excel(file1, index_col=0).loc[report_date]
    df1 = df1.rename(columns={'股票代码': 'stockcode', '持仓市值(元)': 'weight'})
    df1 = df1.set_index('stockcode')['weight']
    df1 /= df1.sum()
    df1 *= 100
    # print(df1.head())

    stockcode_intersection = df1.index.intersection(df0.index)
    res['cons_w'] = df1.loc[stockcode_intersection].sum()
    print(f"成分股权重 {res['cons_w']:.2f}%")
    res['cons_n'] = len(stockcode_intersection)
    print('成分股数量', res['cons_n'])
    res['ncons_n'] = len(df1) - res['cons_n']
    print('非成分股数量', res['ncons_n'])

    df = pd.concat([df1.rename(src_code), df0.rename(mkt_type)], axis=1)
    df = df.sort_values(src_code, ascending=False)
    df['excess_weight'] = (df.iloc[:, 0] - df.iloc[:, 1].fillna(0))
    tgt_filename = filename.split('-')[0] + f'-{report_date}.xlsx'
    df.to_excel(conf['tgt_path'] + tgt_filename)

    fig_title = f'{src_code} {mkt_type} {report_date}'
    fig_filename = fig_title.replace(' ', '_').replace(
        '-', '').replace('.', '') + '.png'
    tmp = df[src_code].dropna().reset_index(drop=True)
    tmp.plot(title=fig_title, linewidth=3)
    plt.tight_layout()
    plt.savefig('./figures/' + fig_filename)
    plt.close()
    del tmp

    stat_res = pd.concat(
        [pd.DataFrame([res]), stat_res], ignore_index=True)

stat_res = stat_res.drop_duplicates()
print(stat_res)
stat_res.to_csv(conf['tgt_file'], index=None, encoding=conf['coding'])
