import pandas as pd

path = '/mnt/c/Users/Winst/Documents/factors_res/'
# folders = [
#     'first_report_i5_100car10_0up_dur3_n_NA_ew_1g_ctc_1hd(0215_094855)',
#     'first_report_i5_80car10_0up_dur3_n_NA_ew_1g_ctc_1hd(0215_094855)',
#     'first_report_i5_60car10_0up_dur3_n_NA_ew_1g_ctc_1hd(0215_094855)',
#     'first_report_i5_50car10_0up_dur3_n_NA_ew_1g_ctc_1hd(0215_094855)',
#     'first_report_i5_40car10_0up_dur3_n_NA_ew_1g_ctc_1hd(0215_094855)',
#     'first_report_i5_20car10_0up_dur3_n_NA_ew_1g_ctc_1hd(0215_094855)',
# ]
folders = [
    'first_report_dur3_n_NA_ew_1g_ctc_1hd(ipo60&suspend)',
    'first_report_dur3_updown_n_NA_ew_1g_ctc_1hd(ipo60&suspend)',
    'first_report_dur3_n_NA_ew_1g_ctc_1hd(ipo60&suspend&up)',
    'first_report_dur3_updown_open_n_NA_ew_1g_ctc_1hd(ipo60&suspend)',
    'first_report_dur3_updown_n_NA_ew_1g_ctc_1hd(ipo60&suspend&up)',
    'first_report_dur3_n_NA_ew_1g_ctc_1hd(ipo60&suspend&up_open)',
    'first_report_dur3_updown_open_n_NA_ew_1g_ctc_1hd(ipo60&suspend&up)',
    'first_report_dur3_updown_n_NA_ew_1g_ctc_1hd(ipo60&suspend&up_open)',
    'first_report_dur3_n_NA_ew_1g_ctc_1hd(ipo60&suspend&updown)',
    'first_report_dur3_updown_open_n_NA_ew_1g_ctc_1hd(ipo60&suspend&up_open)',
    'first_report_dur3_updown_n_NA_ew_1g_ctc_1hd(ipo60&suspend&updown)',
    'first_report_dur3_n_NA_ew_1g_ctc_1hd(ipo60&suspend&updown_open)',
    'first_report_dur3_updown_open_n_NA_ew_1g_ctc_1hd(ipo60&suspend&updown)',
    'first_report_dur3_updown_n_NA_ew_1g_ctc_1hd(ipo60&suspend&updown_open)',
    'first_report_dur3_updown_open_n_NA_ew_1g_ctc_1hd(ipo60&suspend&updown_open)',
    'first_report_dur3_up_n_NA_ew_1g_ctc_1hd(ipo60&suspend)',
    'first_report_dur3_up_open_n_NA_ew_1g_ctc_1hd(ipo60&suspend)',
    'first_report_dur3_up_n_NA_ew_1g_ctc_1hd(ipo60&suspend&up)',
    'first_report_dur3_up_n_NA_ew_1g_ctc_1hd(ipo60&suspend&up_open)',
    'first_report_dur3_up_open_n_NA_ew_1g_ctc_1hd(ipo60&suspend&up)',
    'first_report_dur3_up_n_NA_ew_1g_ctc_1hd(ipo60&suspend&updown)',
    'first_report_dur3_up_open_n_NA_ew_1g_ctc_1hd(ipo60&suspend&up_open)',
    'first_report_dur3_up_open_n_NA_ew_1g_ctc_1hd(ipo60&suspend&updown)',
    'first_report_dur3_up_n_NA_ew_1g_ctc_1hd(ipo60&suspend&updown_open)',
    'first_report_dur3_up_open_n_NA_ew_1g_ctc_1hd(ipo60&suspend&updown_open)',
]

res = []
for folder in folders:
    df = pd.read_excel(path+folder+'/ResLongNC.xlsx', index_col=0, parse_dates=True)
    print(folder, df)
    res.append([folder, df['TotalAnnualRet'].iloc[0], df['TotalRet'].iloc[-1], df['TotalSharpe'].iloc[0]])

df_res = pd.DataFrame(res, columns=['FileName', 'TotalAnnualRet', 'TotalRet', 'TotalSharpe'])
# df_res = df_res.set_index('FildName')
print(df_res)

df_res['EXCLUDE'] = df_res['FileName'].apply(lambda x: x.split('(')[-1][:-1])
df_res['SIGNAL'] = df_res['FileName'].apply(lambda x: 'ex' + x.split('dur3')[-1].split('_n_NA')[0])

# df_res.to_excel('/mnt/c/Users/Winst/Documents/factors_res/event_first_report/' + 'first_report_dur3[0214].xlsx', index=None)

res = pd.DataFrame()
for v_key in ['TotalAnnualRet', 'TotalRet', 'TotalSharpe']:
    tmp = df_res.pivot(index='SIGNAL', columns='EXCLUDE', values=v_key)
    tmp['IND'] = v_key
    tmp = tmp.reset_index().set_index(['IND', 'SIGNAL'])
    res = pd.concat((res, tmp), axis=0)
# res = res.stack().unstack('IND')

res.to_excel('/mnt/c/Users/Winst/Documents/factors_res/event_first_report/' + 'first_report_dur3[0214].xlsx')
# res.to_excel('/mnt/c/Users/Winst/Documents/factors_res/event_first_report/' + 'first_report_CAR_10[0215].xlsx')


