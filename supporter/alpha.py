import pandas as pd
from tqdm import tqdm
import sys

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.request import get_hold_return
from supporter.request import get_sector_ci_all_tradingdate
from supporter.io import get_time_suffix

import warnings

warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style("darkgrid")
plt.rc("figure", figsize=(16, 6))
# plt.rc("figure", figsize=(8, 3))
plt.rc("savefig", dpi=90)
# plt.rc("font", family="sans-serif")
plt.rc("font", size=12)
# plt.rc("font", size=10)
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"


def prepare_t1_prediction_y012(conf, ret_kind, stk_pool):
    """
    用3维的向量表示3种不同的输出类别。
    𝒚=[100]𝑇表示上涨样本（每个时间截面上，将全体股票按照未来1个交易日收益率排序，收益率最高的前10%的股票样本标记为“上涨样本”），
    𝒚=[010]𝑇表示平盘样本（收益率居中的10%的股票样本），
    𝒚=[001]𝑇表示下跌样本（收益率最低的10%的股票样本）
    Accessible: 60 trade days since ipo, yesterday close not max_up_or_down, yesterday close not suspend
    Output: Y012_TmrRtnC2C_Pct10_TopMidBott.pkl

    """

    def get_predict_target(_hold_ret, _ret_kind):
        _hold_ret_tmr = _hold_ret.shift(-1).stack().reset_index()  # Attention: shift -1 for Day+1 prediction
        col = f'rtn_{_ret_kind}'
        _hold_ret_tmr.columns = ['tradingdate', 'stockcode', col]
        _hold_ret_tmr[f'rnk_{col}'] = _hold_ret_tmr.groupby('tradingdate')[col].rank(pct=True, ascending=True)
        _hold_ret_tmr['y0'] = _hold_ret_tmr[f'rnk_{col}'] > 0.9
        _hold_ret_tmr['y1'] = (_hold_ret_tmr[f'rnk_{col}'] > 0.55) & (_hold_ret_tmr[f'rnk_{col}'] <= 0.65)
        _hold_ret_tmr['y2'] = _hold_ret_tmr[f'rnk_{col}'] <= 0.1
        _hold_ret_tmr[['y0', 'y1', 'y2']] = _hold_ret_tmr[['y0', 'y1', 'y2']].astype(int)
        return _hold_ret_tmr

    hold_ret = get_hold_return(conf=conf, ret_kind=ret_kind, bd=conf['begin_date'], ed=conf['end_date'],
                               stk_pool=stk_pool)

    # print('Load tradeable status from local...')
    # tradeable_path = conf['a_list_tradeable']
    # tradeable = pd.DataFrame(pd.read_hdf(tradeable_path, key='tradeable_noupdown', parse_dates=True))  # 60 trade days since ipo, yesterday close not max_up_or_down, yesterday close not suspend
    # tradeable = tradeable.reindex_like(hold_ret)
    # tradeable = tradeable.replace(False, np.nan).replace(True, 1)
    # hold_ret *= tradeable

    print('Cal future status(000, 001, 010, 100)...')
    hold_ret_tmr = get_predict_target(hold_ret, ret_kind)

    save_file_name = conf['data_path'] + f'Y012_TmrRtn{ret_kind.upper()}_{stk_pool}pct10_TopMidBott.pkl'
    hold_ret_tmr.to_pickle(save_file_name)
    print(f'Saved in {save_file_name}\n')

    def get_y_compos(df, save_path=None):
        res = (df.y0 * 100 + df.y1 * 10 + df.y2).value_counts()
        res.rename(index={0: '000', 1: '001', 10: '010', 100: '100'}, inplace=True)
        res /= res.sum()
        res = res.apply(lambda x: f'{x * 100:.2f} %')
        if save_path is not None:
            res.to_csv(save_path)
        return res

    print(get_y_compos(hold_ret_tmr).to_dict())


def merge_all_alpha(conf, tgt_file, feature_files):
    """准备feature，添加alpha101与收盘价对齐"""
    data_path = conf['data_path']
    csv_path = conf['factorscsv_path']
    data = pd.read_pickle(data_path + tgt_file)

    feature_name_dict = pd.DataFrame()
    feature_name_dict['data_y'] = [tgt_file]
    cnt = 0
    # file = feature_files[0]
    for file in tqdm(feature_files):
        # print(file, '...')
        feature = pd.read_csv(csv_path + file, parse_dates=True, index_col=0)
        feature = feature.stack().reset_index()

        feature.columns = ['tradingdate', 'stockcode', f'x{cnt}']
        feature_name_dict[f'x{cnt}'] = [file]
        cnt += 1

        data = data.merge(feature, on=['tradingdate', 'stockcode'], how='left')

    save_path = data_path + f'X{len(feature_files)}{tgt_file}'
    data.to_pickle(save_path)
    feature_name_dict.T.to_excel(save_path.replace('.pkl', '.xlsx'))
    print(f'Saved in {save_path}')


def add_group_labels(conf, src_file, replace=False):
    """Add one column: citic industry label"""
    data_path = conf['data_path']
    data = pd.read_pickle(data_path + src_file)

    col_x0_ind = data.columns.to_list().index('x0')
    data0, data1 = data.iloc[:, :col_x0_ind], data.iloc[:, col_x0_ind:]
    begin_date, end_date = data.tradingdate.min(), data.tradingdate.max()

    sector_ci = get_sector_ci_all_tradingdate(conf, begin_date, end_date).stack().reset_index()
    sector_ci.columns = ['tradingdate', 'stockcode', 'sector_ci']
    data0 = data0[[col for col in data0.columns if col != 'sector_ci']]
    data0 = data0.merge(sector_ci, on=['tradingdate', 'stockcode'], how='left')

    data = pd.concat([data0, data1], axis=1)

    if replace:
        data.to_pickle(data_path + src_file)
    else:
        data.to_pickle(data_path + src_file.replace('.pkl', f'[{get_time_suffix()}].pkl'))


def sector_mean_fill_na(data: pd.DataFrame, fea_cols: list, sct: str = 'sector_ci') -> pd.DataFrame:
    """根据data中sct列的分类标签计算fea_cols截面均值填充空值"""

    data_ = data.copy()

    for col in tqdm(fea_cols):
        tmp = data_[['tradingdate', sct, col]].copy()
        tmp['isNA'] = tmp[col].isna()

        mean_cross_sector = data_.groupby(['tradingdate', sct])[col].mean()
        mean_cross_sector = mean_cross_sector.reset_index().rename(columns={col: 'm_sector'})
        tmp = tmp.merge(mean_cross_sector, on=['tradingdate', sct], how='left')
        data_[col][tmp.isNA] = tmp['m_sector'][tmp.isNA]

    return data_


def exclude_historical_low_coverage_features(conf, src_file):
    """Depreciated: 去除了历史上日覆盖率最小值低于66.7%的因子；每周增量更新模型时，应该过去1000天内最小覆盖率不足的被去除（其余被填充）"""
    data_path = conf['data_path']
    data: pd.DataFrame = pd.read_pickle(data_path + src_file)
    data_desc: pd.DataFrame = pd.read_excel(data_path + src_file.replace('.pkl', '.xlsx'), index_col=0)

    d_stk_num = pd.DataFrame(data.groupby('tradingdate').stockcode.count().rename('Stock Number All'))
    d_stk_num['Stock Number Non-Fill'] = data.dropna().groupby('tradingdate').stockcode.count()
    d_fea_num = data.loc[:, 'x0':].groupby(data.tradingdate).count()
    d_fea_cvg = d_fea_num / d_stk_num['Stock Number All'].values.reshape(-1, 1)

    d_fea_cvg_stat = d_fea_cvg.describe().T
    fea2ex = d_fea_cvg_stat[d_fea_cvg_stat['min'] < 0.667].index.to_list()
    print(data_desc.loc[fea2ex],
          f'\nExclude {len(fea2ex)} features whose historical minimum daily coverage lower than 66.7%.')

    data_ex = data[[col for col in data.columns if col not in fea2ex]]
    d_stk_num[f'Stock Number Exclude {len(fea2ex)}'] = data_ex.dropna().groupby('tradingdate').stockcode.count()

    data_desc_ex = data_desc.loc[[idx for idx in data_desc.index if idx not in fea2ex]]
    save_name = f'X{len(data_desc_ex) - 1}Y012' + src_file.split('Y012')[1]
    data_ex.to_pickle(data_path + save_name)
    save_name1 = save_name.replace('.pkl', '.xlsx')
    data_desc_ex.to_excel(data_path + save_name1)
    save_name2 = 'dStkCnt_' + save_name1
    d_stk_num.to_excel(data_path + save_name2)
    print(f'Save\n - new data in {save_name},\n - description in {save_name1},\n - daily stock count in {save_name2}.')

    d_stk_num.plot(title='Sample Number Daily')
    plt.show()
    # plt.savefig('')
    # plt.close()


def iter_data_td1k(data):
    """划分历史样本1001日为一组"""
    all_td = data.tradingdate.unique()
    data = data.set_index('tradingdate').copy()
    return (split_1k1_train_test(data.loc[bd: ed]) for bd, ed in zip(all_td[:-1004:5], all_td[1004::5]))


def split_1k1_train_test(data_td1k):
    """划分1001日为1000日训练集和5日测试集"""
    if 'tradingdate' in data_td1k.columns:
        data_td1k = data_td1k.set_index('tradingdate')
    td_last = data_td1k.index.unique()[-5]
    mask = data_td1k.index < td_last
    data_train = data_td1k[mask]
    data_test = data_td1k[~mask]
    return data_train.reset_index(), data_test.reset_index()
