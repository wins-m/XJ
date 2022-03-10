"""
(created by swmao on March 3rd)
准备训练数据
- Y0, Y1, Y2: 采用收盘价收益率ctc
    - 通过./data/tradeable.py获得的a_list_tradeable::tradeable_noupdown筛选
    - 去除开盘不满60个交易日，昨日（买入close）涨跌停，昨日（买入close）停牌
    - 对剩余的ctc收益率日内进行排名，分别取top, middle, bottom 10%
- alpha features aligned with tradeable Y (ipo60 & noUpDown & noST)
    - date range: 201301~202202(recently) except for alpha_045
"""
import pandas as pd
import numpy as np
import os
from tqdm import tqdm


def main():
    import yaml

    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    # prepare_t1_prediction_y012(conf)
    merge_all_alpha(conf)  # TODO: normalize alpha features before merge


def prepare_t1_prediction_y012(conf, begin_date='2013-01-01', end_date='2022-12-31'):
    """
    用3维的向量表示3种不同的输出类别。
    𝒚=[100]𝑇表示上涨样本（每个时间截面上，将全体股票按照未来1个交易日收益率排序，收益率最高的前10%的股票样本标记为“上涨样本”），
    𝒚=[010]𝑇表示平盘样本（收益率居中的10%的股票样本），
    𝒚=[001]𝑇表示下跌样本（收益率最低的10%的股票样本）
    Accessible: 60 trade days since ipo, yesterday close not max_up_or_down, yesterday close not suspend
    Output: Y012_TmrRtnC2C_Pct10_TopMidBott.pkl
    """
    path_close_adj = conf['closeAdj']
    tradeable_path = conf['a_list_tradeable']
    data_path = conf['data_path']

    print('Load closeAdj from local...')
    close_adj = pd.read_csv(path_close_adj, index_col=0,
                            parse_dates=True)  # predict returns: long yesterday close, short today close
    return_close_adj = close_adj.pct_change().loc[begin_date:end_date]

    print('Load tradeable status from local...')
    tradeable = pd.DataFrame(pd.read_hdf(tradeable_path, key='tradeable_noupdown',
                                         parse_dates=True))  # 60 trade days since ipo, yesterday close not max_up_or_down, yesterday close not suspend
    tradeable = tradeable.reindex_like(return_close_adj)
    tradeable = tradeable.replace(False, np.nan).replace(True, 1)

    print('Cal future status(000, 001, 010, 100)...')
    return_close_adj_tradeable = return_close_adj * tradeable
    tmp = return_close_adj_tradeable.shift(-1).stack().reset_index()  # Attention: shift -1 for Day+1 prediction
    tmp.columns = ['tradingdate', 'stockcode', 'tmr_rtn_c2c']
    tmp['d_rank_asc_pct'] = tmp.groupby('tradingdate')['tmr_rtn_c2c'].rank(pct=True, ascending=True)
    tmp['y0'] = tmp.d_rank_asc_pct > 0.9
    tmp['y1'] = (tmp.d_rank_asc_pct > 0.55) & (tmp.d_rank_asc_pct <= 0.65)
    tmp['y2'] = tmp.d_rank_asc_pct <= 0.1
    tmp[['y0', 'y1', 'y2']] = tmp[['y0', 'y1', 'y2']].astype(int)

    print('Save pickle...')
    save_file_name = data_path + 'Y012_TmrRtnC2C_Pct10_TopMidBott.pkl'
    tmp.to_pickle(save_file_name)

    print(f'Saved in {save_file_name}\n')

    def get_y_compos(df):
        res = (df.y0 * 100 + df.y1 * 10 + df.y2).value_counts()
        res.rename(index={0: '000', 1: '001', 10: '010', 100: '100'}, inplace=True)
        return res

    print(get_y_compos(tmp))


def merge_all_alpha(conf):
    """准备feature，添加alpha101与收盘价对齐"""
    data_path = conf['data_path']

    data = pd.read_pickle(data_path + 'Y012_TmrRtnC2C_Pct10_TopMidBott.pkl')
    feature_files = [x for x in os.listdir(data_path) if 'alpha_' in x]

    for file in tqdm(feature_files):
        # print(file, '...')
        feature = pd.read_csv(data_path + file, parse_dates=True, index_col=0)
        feature = feature.stack().reset_index()
        feature.columns = ['tradingdate', 'stockcode', file.replace('.csv', '')]
        data = data.merge(feature, on=['tradingdate', 'stockcode'], how='left')

    data.to_pickle(data_path + f'Y012_X{len(feature_files)}.pkl')
    print(f'Saved in {data_path}Y012_X{len(feature_files)}.pkl')


if __name__ == '__main__':
    main()
