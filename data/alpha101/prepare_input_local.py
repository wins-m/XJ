"""
(created by swmao on March 3rd)
"""
import pandas as pd
import numpy as np


def main():
    import yaml

    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    prepare_t1_prediction_y012(conf)


def prepare_t1_prediction_y012(conf, begin_date='2013-01-01', end_date='2022-12-31'):
    """
    用3维的向量表示3种不同的输出类别。
    𝒚=[100]𝑇表示上涨样本（每个时间截面上，将全体股票按照未来1个交易日收益率排序，收益率最高的前10%的股票样本标记为“上涨样本”），
    𝒚=[010]𝑇表示平盘样本（收益率居中的10%的股票样本），
    𝒚=[001]𝑇表示下跌样本（收益率最低的10%的股票样本）
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


if __name__ == '__main__':
    main()
