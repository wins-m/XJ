"""
(created by swmao on March 3rd)
准备训练数据
- Y0, Y1, Y2: 采用收盘价收益率ctc TODO: other predictions (future return rank)
    - 通过./data/tradeable.py获得的a_list_tradeable::tradeable_noupdown筛选
    - 去除开盘不满60个交易日，昨日（买入close）涨跌停，昨日（买入close）停牌
    - 对剩余的ctc收益率日内进行排名，分别取top, middle, bottom 10%
- ./factor_backtest/pipeline.py: format features before merge
- merge alpha features aligned with tradeable Y (ipo60 & noUpDown & noST)
    - date range: 201301~202202(recently) except for alpha_045
- TODO: split training set, validation set, blackout set
"""
import pandas as pd
import os
from tqdm import tqdm
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.request import get_hold_return


def main():
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    # prepare_t1_prediction_y012(conf, ret_kind='ot5c', stk_pool='CSI500')
    # prepare_t1_prediction_y012(conf, ret_kind='ctc', stk_pool='CSI500')
    # prepare_t1_prediction_y012(conf, ret_kind='oto', stk_pool='CSI500')
    # prepare_t1_prediction_y012(conf, ret_kind='ct5o', stk_pool='CSI500')

    # tgt_file = 'Y012_TmrRtnOT5C_CSI500pct10_TopMidBott.pkl'
    feature_files = sorted([x for x in os.listdir(conf['factorscsv_path']) if '[CSI500ranked]alpha_' in x])
    print(feature_files[:5])
    # merge_all_alpha(conf, tgt_file, feature_files)


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

    hold_ret = get_hold_return(conf=conf, ret_kind=ret_kind, bd=conf['begin_date'], ed=conf['end_date'], stk_pool=stk_pool)

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
        res = res.apply(lambda x: f'{x*100:.2f} %')
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


if __name__ == '__main__':
    main()
