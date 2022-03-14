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

TODO
- check recent 1000, exclude
- fill (adding indus group label)
- split training set, validation set, blackout set

"""
from typing import Tuple
import sys, os
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.alpha import *


def main():
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    # Y012: next period return
    # prepare_t1_prediction_y012(conf, ret_kind='ot5c', stk_pool='CSI500')
    # prepare_t1_prediction_y012(conf, ret_kind='ctc', stk_pool='CSI500')
    # prepare_t1_prediction_y012(conf, ret_kind='oto', stk_pool='CSI500')
    # prepare_t1_prediction_y012(conf, ret_kind='ct5o', stk_pool='CSI500')

    # X: alpha 101 raw value from Wind
    # tgt_file = 'Y012_TmrRtnOT5C_CSI500pct10_TopMidBott.pkl'
    # feature_files = sorted([x for x in os.listdir(conf['factorscsv_path']) if x[:6] == 'alpha_'])
    # print(feature_files[:5])
    # merge_all_alpha(conf, tgt_file, feature_files)

    #
    src_file = 'X79Y012_TmrRtnOT5C_CSI500pct10_TopMidBott.pkl'
    # add_group_labels(conf, src_file, replace=True)
    # exclude_historical_low_coverage_features(conf, src_file)
    train_bundle(conf, src_file)


def train_bundle(conf, src_file, cvg_bar=.667):
    """(1000+5)每5天，去除在训练集内覆盖率低于cvg_bar的因子，剩余缺失因子填充以日截面风格大类均值，存储每次训练的结果"""
    data_path = conf['data_path']
    _path = data_path + src_file.replace('.pkl', '/{}')
    os.makedirs(_path.format(''), exist_ok=True)
    data = pd.read_pickle(data_path + src_file)
    data_desc = pd.read_excel(data_path + src_file.replace('.pkl', '.xlsx'), index_col=0)
    feature_cols = data_desc.index[1:].to_list()

    dat_iter = iter_data_td1k(data)
    for dat_train, dat_test in dat_iter:  # 后续数据处理都模拟每5日增量更新，1000 for training, 5 for evaluation
        assert len(dat_train.tradingdate.unique()) == 1000
        assert len(dat_test.tradingdate.unique()) == 5
        bd0 = dat_train.tradingdate.min().strftime('%Y%m%d')
        # ed0 = dat_train.tradingdate.max()
        bd1 = dat_test.tradingdate.min().strftime('%Y%m%d')
        # ed1 = dat_test.tradingdate.max()
        print(bd1, '...')

        dat_train, dat_test, fea2ex = drop_low_coverage_features(dat_train, dat_test, bar=cvg_bar)
        fea_cols = [x for x in feature_cols if x not in fea2ex]
        dat_train = drop_na_after_sector_mean_fill(dat=dat_train, fea_cols=fea_cols, c0='tradingdate', c1='sector_ci')
        dat_test = drop_na_after_sector_mean_fill(dat=dat_test, fea_cols=fea_cols, c0='tradingdate', c1='sector_ci')

        dat_train.to_pickle(_path.format(f'TRAIN_{bd0}_{bd1}.pkl'))  # TODO: 控制周末更新
        dat_test.to_pickle(_path.format(f'TEST_{bd0}_{bd1}.pkl'))

        _desc = data_desc.loc[[idx for idx in data_desc.index if idx not in fea2ex]]
        _desc.to_excel(_path.format(f'DESC_{bd0}_{bd1}.xlsx'))


def drop_na_after_sector_mean_fill(dat: pd.DataFrame, fea_cols: list, c0, c1) -> pd.DataFrame:
    train_feature_filled = feature_group_mean(dat[fea_cols], dat[c0], dat[c1])
    dat[fea_cols] = train_feature_filled
    shape0 = dat.shape
    dat_ = dat.dropna()  # 缺失features且缺失风格大类，无法填充，则去除
    shape1 = dat_.shape
    print(f'Fill NA feature value cross {c0} with {c1}-mean, panel shape from {shape0} to {shape1}')
    return dat_


def feature_group_mean(features: pd.DataFrame, td: pd.Series, gr: pd.Series) -> pd.DataFrame:
    res = pd.DataFrame().reindex_like(features)
    # col = features.columns[0]
    print('Filling NA fval with sector-daily-mean...')
    for col in tqdm(features.columns):
        fea = features[col].copy().rename('fv')
        tmp = pd.concat([td, gr, fea], axis=1)
        m = fea.groupby([td, gr]).mean().rename('sm').reset_index()
        tmp = tmp.merge(m, on=[td.name, gr.name], how='left')
        mask_no_fv = tmp.fv.isna()
        mask_no_sector = tmp.sm.isna()
        tmp['fv'] = tmp.fv.fillna(0)
        tmp['sm'][~mask_no_fv] = 0
        tmp['fv_1'] = tmp.fv + tmp.sm
        assert tmp.fv_1.isna().mean() == (mask_no_fv & mask_no_sector).mean()
        res[col] = tmp.fv_1
    return res


def drop_low_coverage_features(dat_train: pd.DataFrame, dat_test: pd.DataFrame, bar: float) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """

    :param dat_train:
    :param dat_test:
    :param bar:
    :return:
    """
    col2ex = col2ex_cvt_lt_667(dat_train, bar=bar)
    _train = dat_train[[col for col in dat_train if col not in col2ex]]
    _test = dat_test[[col for col in dat_test if col not in col2ex]]
    return _train, _test, col2ex


def col2ex_cvt_lt_667(dat_: pd.DataFrame, bar) -> list:
    """

    :param dat_:
    :param bar:
    :return:
    """
    cnt_all = dat_.groupby(['tradingdate']).stockcode.count()
    fea_cvg: pd.DataFrame = dat_.loc[:, 'x0':].groupby(dat_.tradingdate).count() / cnt_all.values.reshape(-1, 1)
    mask_l_cvg = (fea_cvg.min() < bar)
    fea2ex = [col for col in fea_cvg.columns if mask_l_cvg[col]]

    return fea2ex
    

if __name__ == '__main__':
    main()
