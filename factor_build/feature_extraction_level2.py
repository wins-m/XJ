"""
(created by swmao on Feb. 7th)

"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from datetime import datetime


def cal_column(df: pd.DataFrame, cn=None, force_update=False) -> pd.DataFrame:
    """
    level2快照frame新增列
    :param df: 日内行情（个股）
    :param cn: column name
    :param force_update: 强制覆盖已存在列名下的值
    :return: 增加新列的日内行情
    """
    if cn is None:
        return df
    elif isinstance(cn, list):
        for cname in cn:
            df = cal_column(df, cn=cname, force_update=force_update)
    elif (cn in df.columns) and (not force_update):
        return df
    #
    elif cn == 'amt':
        df[cn] = df.Price * df.Volume
    elif cn == 'amtBuyOrder':
        id2v = (df.BuyOrderVolume * df.BuyOrderPrice).groupby(df.BuyOrderID).sum()
        df[cn] = df.BuyOrderID.apply(lambda x: id2v.loc[x])
    elif cn == 'amtSaleOrder':
        id2v = (df.SaleOrderVolume * df.SaleOrderPrice).groupby(df.SaleOrderID).sum()
        df[cn] = df.SaleOrderID.apply(lambda x: id2v.loc[x])
    else:
        raise ValueError(f'Invalid column name `{cn}`!')
    #
    return df


def cal_fv(df: pd.DataFrame, fname=None) -> float:
    """散户买入金额，单笔成交额小于4万元"""
    """
    file, key = '/mnt/c/Users/Winst/Documents/lvl2/201701/2017-01-03', '/2017-01-03_000001.SZ'
    df = pd.DataFrame(pd.read_hdf(file, key=key))
    """
    if fname is None:
        return np.nan
    elif fname == 'amountbuy_exlarge':
        df = cal_column(df, ['amt', 'amtBuyOrder'])
        mask = df.amtBuyOrder > 100e4
        return df[mask].amt.sum()
    elif fname == 'amountsell_exlarge':
        df = cal_column(df, ['amt', 'amtSaleOrder'])
        mask = df.amtSaleOrder > 100e4
        return df[mask].amt.sum()
    elif fname == 'amountbuy_large':
        df = cal_column(df, ['amt', 'amtBuyOrder'])
        mask = (df.amtBuyOrder > 20e4) & (df.amtBuyOrder <= 100e4)
        return df[mask].amt.sum()
    elif fname == 'amountsell_large':
        df = cal_column(df, ['amt', 'amtSaleOrder'])
        mask = (df.amtSaleOrder > 20e4) & (df.amtSaleOrder <= 100e4)
        return df[mask].amt.sum()
    elif fname == 'amountbuy_med':
        df = cal_column(df, ['amt', 'amtBuyOrder'])
        mask = (df.amtBuyOrder > 4e4) & (df.amtBuyOrder <= 20e4)
        return df[mask].amt.sum()
    elif fname == 'amountsell_med':
        df = cal_column(df, ['amt', 'amtSaleOrder'])
        mask = (df.amtSaleOrder > 4e4) & (df.amtSaleOrder <= 20e4)
        return df[mask].amt.sum()
    elif fname == 'amountbuy_small':
        df = cal_column(df, ['amt', 'amtBuyOrder'])
        mask = df.amtBuyOrder <= 4e4
        return df[mask].amt.sum()
    elif fname == 'amountsell_small':
        df = cal_column(df, ['amt', 'amtSaleOrder'])
        mask = df.amtSaleOrder <= 4e4
        return df[mask].amt.sum()
    elif fname == 'amountdiff_small':
        amountbuy_small = cal_fv(df, fname='amountbuy_small')
        amountsell_small = cal_fv(df, fname='amountsell_small')
        return amountbuy_small - amountsell_small
    elif fname == 'amountdiff_smallact':
        df = cal_column(df, ['amt', 'amtBuyOrder', 'amtSaleOrder'])
        maskBuy = (df.amtBuyOrder <= 4e4) & (df.Type == 'B')
        maskSale = (df.amtSaleOrder <= 4e4) & (df.Type == 'S')
        return df[maskBuy].amt.sum() - df[maskSale].amt.sum()
    elif fname == 'amountdiff_med':
        amountbuy_med = cal_fv(df, fname='amountbuy_med')
        amountsell_med = cal_fv(df, fname='amountsell_med')
        return amountbuy_med - amountsell_med
    elif fname == 'amountdiff_medact':
        df = cal_column(df, ['amt', 'amtBuyOrder', 'amtSaleOrder'])
        maskBuy = (df.amtSaleOrder > 4e4) & (df.amtSaleOrder <= 20e4) & (df.Type == 'B')
        maskSale = (df.amtSaleOrder > 4e4) & (df.amtSaleOrder <= 20e4) & (df.Type == 'S')
        return df[maskBuy].amt.sum() - df[maskSale].amt.sum()
    elif fname == 'amountdiff_large':
        amountbuy_large = cal_fv(df, fname='amountbuy_large')
        amountsell_large = cal_fv(df, fname='amountsell_large')
        return amountbuy_large - amountsell_large
    elif fname == 'amountdiff_largeact':
        df = cal_column(df, ['amt', 'amtBuyOrder', 'amtSaleOrder'])
        maskBuy = (df.amtSaleOrder > 20e4) & (df.amtSaleOrder <= 100e4) & (df.Type == 'B')
        maskSale = (df.amtSaleOrder > 20e4) & (df.amtSaleOrder <= 100e4) & (df.Type == 'S')
        return df[maskBuy].amt.sum() - df[maskSale].amt.sum()
    elif fname == 'amountdiff_exlarge':
        amountbuy_exlarge = cal_fv(df, fname='amountbuy_exlarge')
        amountsell_exlarge = cal_fv(df, fname='amountsell_exlarge')
        return amountbuy_exlarge - amountsell_exlarge
    elif fname == 'amountdiff_exlargeact':
        df = cal_column(df, ['amt', 'amtBuyOrder', 'amtSaleOrder'])
        maskBuy = (df.amtBuyOrder > 100e4) & (df.Type == 'B')
        maskSale = (df.amtBuyOrder > 100e4) & (df.Type == 'S')
        return df[maskBuy].amt.sum() - df[maskSale].amt.sum()
    else:
        raise ValueError(f'Invalid factor name `{fname}`!')


def panel_save_2d(panel_df: pd.DataFrame, csv_path: str):
    """
    批量计算的准备2d因子面板
    :param panel_df: 包含tradedate, stockcode的面板，后各列为单个因子
    :param csv_path: config.factorcsv_path
    :return:
    """
    print('(Pivot & Save)')
    # feature = features2update.F_NAME.iloc[2]
    for feature in (set(panel_df.columns) - {'stockcode', 'tradedate'}):  # features2update.F_NAME):
        # v.to_hdf(f'{csv_path}level2features.hdf', key=k)
        panel2d = panel_df.pivot(index='tradedate', columns='stockcode', values=feature)

        filename = feature + '.csv'
        if filename in os.listdir(csv_path):
            # 若已存在因子文件，则只添加原有最后一天之后的新值
            panel2d_raw = pd.read_csv(csv_path + filename)
            mask = (panel2d.index > panel2d_raw.tradedate.max())
            if mask.sum() > 0:
                panel2d.to_csv(csv_path + f'[{str(datetime.today())}]' + filename)  # 另存冲突的新结果
                pd.concat((panel2d_raw, panel2d[mask]), axis=0).to_csv(csv_path + filename)
        else:
            panel2d.to_csv(csv_path + filename)
        print(f'`{filename}` saved')


def get_fv_in_process(features: list, file: str, key: str) -> list:
    """
    计算单个因子
    :param features: 需计算的因子名列表
    :param file: 日hdf文件 绝对路径
    :param key: 日hdf文件内个股key
    :return: 列表 值依次为 tradedate, stockcode, feature1, feature2...
    """
    tradedate = file.rsplit('/', maxsplit=1)[-1]
    stockcode = key.split('_')[1]
    # res = {'tradedate': tradedate, 'stockcode': stockcode}
    res = [tradedate, stockcode]

    dat_lvl2 = pd.DataFrame(pd.read_hdf(file, key=key))
    # 进入子进程前，计算必要的列（增加传递，减少计算); 若无需传递frame，则新增列只生成一次
    # dat_lvl2 = cal_column(dat_lvl2, ['amt', 'amtBuyOrder', 'amtSaleOrder'])
    for fea in features:
        # res[fea] = cal_fv(dat_lvl2, fea)
        res.append(cal_fv(dat_lvl2, fea))

    print(tradedate, stockcode)
    return res


def feature_extraction(path_lvl2, feature_begin, feature_end, feature_level2, csv_path, test=True):
    """
    多进程计算因子；一次性指定尽量多的因子；保存
    :param path_lvl2: config.lvl2_path
    :param feature_begin: config.feature_begin
    :param feature_end: config.feature_end
    :param feature_level2: config.feature_level2
    :param csv_path: config.factorcsv_path
    :return:
    """
    folders = [f'{path_lvl2}{xx}/{x}/' for xx in os.listdir(path_lvl2) for
               x in os.listdir(path_lvl2 + xx) if (x >= feature_begin) and (x <= feature_end)]
    hdf_files = [x + xx for x in folders for xx in os.listdir(x)]
    features2update: pd.DataFrame = pd.read_excel(feature_level2, index_col=0).loc[1:1]
    features = features2update.F_NAME.to_list()

    # trade_dates = [pd.to_datetime(xx) for x in folders for xx in os.listdir(x)]
    # features_value_stk = pd.DataFrame(columns=features2update.F_NAME, index=trade_dates)
    # feature_values = {k: pd.DataFrame(index=trade_dates) for k in features2update.F_NAME}
    # feature_values_panel = pd.DataFrame(columns=['tradedate', 'stockcode'] + features2update.F_NAME.to_list())
    all_result = []

    p = Pool(6)
    file = hdf_files[0]
    for file in hdf_files[:]:
        # 取出单日行情hdf中的key（对应一支个股）
        hstore = pd.HDFStore(file, 'r')
        keys = hstore.keys()
        hstore.close()
        # print(f'({tradedate})')
        key = keys[0]
        for key in keys[:]:  # 遍历计算该日内所有个股因子值
            # res = get_fv_in_process(features_l, file, key)
            if test:
                print(get_fv_in_process(features, file, key))
                return
            else:
                p.apply_async(get_fv_in_process, args=(features, file, key), callback=all_result.append)
    p.close()
    p.join()

    feature_values_panel = pd.DataFrame(all_result, columns=['tradedate', 'stockcode']+features2update.F_NAME.to_list())
    panel_save_2d(panel_df=feature_values_panel, csv_path=csv_path)


# %%
if __name__ == '__main__':
    # %%
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    path_lvl2 = conf['level2_path2']  # path_level2 = conf['level2_path']
    feature_begin = conf['feature_begin']
    feature_end = conf['feature_end']
    feature_level2 = conf['feature_level2']
    csv_path = conf['factorscsv_path']
    if_test = False  # 此处更改，是否不进入多进程

    # %%
    feature_extraction(path_lvl2, feature_begin, feature_end, feature_level2, csv_path, test=if_test)
