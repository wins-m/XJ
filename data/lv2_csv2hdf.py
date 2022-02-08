"""
(created by swmao on Jan. 28th)

"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")
from supporter.io import break_confirm, table_save_safe


def stockcode_dictionary(stockcode_list: list, kind='dict'):
    """
    由stockcode列表，生成一个六位数字到九位代码的查询字典
    :param stockcode_list 列表，值为stockcodes
    :param kind: 指定返回dict还是dataframe
    :return: df key为6位数字，value为stockcode

    """
    if kind == 'dict':
        return {k.split('.')[0]: k for k in stockcode_list if k is not None}
    df = pd.DataFrame(stockcode_list)
    df.columns = ['stockcode']
    df.index = df['stockcode'].apply(lambda x: x.split('.')[0])
    return df


def lv2_csv2hdf(conf, re_generate=False, remove_raw=False):
    """
    转换快照csv到hdf，高压缩率（依赖于快照目录）
    :param conf: 配置文件
    :param re_generate: 是否重置记录文件
    :param remove_raw: 是否移除原始csv文件

    """
    # 目标股池
    tradeable_ipo = pd.read_hdf(conf['a_list_tradeable'], key='ipo', index_col=0, parse_dates=True)
    tradeable_ipo = pd.DataFrame(tradeable_ipo, dtype='float16')
    tradeable_ipo = tradeable_ipo.replace(0, np.nan)
    # tradeable_ipo_d_cnt = tradeable_ipo.sum(axis=1)

    # stockcode查询字典
    stockcode_list = tradeable_ipo.columns.to_list()
    stk_dict = stockcode_dictionary(stockcode_list)

    # 记录文件
    path0 = conf['level2_path']
    flag_path = path0 + 'csv2hdf_converted.csv'
    if re_generate:
        if break_confirm(re_generate, f'regenerate(replace) {flag_path}'):
            return
        flag_exist = pd.DataFrame()
    else:
        flag_exist = pd.read_csv(flag_path, index_col=0)
        flag_exist.columns = pd.to_datetime(flag_exist.columns)

    # 移除确认
    if break_confirm(remove_raw, 'raw csv file will be removed'):
        return

    # 遍历
    for dir_month in sorted(os.listdir(path0)):
        if '.' in dir_month:
            continue
        path1 = path0 + dir_month + '/'
        for dir_date in (sorted(os.listdir(path1))):
            if '.' in dir_date:
                continue
            tradedate = pd.to_datetime(dir_date)
            if tradedate not in flag_exist.columns:
                flag_exist[tradedate] = tradeable_ipo.loc[tradedate].replace(1, False)
            path2 = path1 + dir_date + f'/{dir_date[:4]}-{dir_date[4:6]}-{dir_date[6:]}/'
            hdf_file = path1 + dir_date + '.h5'
            print(dir_date)
            for csv_filename in tqdm(sorted(os.listdir(path2))):
                stockcode = stk_dict[csv_filename.replace('.csv', '')]
                if flag_exist.loc[stockcode, tradedate]:
                    continue
                stockcode_k = '_'.join(stockcode.split('.')[::-1])
                csv_path = path2 + csv_filename
                df = pd.read_csv(csv_path, index_col=0)
                try:
                    df.to_hdf(hdf_file, key=stockcode_k, append=True, complevel=9, complib='blosc',
                              data_columns=df.columns)
                except FileNotFoundError:
                    df.to_hdf(hdf_file, key=stockcode_k, mode='w', format='table', complevel=9, complib='blosc',
                              data_columns=df.columns)
                finally:
                    flag_exist.loc[stockcode, tradedate] = True
                    # print('SUCCESS:', dir_date, csv_filename)
                    if remove_raw:
                        os.remove(csv_path)
                        # print('REMOVED.')
                # break
        # 更新一次convert记录
        table_save_safe(df=flag_exist, tgt=flag_path, notify=True)

    print('FINISHED.')


if __name__ == '__main__':
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    re_generate = True
    remove_raw = False
    lv2_csv2hdf(conf, re_generate, remove_raw)
