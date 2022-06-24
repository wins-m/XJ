"""
(created by swmao on Jan. 28th)
I/O支持

"""
import os
import time
import pandas as pd


def break_confirm(bo, msg):
    """返回1，强制终止"""
    res = False
    cmd = 'N'
    if bo:
        res = True
        cmd = input(f'{msg}, confirm?')
        if cmd == 'Y' or cmd == 'y':
            res = False
    # print(cmd, res)
    return res


def table_save_safe(df: pd.DataFrame, tgt: str, kind=None, notify=False, **kwargs):
    """
    安全更新已有表格（当tgt在磁盘中被打开，5秒后再次尝试存储）
    :param df: 表格
    :param tgt: 目标地址
    :param kind: 文件类型，暂时仅支持csv
    :param notify: 是否
    :return:
    """
    kind = tgt.rsplit(sep='.', maxsplit=1)[-1] if kind is None else kind

    if kind == 'csv':
        func = df.to_csv
    elif kind == 'xlsx':
        func = df.to_excel
    elif kind == 'pkl':
        func = df.to_pickle
    elif kind == 'h5':
        if 'key' in kwargs:
            hdf_k = kwargs['key']
        elif 'k' in kwargs:
            hdf_k = kwargs['k']
        else:
            raise Exception('Save FileType hdf but key not given in table_save_safe')

        def func():
            df.to_hdf(tgt, key=hdf_k)
    else:
        raise ValueError(f'Save table filetype `{kind}` not supported.')

    try:
        func(tgt)
    except PermissionError:
        print(f'Permission Error: saving `{tgt}`, retry in 5 seconds...')
        time.sleep(5)
        table_save_safe(df, tgt, kind)
    finally:
        if notify:
            print(f'{df.shape} saved in `{tgt}`.')


def get_time_suffix(suffix: str = None):
    """Return a string '%m%d_%H%M%S'"""
    if suffix is None or suffix == '':
        return time.strftime("%m%d_%H%M%S", time.localtime())
    return suffix


def io_make_sub_dir(path, force=False, inp=False):
    if force:
        os.makedirs(path, exist_ok=True)
    else:
        if os.path.exists(path):
            if os.path.isdir(path) and len(os.listdir(path)) == 0:
                return 1
            else:
                if inp:
                    return 0
                cmd = input(f"Write in non-empty dir '{path}' ?(y/N)")
                if cmd != 'y' and cmd != 'Y':
                    raise FileExistsError(path)
        else:
            os.makedirs(path, exist_ok=False)
    return 1


