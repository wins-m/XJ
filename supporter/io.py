"""
(created by swmao on Jan. 28th)
I/O支持

"""
import time


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


def table_save_safe(df, tgt, kind='csv', notify=False):
    """
    安全更新已有表格（当tgt在磁盘中被打开，5秒后再次尝试存储）
    :param df: 表格
    :param tgt: 目标地址
    :param kind: 文件类型，暂时仅支持csv
    :param notify: 是否
    :return:
    """
    if kind == 'csv':
        try:
            df.to_csv(tgt)
        except PermissionError:
            print(f'Permission Error: saving `{tgt}`, retry in 5 seconds...')
            time.sleep(5)
            table_save_safe(df, tgt, kind)
        finally:
            if notify:
                print(f'{df.shape} saved in `{tgt}`.')
    else:
        raise ValueError(f'kind `{kind}` not supported.')


def get_time_suffix(suffix: str = None):
    """Return a string '%m%d_%H%M%S'"""
    if suffix is None or suffix == '':
        return time.strftime("%m%d_%H%M%S", time.localtime())
    return suffix