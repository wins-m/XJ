"""
(created by swmao on Jan. 28th)
I/O支持

"""
import time


def break_confirm(bo, msg):
    """返回1，强制终止"""
    res = True
    cmd = 'N'
    if bo:
        cmd = input(f'{msg}, confirm?')
        if cmd == 'Y' or cmd == 'y':
            res = False
    # print(cmd, res)
    return res


def table_save_safe(df, tgt, kind='csv', notify=False):
    if kind == 'csv':
        try:
            df.to_csv(tgt)
        except PermissionError:
            print(f'Permission Error: saving `{tgt}`, retry in 5 seconds...')
            time.sleep(5)
            table_save_safe(df, tgt, kind)
        finally:
            if notify:
                print(f'{df.name} saved in `{tgt}`.')
    else:
        raise ValueError(f'kind `{kind}` not supported.')
