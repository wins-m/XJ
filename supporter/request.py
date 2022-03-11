"""
(created by swmao on March 10th)
From config file request calculated frame from local or remote.
- get_hold_return(conf, ret_kind, bd, ed, stk_pool) :
-

"""
import pandas as pd
import numpy as np
from datetime import timedelta


def get_hold_return(conf: dict, ret_kind='ctc', bd=None, ed=None, stk_pool='NA') -> pd.DataFrame:
    """Return a frame of nominal holding returns."""
    from supporter.stk_pool import get_stk_pool, keep_in_stk_pool

    # TODO: probably not attainable when selling for reasons like suspend or max_down

    print(f'Request holding return (kind={ret_kind})...')
    k0, k1 = ret_kind.split('t')

    bd = '2010-01-01' if bd is None else bd  # TODO: first day return is NA
    ed = '2022-12-31' if ed is None else ed

    suspend = pd.DataFrame(pd.read_hdf(conf['a_list_tradeable'], key='suspend')).replace(False, np.nan).loc[bd: ed]
    ipo60 = pd.DataFrame(pd.read_hdf(conf['a_list_tradeable'], key='ipo60')).replace(False, np.nan).loc[bd: ed]

    in_stk_pool: pd.DataFrame = get_stk_pool(conf, stk_pool, bd, ed)
    suspend = keep_in_stk_pool(suspend, in_stk_pool)
    ipo60 = keep_in_stk_pool(ipo60, in_stk_pool)

    if k0 == 'c':
        p0 = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True).shift(1).loc[bd: ed]  # long T-1 close
        p0 *= suspend.shift(1)  # not necessary
        max_up = pd.DataFrame(pd.read_hdf(conf['a_list_tradeable'], key='up')).replace(False, np.nan).loc[bd: ed]
        max_up = keep_in_stk_pool(max_up, in_stk_pool)
        p0 *= max_up.shift(1)
        p0 *= ipo60.shift(1)
    elif k0 == 'o':
        p0 = pd.read_csv(conf['openAdj'], index_col=0, parse_dates=True).shift(0).loc[bd: ed]  # long T0 open
        p0 *= suspend  # (may) not necessary
        max_up_open = pd.DataFrame(pd.read_hdf(conf['a_list_tradeable'], key='up_open')).replace(False, np.nan).loc[
                     bd: ed]
        max_up_open = keep_in_stk_pool(max_up_open, in_stk_pool)
        p0 *= max_up_open
        p0 *= ipo60
    else:
        raise AttributeError(f'Invalid return kind: {ret_kind}!')

    if len(k1) == 1:
        hd = 1
    elif len(k1) == 2:
        hd = int(k1[0])
        k1 = k1[1]
    else:
        raise AttributeError(f'Invalid return kind: {ret_kind}!')

    if k1 == 'c':
        p1 = pd.read_csv(conf['closeAdj'], index_col=0, parse_dates=True).shift(1 - hd).loc[bd: ed]  # short T+{hd - 1} close
    elif k1 == 'o':
        p1 = pd.read_csv(conf['openAdj'], index_col=0, parse_dates=True).shift(-hd).loc[bd: ed]  # short T+{hd} open
    else:
        raise AttributeError(f'Invalid return kind: {ret_kind}!')

    rtn_log = p1.applymap(np.log) - p0.applymap(np.log)
    return rtn_log


def get_ind_citic_all_tradingdate(conf: dict, begin_date, end_date) -> pd.DataFrame:
    """Get citic industry label, index is all tradingdate from begine_date to end_date"""
    bd, begin_date, ed = pd.to_datetime(begin_date) - timedelta(60), pd.to_datetime(begin_date), pd.to_datetime(end_date)
    ind_citic = pd.read_csv(conf['ind_citic'], index_col=0, parse_dates=True, dtype=object).loc[bd:ed]
    tdays_d = pd.read_csv(conf['tdays_d'], header=None, index_col=0, parse_dates=True).loc[bd:ed]
    tdays_d = tdays_d.reset_index().rename(columns={0:'tradingdate'})
    ind_citic = ind_citic.reset_index().merge(tdays_d, on='tradingdate', how='right')
    ind_citic = ind_citic.set_index('tradingdate').fillna(method='ffill').loc[begin_date:]

    return ind_citic


def get_sector_ci_all_tradingdate(conf: dict, begin_date, end_date) -> pd.DataFrame:
    """
    Request a frame of 2d-ci-sector-label, all tradingdate within range, domain: {1, 2, 3, 4}, ffill and then backfill
    :param conf:
    :param begin_date:
    :param end_date:
    :return:
    """
    bd, begin_date, ed = pd.to_datetime(begin_date) - timedelta(60), pd.to_datetime(begin_date), pd.to_datetime(end_date)
    sector_ci = pd.read_csv(conf['sector_constituent'], index_col=0, parse_dates=True, dtype=object).loc[bd:ed]
    tdays_d = pd.read_csv(conf['tdays_d'], header=None, index_col=0, parse_dates=True).loc[bd:ed]
    tdays_d = tdays_d.reset_index().rename(columns={0:'tradingdate'})
    sector_ci = sector_ci.reset_index().merge(tdays_d, on='tradingdate', how='right')
    sector_ci = sector_ci.set_index('tradingdate').fillna(method='ffill').fillna(method='bfill').loc[begin_date:]
    return sector_ci
