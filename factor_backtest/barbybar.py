import numpy as np
import pandas as pd


def get_idx_data(field: str, idx_name: str, dates_list: list) -> pd.DataFrame:
    pass


def get_data(field: str, dates_list: list, security_list: list) -> pd.DataFrame:
    pass


def get_stock_limit(security_list: list, dates_list: list) -> pd.DataFrame:
    pass


def barbybar(weight_df: pd.DataFrame(), limit=None, benchmark: str = 'CSI500',
             Close_df: pd.DataFrame() = None, Volume_df: pd.DataFrame() = None,
             Open_df: pd.DataFrame() = None, PredClose_df: pd.DataFrame() = None):
    benchmark = get_idx_data(field='CLOSE_RETURN', idx_name=benchmark, dates_list=weight_df.index.to_list()).values

    weight_df = weight_df.shift(1).fillna(0)  # 滞后一天
    Wopt = weight_df.fillna(0).values()
    if Close_df is None:
        Close_df = get_data(field='CLOSE_PRICE_2', dates_list=weight_df.index.tolist(),
                            security_list=weight_df.columns.tolist())

    if Volume_df is None:
        Volume_df = get_data(field='TURNOVER_VOL', dates_list=weight_df.index.tolist(),
                            security_list=weight_df.columns.tolist())

    if Open_df is None:
        Open_df = get_data(field='OPEN_PRICE_2', dates_list=weight_df.index.tolist(),
                       security_list=weight_df.columns.tolist())
    if PredClose_df is None:
        PredClose_df = get_data(field='PRE_CLOSE_PRICE_2', dates_list=weight_df.index.tolist(),
                            security_list=weight_df.columns.tolist())

    Close = Close_df.dropna(how='all').ffill().bfill().fillna(0).values
    Volume = Volume_df.dropna(how='all').ffill().fillna(0).values
    Open = Open_df.dropna(how='all').ffill().bfill().fillna(0).values
    PredClose = PredClose_df.dropna(how='all').ffill().bfill().fillna(0).values

    OC = Open / PredClose - 1
    T = Wopt.shape[0] - 1
    n = Wopt.shape[1]

    if limit is None:
        Stock_Limit = get_stock_limit(security_list=weight_df.columns.tolist(), dates_list=weight_df.index.tolist())
        BarLimitUp = Stock_Limit.to_numpy()
        BarLimitDown = -1 * BarLimitUp

    nstart = 1
    capital = 1e13
    MultiplierEquity = 100
    Ncontract = np.zeros(shape=(T+1, n))  # target number of contracts/stocks
    NcontracT = np.zeros(shape=(T, n))  # actural number of contracts/stocks
    Xt = np.zeros(shape=(T, n))  # execution price
    PnL_Equity = np.zeros(shape=(T, n))  # Equity PnL
    PnL = np.zeros(T)  # Total PnL
    PortValue = np.full(T, capital)  # Portfolio Net Asset Value
    UpLimitHit = np.zeros(shape=(T, n))  # 涨停板
    DownLimitHit = np.zeros(shape=(T, n))  # 跌停板
    SuspendIndi = np.zeros(shape=(T, n))  #

    for t in range(nstart - 1, T):
        pass