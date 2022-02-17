# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 15:06:10 2022

@author: jeffr

测试的时候，可以设置输入参数TargetDate = dt.date.today() - dt.timedelta(days=1)

"""

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import datetime as dt
import logging

from WindPy import w


# 中信一级行业行情数据 from WIND
def update_idx_price(TargetDate, engine, w, logger=None):
    
    CurrentTD = TargetDate.strftime('%Y-%m-%d')
    
    # 读取行业列表 & 行情
    Ind_List = pd.read_sql_query("select industry, industryname from jeffdatabase.ind_list_citic where level=1", engine)
    
    fields_list = ['pre_close', 'open', 'high', 'low', 'close', 'volume', 'amt']
    
    for target_field in fields_list:
        data_add = w.wsd(list(Ind_List['industry']), target_field, CurrentTD, CurrentTD, "", usedf=True)
        if data_add[0] != 0:
            raise ValueError("WIND数据读取失败，错误代码"+str(data_add[0]))
        data_add = data_add[1]
        # 合并数据
        data_add.reset_index(inplace=True)
        data_add.rename(columns={'index':'industry'},inplace=True)
        Ind_List = pd.merge(Ind_List, data_add, on='industry', how='left')
    
    Ind_List['td'] = CurrentTD
    Ind_List.rename(columns={'industry':'code', 'industryname':'name'},inplace=True)
    
    # 写入数据库
    pd.io.sql.to_sql(Ind_List, 'idx_price', engine, schema='stra_ind', if_exists='append', index=False)
    
    if logger is None:
        print("stra_ind.idx_price成功导入"+CurrentTD+"数据"+str(Ind_List.shape[0])+"条..")
    else:
        logger.info("stra_ind.idx_price成功导入"+CurrentTD+"数据"+str(Ind_List.shape[0])+"条..")