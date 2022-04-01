import sys, os, time
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple

sys.path.append("/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/")

# %matplotlib inline
import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style("darkgrid")
plt.rc("figure", figsize=(16, 6))
# plt.rc("figure", figsize=(8, 3))
plt.rc("savefig", dpi=90)
# plt.rc("font", family="sans-serif")
plt.rc("font", size=12)
# plt.rc("font", size=10)
plt.rcParams["date.autoformatter.hour"] = "%H:%M:%S"


def prepare_bundle_filename_lists(bundle_path) -> Tuple[list, list, list]:
    """文件名列表"""
    bundle_files = sorted(os.listdir(bundle_path))
    train_files = [x for x in bundle_files if x[:5] == 'TRAIN']
    test_files = [x for x in bundle_files if x[:4] == 'TEST']
    desc_files = [x for x in bundle_files if x[:4] == 'DESC']
    assert len(train_files) == len(test_files) == len(desc_files)
    print(len(train_files))
    return train_files, test_files, desc_files


def get_dat_xy(dat: pd.DataFrame, x_id, y_id, rnk=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """获得x,y排名，index为日期-个股"""
    dat = dat.copy().set_index(['tradingdate', 'stockcode'])
    cols_x = [col for col in dat.columns if col[0] == x_id]
    # dat[cols_X].hist(figsize=(40, 40));  # Original FV Alpha10
    if rnk:
        rnk_x = dat[cols_x].groupby('tradingdate').rank(pct=True)
    else:
        rnk_x = dat[cols_x].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    rnk_y = dat[y_id].rename('y') if isinstance(y_id, str) else dat[y_id]

    # dat_rnk_yX = pd.concat([rnk_y, rnk_X], axis=1)
    return rnk_x, rnk_y


def plot_ls_abs_res(period_rtn, mask_l, mask_s):
    """给long条件和short条件画图"""
    rtn_l = period_rtn[mask_l].groupby('tradingdate').mean()
    rtn_s = period_rtn[mask_s].groupby('tradingdate').mean()
    rtn_l.cumsum().plot(label='long')
    rtn_s.cumsum().plot(label='short')
    (rtn_l - rtn_s).fillna(0).cumsum().plot(label='long_short')
    period_rtn.groupby('tradingdate').mean().cumsum().plot(label='baseline')
    plt.title('Long Short Absolute Result No Cost')
    plt.legend()
    plt.show()


def ridge_reg(bundle_path):
    # % 本地数据文件
    train_files, test_files, desc_files = prepare_bundle_filename_lists(bundle_path)

    # group_cnt = 0
    # %
    for group_cnt in range(len(train_files)):
        # %
        dat_train = pd.read_pickle(bundle_path + train_files[group_cnt])
        dat_test = pd.read_pickle(bundle_path + test_files[group_cnt])
        # dat_desc = pd.read_excel(bundle_path + desc_files[group_cnt], index_col=0)
        # print(dat_desc.head())
        assert dat_train.isna().sum().sum() == 0  # 已不存在空值

        x_train, y_train = get_dat_xy(dat_train, x_id='x', y_id='rnk_rtn_ot5c')
        x_test, y_test = get_dat_xy(dat_test, x_id='x', y_id='rnk_rtn_ot5c')

        # Rank IC
        # corr_yX = dat_rnk_yX.groupby('tradingdate').corr(method='spearman')
        # corr_yX['y'].unstack().mean().iloc[1:].abs().plot.bar(title='Train Period Rank IC');
        # plt.show()

        # % Model Fit
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error

        ridge_model = Ridge(alpha=0.1)
        ridge_model.fit(x_train, y_train)

        # make in-sample and out-of-sample predictions
        y_pred_train = ridge_model.predict(x_train)
        y_pred = ridge_model.predict(x_test)

        # in-sample and out-of-sample evaluation using RMSE
        print(f"In-sample RMSE: {mean_squared_error(y_train, y_pred_train, squared=False):.4f}", end='\t')
        print(f"Out-of-sample RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")

        # % Save Predictions
        pred_res = pd.DataFrame(y_pred, columns=['y_hat'], index=x_test.index)
        save_name = test_files[group_cnt].replace('TEST', 'PredRR')
        pred_res.to_pickle(bundle_path + save_name)
        print(f'SAVE {save_name} IN {bundle_path}.')

        # % Prediction Performance
        # threshold = .05
        #
        # period_rtn = dat_train.set_index(['tradingdate', 'stockcode']).rtn_ot5c / 5
        # mask_l = (y_pred_train > .5 + threshold)
        # mask_s = (y_pred_train < .5 - threshold)
        # plot_ls_abs_res(period_rtn, mask_l, mask_s)
        #
        # period_rtn = dat_test.set_index(['tradingdate', 'stockcode']).rtn_ot5c / 5
        # mask_l = (y_pred > .5 + threshold)
        # mask_s = (y_pred < .5 - threshold)
        # plot_ls_abs_res(period_rtn, mask_l, mask_s)


def merge_5d_pred(bundle_path, save_path=None, kw='PredRR'):
    """五日的预测结果合并"""
    bundle_files = sorted(os.listdir(bundle_path))
    pred_files = [x for x in bundle_files if x[:len(kw)] == kw]

    res = pd.DataFrame()
    for file in tqdm(pred_files):
        rows = pd.read_pickle(bundle_path + file)
        if rows.shape[1] == 1:
            rows = rows.iloc[:, 0].unstack()
        res = pd.concat([res, rows], axis=0)

    if save_path is not None:
        res.to_csv(save_path)

    return res


def back_roll_ridge_reg(bundle_path, y_ret_path, pred_res_path):
    """合并RidgeRegression的每五日预测收益率排名"""
    try:
        pred_ot5c = pd.read_csv(pred_res_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        pred_ot5c = merge_5d_pred(bundle_path, pred_res_path, kw='PredRR')
    rtn_ot5c = pd.read_pickle(y_ret_path).set_index(['tradingdate', 'stockcode']).rtn_ot5c
    rtn_ot5c = rtn_ot5c.unstack().loc[pred_ot5c.index[0]:pred_ot5c.index[-1]]
    pred_ot5c = pred_ot5c['y_hat'].reindex_like(rtn_ot5c)

    def decide_mask_ls(fv, th):
        ml = fv > .5 + th
        ms = fv < .5 - th
        return ml, ms

    pred_ot5c.count(axis=1).plot()
    plt.show()

    mask_l, mask_s = decide_mask_ls(pred_ot5c, .05)

    rtn_long = rtn_ot5c[mask_l].mean(axis=1).fillna(0).rename('long')
    rtn_short = rtn_ot5c[mask_s].mean(axis=1).fillna(0).rename('short')
    rtn_long_short = (rtn_long - rtn_short).rename('long_short')
    rtn_baseline = rtn_ot5c.mean(axis=1).rename('baseline')
    panel_rtn = pd.concat([rtn_long_short, rtn_baseline, rtn_long, rtn_short], axis=1)
    panel_wealth = panel_rtn.cumsum()
    panel_wealth.plot(title='Ridge Regression CSI500 ot5c')
    plt.savefig(pred_res_path.replace('factors_csv', 'factors_res').replace('.csv', '.png'))
    plt.close()


def deep_neutral_network(bundle_path):
    train_files, test_files, desc_files = prepare_bundle_filename_lists(bundle_path)

    group_cnt = 0
    # %
    for group_cnt in range(len(train_files)):
        # %
        dat_train = pd.read_pickle(bundle_path + train_files[group_cnt])
        dat_test = pd.read_pickle(bundle_path + test_files[group_cnt])
        # dat_desc = pd.read_excel(bundle_path + desc_files[group_cnt], index_col=0)
        # print(dat_desc.head())
        assert dat_train.isna().sum().sum() == 0  # 已不存在空值

        x_train, y_train = get_dat_xy(dat_train, x_id='x', y_id=['y0', 'y1', 'y2'], rnk=False)
        x_test, y_test = get_dat_xy(dat_test, x_id='x', y_id=['y0', 'y1', 'y2'], rnk=False)
        print(x_train.shape, y_train.shape)

        # % Split
        # t_train = x_train.index.get_level_values(0).unique()
        # assert len(t_train) == 1000
        # t_train_e = t_train[850]
        # t_train_v = t_train[900]

        # x_train_t = x_train.loc[:t_train_e]
        # y_train_t = y_train.loc[:t_train_e]
        # x_train_v = x_train.loc[t_train_v:]
        # y_train_v = y_train.loc[t_train_v:]

        # % Model: DNN
        from tensorflow import keras
        from keras import layers
        # from keras import utils
        from keras.models import Model
        import tensorflow as tf

        # Network
        input_unit_size = x_train.shape[1]
        num_classes = y_train.shape[1]

        model = None
        try:
            del model
        except NameError:
            pass
        np.random.seed(99)
        model = keras.Sequential(name='DNN001')
        model.add(layers.Dense(128, input_dim=input_unit_size, activation='relu', name='dense_74_128'))
        # model.add(layers.Dropout(rate=0.5))
        model.add(layers.Dense(128, activation='relu', name='dense_128_128'))
        # model.add(layers.Dropout(rate=0.4))
        model.add(layers.Dense(64, activation='relu', name='dense_128_64'))
        # model.add(layers.Dropout(rate=0.3))
        model.add(layers.Dense(64, activation='relu', name='dense_64_64'))
        # model.add(layers.Dropout(rate=0.2))
        model.add(layers.Dense(32, activation='relu', name='dense_64_32'))
        # model.add(layers.Dropout(rate=0.1))
        model.add(layers.Dense(num_classes, activation='softmax', name='dense_32_3'))

        model.summary()

        # Training
        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adadelta(),
                      metrics=['categorical_accuracy'])  # 'adam'  'adadelta'

        # result = model.fit(x_train, y_train, validation_split=0.1, epochs=50, batch_size=500)
        # result = model.fit(x_train_t, y_train_t, validation_data=(x_train_v, y_train_v) ,epochs=50, batch_size=500, callbacks=[callback])
        result = model.fit(x_train, y_train, validation_split=0, epochs=20, batch_size=500, callbacks=[callback])

        # % Save Outputs
        dense4_layer_model = Model(inputs=model.input, outputs=model.get_layer('dense_64_32').output)
        predict_train = dense4_layer_model.predict(x_train)
        predict = dense4_layer_model.predict(x_test)

        fv_train = pd.DataFrame(predict_train, index=x_train.index,
                                columns=[f'fv{x}' for x in range(predict_train.shape[1])])
        fv_test = pd.DataFrame(predict, index=x_test.index, columns=[f'fv{x}' for x in range(predict.shape[1])])

        save_name = test_files[group_cnt].replace('TEST', 'Fv2DNN{}')
        fv_train.to_pickle(bundle_path + save_name.format('train'))
        fv_test.to_pickle(bundle_path + save_name.format('test'))
        print(f'SAVE {save_name} IN {bundle_path}.')

        # % Visualization
        x = range(len(result.history['loss']))
        plt.plot(x, result.history['categorical_accuracy'], label='Accuracy')
        plt.plot(x, result.history['loss'], label='Loss')
        plt.title(train_files[group_cnt].replace('.pkl', ''))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(bundle_path + train_files[group_cnt].replace('.pkl', '.png').replace('TRAIN', 'AccLoss2'))  # plt.show()
        plt.close()


def dnn_fv_reg_combine(bundle_path, y_ret_path, kw='FvDNNtrain', kw1='PredDNN'):
    """用DNN倒数层32个机器因子，对过去250日内ot5c收益率截面回归，系数平均值，得到一个因子，5日存一个文件 PredDNN*.pkl"""
    import statsmodels.formula.api as sm

    # rtn_ot5c = pd.read_pickle(y_ret_path).set_index(['tradingdate', 'stockcode']).rtn_ot5c  # rnk_rtn_ot5c
    rtn_ot5c = pd.read_pickle(y_ret_path).set_index(['tradingdate', 'stockcode']).rnk_rtn_ot5c
    fv1k_files = sorted([x for x in os.listdir(bundle_path) if x[:len(kw)] == kw])

    group_cnt = 0
    lst_time = time_loop_start = time.time()
    for group_cnt in range(len(fv1k_files)):
        fv1y = pd.read_pickle(bundle_path + fv1k_files[group_cnt])
        # corr = fv1y.corr()
        # corr.stack().hist(bins=100); plt.show()
        fv1y = fv1y.loc[fv1y.index.get_level_values(0).unique()[-250]:]
        fv5d = pd.read_pickle(bundle_path + fv1k_files[group_cnt].replace('train', 'test'))

        dat = pd.concat([rtn_ot5c.loc[fv1y.index], fv1y], axis=1)

        def ols1d(s):
            s1 = s  # s.loc[:, s.var() != 0]
            var_ls = s1.columns.to_list()
            fm = f"""{var_ls[0]} ~ {' + '.join(var_ls[1:])}"""
            ols_res = sm.ols(formula=fm, data=s1).fit()
            # print(ols_res.summary())
            return ols_res.params

        ols_params = dat.groupby('tradingdate').apply(ols1d)
        betas = ols_params.mean().drop(index='Intercept')
        pred_res = (fv5d * betas).sum(axis=1).unstack()

        save_name = fv1k_files[group_cnt].replace(kw, kw1)
        pred_res.to_pickle(bundle_path + save_name)
        cur_time = time.time()
        print(f'SAVE {save_name} IN {bundle_path}. Loop time: {(cur_time - lst_time):.3f} s')
        lst_time = cur_time

    print(f'Finished. Total loop time: {(lst_time - time_loop_start):.3f} s')


# %%
def main():
    # %%
    import yaml
    conf_path = r'/mnt/c/Users/Winst/Nutstore/1/我的坚果云/XJIntern/PyCharmProject/config.yaml'
    conf = yaml.safe_load(open(conf_path, encoding='utf-8'))

    bundle_path = conf['train_bundle']
    csv_path = conf['factorscsv_path']
    y_ret_path = conf['alpha_y012_csi500'].format('OT5C')
    # data_path = conf['data_path']

    # %%
    # ridge_reg(bundle_path)  # 用 rank x 拟合 rank y 存结果 PredRR_20171113_20211222.pkl 五日预测值
    # merge_5d_pred(bundle_path, save_path=csv_path+'alpha101_ot5c_1k_ridge_reg.csv', kw='PredRR')
    # back_roll_ridge_reg(bundle_path, y_ret_path, pred_res_path=csv_path+'alpha101_ot5c_1k_dnn.csv')  # 用 PredRR 分两组历史表现 alpha101_ot5c_1k_ridge_reg.png
    # deep_neutral_network(bundle_path)
    dnn_fv_reg_combine(bundle_path, y_ret_path, kw='FvDNNtrain', kw1='PredDNNrnk')  # 'FvDNNtrain'  'PredDNN'
    dnn_fv_reg_combine(bundle_path, y_ret_path, kw='Fv2DNNtrain', kw1=  'Pred2DNNrnk')  # 'FvDNNtrain'  'PredDNN'
    merge_5d_pred(bundle_path, save_path=csv_path + 'alpha101_ot5c_1k_dnn_rnk.csv', kw='PredDNNrnk')
    merge_5d_pred(bundle_path, save_path=csv_path + 'alpha101_ot5c_1k_dnn2_rnk.csv', kw='Pred2DNNrnk')


# %%
if __name__ == '__main__':
    main()
