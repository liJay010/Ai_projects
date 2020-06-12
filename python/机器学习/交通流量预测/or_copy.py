
# -*- coding: utf-8 -*-
import os
import re
import json
import pandas as pd
from pandas import to_datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from collections import OrderedDict
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
LOCAL_DATA_PATH =  r"C:\Users\ASUS\Desktop\数据挖掘\交通流量预测\traffic_flow_dataset"


# read data of one day and one direction
def read_file(path, filename):
    calfile = os.path.join(path, filename)
    original = pd.read_csv(calfile, header=None)
    data = pd.DataFrame()
    data["time"] = original[0]
    data["left"] = original[3]
    data['straight'] = original[4]
    data["number"] = original[3] + original[4]
    return data
    
# read data of one day
def read_data_day(path, date):
    day_data = pd.DataFrame()
    caldir = os.path.join(path, date)
    # read data of one day
    for f in os.listdir(caldir):
        if re.match(r'wuhe_zhangheng.*\.csv', f):
            day_data = day_data.append(read_file(caldir, f), ignore_index=True)
    return day_data

# get and preprocess data
def get_data(path = r"C:\Users\ASUS\Desktop\数据挖掘\交通流量预测\traffic_flow_dataset" ):
    raw_data = pd.DataFrame()
    for day in os.listdir(path):
        raw_data = raw_data.append(read_data_day(path, day))
    # encode time in raw data to weekday and timeindex(the n minutes of the day)
    df_dt = to_datetime(raw_data.loc[:, "time"], format="%Y/%m/%d %H:%M:%S")
    raw_data["timeindex"] = (df_dt.dt.hour * 60 + df_dt.dt.minute)/(24*60.0) #5分钟编码
    all_data = raw_data.copy()
    all_data["weekday"] = df_dt.dt.weekday / 6 #具体星期几
    all_data["number"] = raw_data["number"].astype(int)
    all_data = all_data.groupby(["weekday", "timeindex"]).mean().reset_index(level = ["weekday", "timeindex"])
    return all_data

def train_model():
    X_train, X_test, y_train, y_test = train_test_split(local_data, local_data, test_size=0.1, random_state=42)
    params = {'num_leaves': 64,
          'min_data_in_leaf': 50,
          'objective': 'regression',
          'max_depth': 8,
          'learning_rate': 0.2,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.8,
          "bagging_freq": 1,
          "bagging_fraction": 0.7,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "reg_alpha":0.5,
          "verbosity": -1,
          "nthread": 4,
          "reg_lambda":0.5,
          "subsample":0.8,
           "metric": 'rmse',
          "random_state": 2019,
          # 'device': 'gpu'
          }
    """
    all
    """
    x_train = X_train[['weekday', 'timeindex']]
    Y_train = y_train['number']
    x_test = X_test[['weekday', 'timeindex']]
    Y_test = y_test['number']
    Y_all_test = Y_test
    trn_data = lgb.Dataset(x_train, Y_train)
    val_data = lgb.Dataset(x_test,  Y_test,reference=trn_data)

    clf = lgb.train(params,
                    trn_data,
                    num_boost_round=5000,
                    valid_sets=[trn_data, val_data],
#                    verbose_eval=200,
                    early_stopping_rounds=500)
    

    y_predict = clf.predict(x_test)
    mse = mean_squared_error(Y_test, y_predict)
    print("MSE: %.4f" % mse)
    
    
    """
    left
    """
    x_train = X_train[['weekday', 'timeindex']]
    Y_train = y_train['left']
    x_test = X_test[['weekday', 'timeindex']]
    Y_test = y_test['left']
    trn_data = lgb.Dataset(x_train, Y_train)
    val_data = lgb.Dataset(x_test,  Y_test,reference=trn_data)

    clf = lgb.train(params,
                    trn_data,
                    num_boost_round=5000,
                    valid_sets=[trn_data, val_data],
#                    verbose_eval=200,
                    early_stopping_rounds=500)
    

    y_predict_left = clf.predict(x_test)
    mse = mean_squared_error(Y_test, y_predict_left)
    print("MSE: %.4f" % mse)
    
    
    
    """
    straight
    """
    x_train = X_train[['weekday', 'timeindex']]
    Y_train = y_train['straight']
    x_test = X_test[['weekday', 'timeindex']]
    Y_test = y_test['straight']
    trn_data = lgb.Dataset(x_train, Y_train)
    val_data = lgb.Dataset(x_test,  Y_test,reference=trn_data)

    clf = lgb.train(params,
                    trn_data,
                    num_boost_round=5000,
                    valid_sets=[trn_data, val_data],
#                    verbose_eval=200,
                    early_stopping_rounds=500)
    

    y_predict_straight = clf.predict(x_test)
    mse = mean_squared_error(Y_test, y_predict_straight)
    print("MSE: %.4f" % mse)
    
    
    y_all_predict = y_predict_straight + y_predict_left
    mse = mean_squared_error(Y_all_test, y_all_predict)
    print("MSE: %.4f" % mse)
#    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
#              'learning_rate': 0.01, 'loss': 'ls'}
#    clf = GradientBoostingRegressor(**params)
#    clf.fit(X_train, y_train)
#
#    y_predict = clf.predict(X_test)
#    mse = mean_squared_error(y_test, y_predict)
#    print("MSE: %.4f" % mse)
#    

    
if __name__ == "__main__":
    # read and preprocess data
    local_data = get_data(LOCAL_DATA_PATH)
    # train model
    train_model()
    # create config.json
