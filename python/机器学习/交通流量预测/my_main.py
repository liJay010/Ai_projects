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
from sklearn.externals import joblib
from collections import OrderedDict
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
import lightgbm as lgb

# 获取竞赛数据集：将“obs-mybucket-bj4/myfolder”改成您的OBS桶名及文件夹


LOCAL_DATA_PATH =  r"C:\Users\ASUS\Desktop\数据挖掘\交通流量预测\traffic_flow_dataset"

# read data of one day and one direction
def read_file(path, filename):
    calfile = os.path.join(path, filename)
    original = pd.read_csv(calfile, header=None)
    data = pd.DataFrame()
    data["time"] = original[0]
    data["direction"] = original[2]
    data["left_number"] = original[3]
    data["straight_number"] = original[4]
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
def get_data(path):
    raw_data = pd.DataFrame()
    for day in os.listdir(path):
        raw_data = raw_data.append(read_data_day(path, day))
    
    # encode time in raw data to weekday and timeindex(the n minutes of the day)
    df_dt= to_datetime(raw_data.loc[:, "time"], format="%Y/%m/%d %H:%M:%S")
#    
#    raw_data['mounth'] = df_dt.dt.month
#    raw_data['day'] = df_dt.dt.day
#    raw_data["timeindex"] = (df_dt.dt.hour * 60 + df_dt.dt.minute)/5 #5分钟编码
#    raw_data["weekday"] = df_dt.dt.weekday + 1
#    raw_data.to_csv("sum.csv",index = None)
#    return 0
    raw_data = raw_data.drop("time",axis = 1)
    all_data = raw_data.copy()
#    all_data["weekday"] = df_dt.dt.weekday/6.0
    all_data["weekday"] = df_dt.dt.weekday / 6 #具体星期几
    all_data["is_weekday"] = df_dt.dt.weekday.map(lambda x: 1 if x == 5 or x == 6 else 0) #是否周末
    all_data = pd.concat([all_data, pd.get_dummies(all_data['weekday'], prefix='weekday')], axis=1)  # one-hot离散星期几
    all_data["timeindex"] = (df_dt.dt.hour * 60 + df_dt.dt.minute)/(24*60.0) #5分钟编码
    all_data["is_holiday"] = df_dt.map(lambda x: 1 if (x.day in [13,4,5,6,7,8,28]) or (x.day == 14 and x.month == 2)  else 0)# 是否为节日
    all_data["number"] = raw_data["number"].astype(int)
    time_of_day = [] # 上午/下午/晚上分类
    for ti in df_dt:
        if ti.hour < 12:
            time_of_day.append(0)
        elif 18>ti.hour >= 12:
            time_of_day.append(1/2)
        else :
            time_of_day.append(2/2)
    all_data['time_of_day']=time_of_day

    """
    对方向进行编码
    """
    #set(all_data['direction'])
    dri = []
    for d in all_data["direction"]:
        if d=='east' :
            dri.append(1/4)
        elif d=='north' :
            dri.append(2/4)
        elif d=='south' :
            dri.append(3/4)
        else:
            dri.append(4/4)
            
    all_data["direction"] = dri
#    all_data = all_data .groupby(["weekday", "timeindex","is_weekday"]).mean().reset_index(level = ["weekday", "timeindex","is_weekday"])
    
    """
    天气情况
    """
    

    
    
    
    
#    all_data = all_data.groupby(["weekday", "timeindex"]).mean().reset_index(level = ["weekday", "timeindex"])
    
#    avg_time_num = all_data.groupby(["weekday", "timeindex"]).mean()
    

    return all_data
local_data = get_data(LOCAL_DATA_PATH)


def train_model():
    y = local_data['number']
    x = local_data.drop(['number','left_number','straight_number'],axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    print("X_train shape is: " + str(X_train.shape))
    print("X_test shape is: " + str(X_test.shape))
    
    params = {'num_leaves': 64,
          'min_data_in_leaf': 50,
          'objective': 'regression',
          'max_depth': 8,
          'learning_rate': 0.1,
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
    
    trn_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_test,  y_test,reference=trn_data)
    clf = lgb.train(params,
                    trn_data,
                    num_boost_round=5000,
                    valid_sets=[trn_data, val_data],
#                    verbose_eval=200,
                    early_stopping_rounds=500)
    y_predict = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_predict)
    print("MSE: %.4f" % mse)
#    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
#              'learning_rate': 0.01, 'loss': 'ls'}
#    clf = GradientBoostingRegressor(**params)
#    clf.fit(X_train, y_train)
#
#    y_predict = clf.predict(X_test)
#    mse = mean_squared_error(y_test, y_predict)
#    print("MSE: %.4f" % mse)
    


if __name__ == "__main__":
    # copy data from obs to local
    # read and preprocess data
    local_data = get_data(LOCAL_DATA_PATH)
    # train model
    train_model()
    # create config.json
    # upload model to obs
    print("Model training has been completed!")
    local_data.to_csv("sum.csv",index = None)