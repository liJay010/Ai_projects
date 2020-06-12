# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
data = pd.read_csv("data.csv")
test = pd.read_csv("test.csv")
label = data['Label']
data = data.drop('Label',1)



def process(data):
    data['max'] = data.iloc[:,0:3151].max(axis = 1)
    data['sum'] = data.iloc[:,0:3151].sum(axis = 1)
#    data['sum_of_zero'] = (data.iloc[:,0:3151] == 0).sum(axis = 1)
#    data['max_of_loc'] = np.array(data.iloc[:,0:3151]).argmax(axis = 1)
#    data['mean'] = data.iloc[:,0:3151].mean(axis = 1)
#    data['var'] = data.iloc[:,0:3151].var(axis = 1)

#    for i in range(1,data.iloc[:,0:3151].max(axis = 1).max()+1):
#        name = 'sum_of_' + str(i)
#        name2 = 'egst' +str(i)
#        name3 = 'west' +str(i)
#        data[name] = (data.iloc[:,0:3151] == i).sum(axis = 1)
#        data[name2] = (((data.iloc[:,0:3151] == i).sum(axis = 1)) >1).astype('int8')
#        data[name3] = (data.iloc[:,0:3151] == i).sum(axis = 1) * i
    #不为0的个数
    data['not_0'] = 3151 - (data.iloc[:,0:3151] == 0).sum(axis = 1)
    
    
    
    return data

data = process(data)
test = process(test)

#def process_one(data):
#    for i in range(0,3150,10):
#        name = 'range_'+str(i)
#        data[name] = data.iloc[:,i:i+10].sum(axis = 1)
#    return data
#
#data = process_one(data)
#test = process_one(test)

"""
寻找0元素,并去除
"""
zero = []
for i in range(3172):
    if data.iloc[:,i].max() == 0 and data.iloc[:,i].min() ==0 :
        zero.append(i)
zero_test = []
for i in range(3172):
    if test.iloc[:,i].max() == 0 and test.iloc[:,i].min() ==0 :
        zero_test.append(i)

zero_all = zero or zero_test
for i in range(len(zero_all)):
    zero_all[i] = str(zero_all[i])
    
test = test.drop(zero_all,1)
data =data.drop(zero_all,1)
data.columns = [str(i) for i in range(data.shape[1])]
test.columns = [str(i) for i in range(test.shape[1])]





def calc_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())



"""
均值化
"""

"""
lgb
"""
# specify your configurations as a dict

params = {'num_leaves': 38,
          'min_data_in_leaf': 50,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.02,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.7,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 4,
           "metric": 'rmse',
          "random_state": 2019,
          # 'device': 'gpu'
          }


#folds = KFold(n_splits=4, shuffle=True, random_state=2019)
#oof = np.zeros(data.shape[0])
#predictions = np.zeros(test.shape[0])
#
##label_l = np.log1p(label) # Data smoothing
#
#
#feature_importance_df = pd.DataFrame()
#for fold_, (trn_idx, val_idx) in enumerate(folds.split(data)):
#    print("fold {}".format(fold_ + 1))
#    trn_data = lgb.Dataset(data.iloc[trn_idx], label=label.iloc[trn_idx])
#    val_data = lgb.Dataset(data.iloc[val_idx], label=label.iloc[val_idx])
#
#
#    clf = lgb.train(params,
#                    trn_data,
#                    num_boost_round=5000,
#                    valid_sets=[trn_data, val_data],
##                    verbose_eval=200,
#                    early_stopping_rounds=200)
#    oof[val_idx] = clf.predict(data.iloc[val_idx], num_iteration=clf.best_iteration)
#
#    fold_importance_df = pd.DataFrame()
##    fold_importance_df["Feature"] = features
#    fold_importance_df["importance"] = clf.feature_importance()
#    fold_importance_df["fold"] = fold_ + 1
#    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
#
#    predictions += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits
#
#
#
#
#m = pd.DataFrame()
#m['Label'] = predictions
#m.to_csv("lgb3.csv")



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = \
                train_test_split(data,label,test_size=0.25,random_state=100)
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=4000,
                valid_sets=lgb_eval,
                early_stopping_rounds=200,
                )




zero = []
one = []
impo = gbm.feature_importance()
for i in range(len(impo)):
    if impo[i] == 0:
        zero.append(str(i))
    if impo[i] == 1:
        one.append(str(i))
data.columns = [str(i) for i in range(len(impo))]

print('Starting predicting...')
# predict
y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
# eval
calc_rmse(y_pred, y_test)





m = pd.DataFrame()
m['Label'] = y_pred 
m.to_csv("lgb4.csv")





           











































#from sklearn import preprocessing
#data_p=preprocessing.scale(data)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = \
                train_test_split(data,label,test_size=0.2,random_state=100)

#lgb_train = lgb.Dataset(data, label)
#lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
#
#
                
a = np.exp(y_train)
b = np.log
lgb_train = lgb.Dataset(x_train,  y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
     
label.min()
label.max()

"""
xgboost
"""
params = {'booster': 'gbtree',
      'objective':'reg:gamma',
      'eval_metric': 'rmse',
      'silent': 1,
      'eta': 0.05,
      'max_depth': 5,
      'min_child_weight': 1,
      'gamma': 0,
      'lambda': 1,
      'colsample_bylevel': 0.6,
      'colsample_bytree': 0.6,
      'subsample': 0.8,
      'scale_pos_weight': 1}


dtrain = xgb.DMatrix(x_train,y_train + 10)
# 训练
watchlist = [(dtrain, 'train')]
model = xgb.train(params, dtrain, num_boost_round=3000, evals=watchlist,early_stopping_rounds=200)
# 预测
dtest = xgb.DMatrix(x_test)
predict = model.predict(dtest) -10
calc_rmse(predict, y_test)



























dtrain = xgb.DMatrix(data, label +10)
# 训练
watchlist = [(dtrain, 'train')]
model = xgb.train(params, dtrain, num_boost_round=3000, evals=watchlist,early_stopping_rounds=200)
# 预测
dtest = xgb.DMatrix(test)
predict = model.predict(dtest) -10
m = pd.DataFrame()
m['Label'] = predict
m.to_csv("5-0.05-3000.csv")