# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from xgboost import plot_importance
from sklearn.preprocessing import Imputer


def loadDataset(filePath):
    df = pd.read_csv(filepath_or_buffer=filePath)
    return df




def loadTestData(data,p):
    Features = data['Features']
    f = []
    for i in range(len(Features)):
        
        a= Features[i].split(",")
        b = []
        for j in a:
            if "[" in j:
                b.append(eval(j[1:]))
            elif "]" in j:
                b.append(eval(j[:-1]))
            elif j==" nan":
                b.append(0)
            else:
                b.append(eval(j))
        f.append(b)
        print(i)
    Features = pd.DataFrame(f)
    Features["Molecule_max_phase"] = data['Molecule_max_phase']
    Features["Molecular weight"] = data['Molecular weight']
    Features["RO5_violations"] = data['RO5_violations']
    Features["AlogP"] = data['AlogP']
    if p == 1:
        return Features, data['Label']
    else:
        return Features


def trainandTest(X_train, y_train):
    # XGBoost训练过程
#    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
    params = {'booster': 'gbtree',
          'objective':'reg:gamma',
          'eval_metric': 'rmse',
          'silent': 1,
          'eta': 0.01,
          'max_depth': 4,
          'min_child_weight': 1,
          'gamma': 0,
          'lambda': 1,
          'colsample_bylevel': 0.6,
          'colsample_bytree': 0.6,
          'subsample': 0.8,
          'scale_pos_weight': 1}
    # 数据集
    data[data['Label'] < 0]=0
    dtrain = xgb.DMatrix(X_train,label= data['Label'])
    # 训练
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist)
    # 预测
    F = f_regression(X_train,data['Label'])
    
    dtest = xgb.DMatrix(test)
    predict = model.predict(dtest)
    
    m = pd.DataFrame()
    m['Label'] = predict
    m.to_csv("a.csv")


    

if __name__ == '__main__':
    trainFilePath = r"E:\Sougou\Molecule_prediction\Molecule_prediction\train.csv"
    testFilePath = r"E:\Sougou\Molecule_prediction\Molecule_prediction\test_noLabel.csv"
#    testFilePath = 'dataset/soccer/test.csv'
    data = loadDataset(trainFilePath)
    testFilePath = loadDataset(testFilePath)
    X_train, y_train = loadTestData(testFilePath,1)
    test = loadTestData(testFilePath,0)
    trainandTest(X_train, y_train, test)