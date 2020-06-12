import moxing as mox
mox.file.shift('os', 'mox')
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

# 获取竞赛数据集：将“obs-mybucket-bj4/myfolder”改成您的OBS桶名及文件夹
import moxing as mox
mox.file.copy_parallel('s3://obs-bdc2020-bj4/traffic_flow_dataset', 's3://data.li/traffic_flow_dataset')
print('Copy procedure is completed !')

OBS_DATA_PATH = "s3://data.li/traffic_flow_dataset"
LOCAL_DATA_PATH = './dataset/train'
OBS_MODEL_DIR = "s3://data.li/modelfiles/model"
OBS_MODEL_PATH = OBS_MODEL_DIR + "/modelfile.m"
OBS_CONFIG_PATH = OBS_MODEL_DIR + "/config.json"
LOCAL_MODEL_PATH = './modelfile.m'
LOCAL_CONFIG_PATH = './config.json'

# read data of one day and one direction
def read_file(path, filename):
    calfile = os.path.join(path, filename)
    original = pd.read_csv(calfile, header=None)
    data = pd.DataFrame(columns=["time", "number"])
    data["time"] = original[0]
    data["number"] = original[3] + original[4]
    return data
    
# read data of one day
def read_data_day(path, date):
    day_data = pd.DataFrame(columns=["time","number"])
    caldir = os.path.join(path, date)
    # read data of one day
    for f in os.listdir(caldir):
        if re.match(r'wuhe_zhangheng.*\.csv', f):
            day_data = day_data.append(read_file(caldir, f), ignore_index=True)
    return day_data

# get and preprocess data
def get_data(path):
    raw_data = pd.DataFrame(columns=["time", "number"])
    for day in os.listdir(path):
        raw_data = raw_data.append(read_data_day(path, day))
        
    # encode time in raw data to weekday and timeindex(the n minutes of the day)
    df_dt = to_datetime(raw_data.loc[:, "time"], format="%Y/%m/%d %H:%M:%S")
    
    
    
    
    all_data = pd.DataFrame({
        "weekday": df_dt.dt.weekday/6.0,
        "timeindex": (df_dt.dt.hour * 60 + df_dt.dt.minute)/(24*60.0),
        "number": raw_data["number"].astype(int)}) 
    all_data = all_data.groupby(["weekday", "timeindex"]).mean().reset_index(level = ["weekday", "timeindex"])
    return all_data

def train_model():
    X_train, X_test, y_train, y_test = train_test_split(local_data[['weekday','timeindex']], local_data['number'], test_size=0.1, random_state=42)
    print("X_train shape is: " + str(X_train.shape))
    print("X_test shape is: " + str(X_test.shape))
    
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    clf = GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)
    joblib.dump(clf, LOCAL_MODEL_PATH)

    y_predict = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_predict)
    print("MSE: %.4f" % mse)
    
def create_config():
    schema_model=json.loads('{"model_algorithm":"gbtree_classification","model_type":"Scikit-learn","runtime":"python3.6","metrics":{},"apis":[{"procotol":"http","url":"/","method":"post","request":{"Content-type":"applicaton/json","data":{"type":"object","properties":{"req_data":{"type":"array","items":[{"type":"string"}]}}}},"response":{"Content-type":"applicaton/json","data":{"type":"object","properties":{"resp_data":{"type":"array","items":[{"type":"number"}]}}}}}]}',object_pairs_hook=OrderedDict)
    schema_model['model_algorithm'] = "gbtree_regression"
    schema_model['model_type'] = "Scikit_Learn"
    with open(LOCAL_CONFIG_PATH, 'w') as f:
        json.dump(schema_model, f)
    
if __name__ == "__main__":
    # copy data from obs to local
    mox.file.copy_parallel(OBS_DATA_PATH, LOCAL_DATA_PATH)
    # read and preprocess data
    local_data = get_data(LOCAL_DATA_PATH)
    # train model
    train_model()
    # create config.json
    create_config()
    # upload model to obs
    mox.file.copy(LOCAL_MODEL_PATH, OBS_MODEL_PATH)
    mox.file.copy(LOCAL_CONFIG_PATH, OBS_CONFIG_PATH)
    print("Model training has been completed!")