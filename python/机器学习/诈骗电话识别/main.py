# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def prepare(dataset):
    data = dataset.copy()
    data['city_name'] = data['city_name'].map(str)
    data['county_name'] = data['county_name'].map(str)
    
    
    city = list(set(list(data['city_name'])))
    city_dict = dict()
    for i in range(len(city)):
        city_dict[city[i]] = i
    
        
    county = list(set(list(data['county_name'])))
    county_dict = dict()
    for j in range(len(county)):
        county_dict [county[j]] = j
    
    
    for i in range(len(data)):
        
        data.iloc[i,1] = city_dict[data.iloc[i,1]]
        data.iloc[i,2] = county_dict[data.iloc[i,2]]
        
    return data;

    
    
    

data_app = pd.read_csv(r'E:\竞赛\诈骗电话识别\诈骗电话识别\诈骗电话号码识别-0527\train\train_app.csv')
data_sms = pd.read_csv(r'E:\竞赛\诈骗电话识别\诈骗电话识别\诈骗电话号码识别-0527\train\train_sms.csv')
data_voc = pd.read_csv(r'E:\竞赛\诈骗电话识别\诈骗电话识别\诈骗电话号码识别-0527\train\train_voc.csv')
data_user = pd.read_csv(r'E:\竞赛\诈骗电话识别\诈骗电话识别\诈骗电话号码识别-0527\train\train_user.csv')


data = prepare(data_user)

