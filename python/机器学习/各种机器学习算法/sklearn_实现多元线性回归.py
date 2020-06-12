# -*- coding: utf-8 -*-

import pandas as  pd
data  = pd.read_excel(r'E:\python_learning\code\project\机器学习-线性回归\files\多元线性回归data.xlsx')
#打印相关系数矩阵
data[['体重','年龄','血压收缩']].corr()
from sklearn.linear_model import LinearRegression
#实例化一个对象
lr_model = LinearRegression()
#数据切分
x= data[['体重','年龄']]
y = data[['血压收缩']]
#数据训练
lr_model.fit(x,y)

#评判模型 -- 调整R2
lr_model.score(x,y)

#查看参数

k = lr_model.coef_
b = lr_model.intercept_

#预测
lr_model.predict([80,60])
lr_model.predict([[70,30],[70,20]])

print('线性回归模型 :y = {}x1 + {}x2 + {}'.format(k[0][0],k[0][1],b[0]))