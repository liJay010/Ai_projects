# -*- coding: utf-8 -*-

import pandas as  pd
data  = pd.read_excel(r'E:\python_learning\code\project\机器学习-线性回归\files\多元线性回归data.xlsx')

#数据切分
x= data[['体重','年龄']]
y = data[['血压收缩']]

import statsmodels.api as sm
#添加常数项
X = sm.add_constant(x)

#最小二乘法

model = sm.OLS(y,X)
result = model.fit()

#系数
result.params
"""
常用的置信水平0.05
"""
#汇总结果
result.summary()

#预测
y_pr = result.fittedvalues