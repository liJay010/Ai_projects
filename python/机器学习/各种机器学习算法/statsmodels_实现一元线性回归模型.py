# -*- coding: utf-8 -*-

import pandas as pd
data = pd.read_excel(r"E:\python_learning\code\project\机器学习-线性回归\files\data.xlsx")

x = data[['大小']]
y = data[['房价']]

import statsmodels.api as sm
#添加常数项
X = sm.add_constant(x)

#最小二乘法

model = sm.OLS(y,X)
result = model.fit()
#系数
result.params
#汇总结果
result.summary()

#预测
y_pr = result.fittedvalues

#绘图

import matplotlib.pyplot as plt

fig,ax = plt.subplots(figsize = (8,6))

ax.plot(x,y,'o',label = 'data')
ax.plot(x,y_pr,'r--',label = 'OLS')
ax.legend(loc = 'best')