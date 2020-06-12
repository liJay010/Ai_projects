import pandas as pd
data = pd.read_excel(r"E:\python_learning\code\project\机器学习-线性回归\files\data.xlsx")
import matplotlib.pyplot as plt
#绘制散点图
plt.scatter(data.大小,data.房价)
#相关系数
data.大小.corr(data.房价)
#导入机器学习库
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

x = data[['大小']]
y = data[['房价']]

#训练模型
lr.fit(x,y)

#效果评分 -- R2 拟合优度
lr.score(x,y)

#查看截距
b = round(lr.intercept_[0],2)

#查看斜率
k = round(lr.coef_[0][0],2)
#最后结果
print("y = {}x + {}".format(k,b))
#y = 45.15x + 1096.82

#利用结果进行预测
lr.predict([[48],[119],[188]])
