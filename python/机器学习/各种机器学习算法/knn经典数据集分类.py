# -*- coding: utf-8 -*-

import pandas as pd
train_data = pd.read_excel(r'E:\python_learning\code\project\knn\鸢尾花案例\鸢尾花训练数据.xlsx')
test_data = pd.read_excel(r'E:\python_learning\code\project\knn\鸢尾花案例\鸢尾花测试数据.xlsx')

#可视化混淆矩阵
def cm_plot(y,yp):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y, yp)
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(y,x),horizontalalignment='center',
                         verticalalignment='center')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
    return plt
"""
变量名分离
"""
train_data.columns
train_x = train_data[['萼片长(cm)', '萼片宽(cm)', '花瓣长(cm)', '花瓣宽(cm)']]
trian_y = train_data.类型_num
"""
Z-score标准化
"""
from sklearn.preprocessing import scale
data = pd.DataFrame()
data['萼片长标准化'] = scale(train_x['萼片长(cm)'])
data['萼片宽标准化'] = scale(train_x['萼片宽(cm)'])
data['花瓣长标准化'] = scale(train_x['花瓣长(cm)'])
data['花瓣宽标准化'] = scale(train_x['花瓣宽(cm)'])

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(data,trian_y)
knn.score(data,trian_y)
pre = knn.predict(data)
cm_plot(trian_y,pre).show()

"""
测试集
"""
test_x =test_data[['萼片长(cm)', '萼片宽(cm)', '花瓣长(cm)', '花瓣宽(cm)']]
test_y = test_data.类型_num
test_data1 = pd.DataFrame()
test_data1['萼片长标准化'] = scale(test_x['萼片长(cm)'])
test_data1['萼片宽标准化'] = scale(test_x['萼片宽(cm)'])
test_data1['花瓣长标准化'] = scale(test_x['花瓣长(cm)'])
test_data1['花瓣宽标准化'] = scale(test_x['花瓣宽(cm)'])
pre1 = knn.predict(test_data1)
cm_plot(test_y,pre1).show()
