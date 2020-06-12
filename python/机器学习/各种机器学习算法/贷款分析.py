# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
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
data = pd.read_excel(r'E:\python_learning\code\project\逻辑回归案例\贷款风险用户识别案例\贷款风险.xls')
#打印前五行
data.head()

#数据分离
x = data.drop('还款拖欠情况',axis = 1)
y = data.还款拖欠情况

from sklearn.model_selection import train_test_split 
#数据集切分  random_state(随机状态-固定)
x_train,x_test,y_train,y_test = \
    train_test_split(x,y,test_size = 0.2 , random_state = 0)
    
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
scores = []
c_param_range = [0.01,0.1,1,10,100]
z=1

for i in c_param_range:
    strat_time = time.time()
    lr = LogisticRegression(C = i,penalty = 'l2',solver = 'lbfgs')
    score = cross_val_score(lr,x_train,y_train,cv=10,scoring='recall')
    score_mean = sum(score)/len(score)
    scores.append(score_mean)
    end_time = time.time()
    print('第{}次'.format(z))
    print("time spend:{:.2f}".format(end_time - start_time))
    print("recall值为:{}".format(score_mean))
    z +=1
best_c = c_param_range[np.argmax(scores)]

lr = LogisticRegression(C=best_c,penalty = 'l2',solver = 'lbfgs')
lr.fit(x_train,y_train)
y_predict = lr.predict(x_train)
from sklearn import metrics
print(metrics.classification_report(y_train,y_predict))
cm_plot(y_predict,y_train).show()

test_pedict = lr.predict(x_test)
print(metrics(y_test,test_pedict))