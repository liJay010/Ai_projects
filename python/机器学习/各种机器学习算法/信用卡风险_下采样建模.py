# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
data = pd.read_csv(r'E:\python_learning\code\project\逻辑回归案例\信用卡风险用户识别\data.csv',encoding = 'utf8',engine = 'python')
data.head()

"""
绘制图形，查看正负样本的个数
"""
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
mpl.rcParams['axes.unicode_minus'] = False 
            
labels_count = pd.value_counts(data['Class'])            
plt.title("正负例样本数")
plt.xlabel("类别")
plt.ylabel("频数")
labels_count.plot(kind = 'bar')

"""
Z标准化
"""
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])
data.head()
#删除无用列
data = data.drop(['Time'],axis = 1)

posi_eg = data[data['Class']==0]
nega_eg = data[data['Class']==1]
np.random.seed(10)
posi_eg = posi_eg.sample(n = len(nega_eg))

#数据拼接
data_c= pd.concat([posi_eg,nega_eg])

"""
训练集使用下采样数据集，测试集使用原始数据集进行预测
"""
from sklearn.model_selection import train_test_split
X  = data_c.drop('Class',axis=1)
Y = data_c.Class

x_train,x_test,y_train,y_test = \
                train_test_split(X,Y,test_size = 0.3,random_state = 0)

#对原始数据集进行切分
X_h  = data.drop('Class',axis=1)
Y_h = data.Class
x_train_w,x_test_w,y_train_w,y_test_w = \
                train_test_split(X_h,Y_h,test_size = 0.3,random_state = 0)
                
"""
交叉验证
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
scores = []
c_pamas = [0.01,0.1,1,10,100]
for i in c_pamas:
    lr = LogisticRegression(C=i,penalty='l2',solver='liblinear')
    score = cross_val_score(lr,x_train,y_train,cv=10,scoring='recall')
    score_mean = sum(score)/len(score)
    scores.append(score_mean)
    print(score_mean)
best_c = c_pamas[np.argmax(scores)]
print()
print('。。。。。。。。。。惩罚因子最优  {}'.format(best_c))

#建立最优模型
lr  =LogisticRegression(C=best_c,penalty='l2')
lr.fit(x_train,y_train)

train_p = lr.predict(x_train)

from sklearn import metrics
print(metrics.classification_report(y_train,train_p))

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
cm_plot(y_train,train_p).show()

#小测试集合测试
text_pr = lr.predict(x_test)
print(metrics.classification_report(y_test,text_pr))
cm_plot(y_test,text_pr).show()


#大测试集
test_pr = lr.predict(x_test_w)
print(metrics.classification_report(y_test_w,test_pr))
cm_plot(y_test_w,test_pr).show()