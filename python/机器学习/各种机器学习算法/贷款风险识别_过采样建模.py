import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
import time
data = pd.read_csv(r'E:\python_learning\code\project\逻辑回归案例\信用卡风险用户识别\data.csv',encoding = 'utf8',engine = 'python')
data.head()


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

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])
data.head()

#删除无用列
data = data.drop(['Time'],axis=1)

#对原始数据集进行切分
X_whole = data.drop('Class', axis=1)
y_whole = data.Class

from sklearn.model_selection import train_test_split

X_whole = data.drop('Class', axis=1)
y_whole = data.Class
x_train_w, x_test_w, y_train_w, y_test_w = \
    train_test_split(X_whole, y_whole, test_size = 0.2, random_state = 0)
    
from imblearn.over_sampling import SMOTE

oversampler=SMOTE(random_state=0)
os_x_train, os_y_train = oversampler.fit_sample(x_train_w, y_train_w)

len(os_y_train[os_y_train==1])

os_x_train = pd.DataFrame(os_x_train)
os_y_train = pd.Series(os_y_train)

"""
执行交叉验证操作
scoring:可选“accuracy”（精度）、recall（召回率）、roc_auc（roc值）
        neg_mean_squared_error（均方误差）、
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#交叉验证选择较优惩罚因子
scores = []
c_param_range = [0.01,0.1,1,10,100]
z = 1
for i in c_param_range:
    start_time = time.time()
    lr = LogisticRegression(C = i, penalty = 'l2', solver='lbfgs')    
    score = cross_val_score(lr, os_x_train, os_y_train, cv=10, scoring='roc_auc')
    score_mean = sum(score)/len(score)
    scores.append(score_mean)
    end_time = time.time()
    print("第{}次...".format(z))
    print("time spend:{:.2f}".format(end_time - start_time))
    print("roc_auc值为:{}".format(score_mean))
    z +=1

best_c = c_param_range[np.argmax(scores)]
print()
print("最优惩罚因子为: {}".format(best_c))

"""
建立最优模型
"""
lr = LogisticRegression(C = best_c, penalty = 'l2', solver='lbfgs')
lr.fit(os_x_train, os_y_train)

"""
训练集预测
"""
from sklearn import metrics
#训练集预测概率【大数据集】
train_predicted_pr = lr.predict_proba(os_x_train)
train_predicted = lr.predict(os_x_train)
print(metrics.classification_report(os_y_train, train_predicted))
cm_plot(os_y_train, train_predicted).show() 

#训练集预测概率【小数据集】
train_predicted = lr.predict(x_train_w)
print(metrics.classification_report(y_train_w, train_predicted))
cm_plot(y_train_w, train_predicted).show() 

"""
测试集预测
"""
#预测结果
test_predicted = lr.predict(x_test_w)
#绘制混淆矩阵
print(metrics.classification_report(y_test_w, test_predicted))
cm_plot(y_test_w, test_predicted).show() 