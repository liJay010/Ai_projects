import pandas as pd

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
数据预处理
"""
data = pd.read_csv("iris.csv",encoding='utf8', engine='python',header=None)
data = data.drop(0, axis=1)
data.head()

#对原始数据集进行切分
X_whole = data.drop(5, axis=1)
y_whole = data[5]

"""
切分数据集
"""
from sklearn.model_selection import train_test_split

x_train_w, x_test_w, y_train_w, y_test_w = \
    train_test_split(X_whole, y_whole, test_size = 0.2, random_state = 0)

#导入朴素贝叶斯分类器
from sklearn.naive_bayes import MultinomialNB
#实例化贝叶斯分类器
classifier = MultinomialNB(alpha=1)

#传入训练集数据
classifier.fit(x_train_w, y_train_w)

"""
训练集预测
"""
#绘制训练集混淆矩阵
train_pred = classifier.predict(x_train_w)
cm_plot(y_train_w, train_pred)

"""
测试集预测
"""
test_pred = classifier.predict(x_test_w)
cm_plot(y_test_w, test_pred)





























