# -*- coding: utf-8 -*-
import pandas as pd
data = pd.read_excel(r'E:\python_learning\code\project\决策树算法\决策树\电信客户流失数据2.xlsx')
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
#变量结果分离
x = data.ix[:,:-1]
y = data.ix[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = \
        train_test_split(x,y,test_size=0.2,random_state = 42)

from sklearn import tree
dtr = tree.DecisionTreeClassifier(max_depth=10,random_state=42,min_samples_leaf=5)
dtr.fit(x_train,y_train)
train_pre = dtr.predict(x_train)

from sklearn import metrics
print(metrics.classification_report(y_train,train_pre))
cm_plot(y_train,train_pre)

 
#测试集
test_pre = dtr.predict(x_test)
print(metrics.classification_report(y_test,test_pre))
cm_plot(y_test,test_pre)
dtr.score(x_train,y_train)



dot_data = tree.export_graphviz(dtr,feature_names=x.columns)
import pydotplus
grap = pydotplus.graph_from_dot_data(dot_data)
grap.get_nodes()[7].set_fillcolor('#fff200')
from IPython.display import Image
Image(grap.create_png())
grap.write_png('dtr2.png')