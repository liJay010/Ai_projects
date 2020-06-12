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


#导入数据
datas = pd.read_excel("电信客户流失数据.xlsx",encoding='utf8')
#将变量与结果划分开
data = datas.ix[:,:-1]
target = datas.ix[:,-1]

#划分数据集
"""
导入模块对数据进行划分；
"""
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = \
    train_test_split(data, target, test_size = 0.2, 
                     random_state = 42)
#定义决策树  
from sklearn import tree 
   
dtr = tree.DecisionTreeClassifier(criterion='gini', max_depth = 6,
                                  random_state = 42)
dtr.fit(data_train, target_train)

"""
训练集混淆矩阵
"""
#训练集预测值
train_predicted = dtr.predict(data_train)

from sklearn import metrics
#绘制混淆矩阵
print(metrics.classification_report(target_train, train_predicted))
#可视化混淆矩阵
cm_plot(target_train, train_predicted).show() 


"""
测试集混淆矩阵
"""
#测试集预测值
test_predicted = dtr.predict(data_test)

from sklearn import metrics
#绘制混淆矩阵
print(metrics.classification_report(target_test, test_predicted))
#可视化混淆矩阵
cm_plot(target_test, test_predicted).show() 
#对决策树测试集进行评分
dtr.score(data_test, target_test)

#要可视化显示 
"""
修改dtr为自己的变量名；
修改feature_names为自己的数据
最终生成一个.dot文件
"""
dot_data = \
    tree.export_graphviz(
        dtr,
        out_file = None,
        feature_names = data.columns,
        filled = True,
        impurity = False,
        rounded = True
    )
#导入pydotplus库解读.dot文件
"""
只用修改颜色"#FFF2DD"
"""
import pydotplus

graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor("#FFF2DD")
               
from IPython.display import Image
Image(graph.create_png())

#导出决策树的图
graph.write_png("dtr.png")












