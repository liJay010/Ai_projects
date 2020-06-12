"""
此数据库包含有关4597条电子邮件的信息.任务是确定给定的电子邮件是否是垃圾
邮件(类别1)，取决于其内容。

大多数属性表明某个特定的单词或字符是否经常出现在电子邮件中。

以下是属性的定义：
-48个连续的实属性，类型为word_freq_“word”=与“word”匹配的电子邮件中
单词的百分比。在这种情况下，“Word”是由非字母数字字符或字符串结尾的任
何字母数字字符组成的字符串。
-6个连续的实属性char_freq_“char”=与“char”匹配的电子邮件中字符的百分比。
-1连续实属性类型：Capital_Run_Length_Average=不间断大写字母序列的平均长度。
-1连续整数属性，类型为Capital_Run_Length=最长不间断大写字母序列的长度。
-1连续整数属性，类型为Capital_Run_Length_Total=电子邮件中大写字母的总数。
"""
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
数据读取与划分
"""
df = pd.read_csv('spambase.csv')

from sklearn.model_selection import train_test_split
#数据划分
y = df.ix[:,-1]   #将标签列置于此
X = df.ix[:,:-1]   #删除标签列即可

xtrain, xtest, ytrain, ytest =\
    train_test_split(X, y, test_size=0.2, random_state=100)

#进行预测分析
"""
n_estimators:决策树的个数
max_features:特征的个数
此处可以根据需要设置每棵决策树的深度以及最小叶子节点的样本数等
"""
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_features=0.8,
    random_state=1,
    
)
rf.fit(xtrain, ytrain)

#预测训练集结果
train_predicted = rf.predict(xtrain)
"""
训练集结果
"""
from sklearn import metrics
#绘制混淆矩阵
print(metrics.classification_report(ytrain, train_predicted))

#可视化混淆矩阵
cm_plot(ytrain, train_predicted).show() 

"""
测试集结果
"""
#预测测试集结果
test_predicted = rf.predict(xtest)

from sklearn import metrics
#绘制混淆矩阵
print(metrics.classification_report(ytest, test_predicted))

#可视化混淆矩阵
cm_plot(ytest, test_predicted).show() 

"""
绘制特征重要程度排名
"""
import matplotlib.pyplot as plt
from pylab import mpl

importances = rf.feature_importances_
im = pd.DataFrame(importances)
clos = df.columns.values.tolist()
clos = clos[0:-1]
im['clos'] = clos
im = im.sort_values(by=[0], ascending=False)[:10]

#设置中文字体
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
#解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False 
index = range(len(im))
plt.yticks(index, im.clos)
plt.barh(index, im[0])