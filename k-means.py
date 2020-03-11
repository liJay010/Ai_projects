import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics 

"""
可视化网站
https://www.naftaliharris.com/blog/visualizing-k-means-clustering/
"""
#读取文件
beer = pd.read_table("C:/Users/Administrator/Desktop/数据/K-means/data.txt",
                   sep=' ', encoding='utf8', engine='python')
#传入变量（列名）
X = beer[["calories","sodium","alcohol","cost"]]

#寻找合适的K值
"""
根据分成不同的簇，自动计算轮廓系数得分

"""
scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(X).labels_
    score = metrics.silhouette_score(X, labels)
    scores.append(score) 
print(scores)     
#绘制得分结果
import matplotlib.pyplot as plt

plt.plot(list(range(2,20)), scores)
plt.xlabel("Number of Clusters Initialized")
plt.ylabel("Sihouette Score")

#聚类
km = KMeans(n_clusters=2).fit(X)  #K值为3【分为3类】
beer['cluster'] = km.labels_
beer.sort_values('cluster')     #对聚类结果排序【排序时不能修改beer数据框，否则会与X中的数据对不上】

#对聚类结果进行评分
"""
采用轮廓系数评分
X:数据集   scaled_cluster：聚类结果
score：非标准化聚类结果的轮廓系数
"""
score = metrics.silhouette_score(X,beer.cluster)
print(score)  










