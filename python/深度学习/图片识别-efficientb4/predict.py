# -*- coding: utf-8 -*
'''
实现模型的调用
'''

from flyai.dataset import Dataset
from model import Model

data = Dataset()
model = Model(data)

dataset = Dataset()
x_test, y_test = dataset.evaluate_data_no_processor('dev.csv')
# 用于测试 predict_all 函数
preds = model.predict_all(x_test)
labels = [i['label'] for i in y_test]
acc=0
for x in range(len(labels)):
    if labels[x] ==preds[x]:
        acc+=1
print(acc)