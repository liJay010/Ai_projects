# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import os,shutil
data  = pd.read_csv(r'E:\竞赛\冠状病毒肺炎X光检测\dataset\dataset\测试集\test.csv')
name = np.array(data['image_path'].T)
label = np.array(data['labels'].T)

folderName = r'E:\竞赛\冠状病毒肺炎X光检测\dataset\dataset\测试集\images'

for i in range(len(label)):
    print(name[i][7:],i)
    if label[i] == 0:
        shutil.move(os.path.join(folderName,name[i][7:]), r"E:\竞赛\冠状病毒肺炎X光检测\dataset\dataset\测试集\0"+ '\\'+ name[i][7:])
    if label[i] == 1:
        shutil.move(os.path.join(folderName,name[i][7:]), r"E:\竞赛\冠状病毒肺炎X光检测\dataset\dataset\测试集\1" + '\\'+ name[i][7:])
    if label[i] == 2:
        shutil.move(os.path.join(folderName,name[i][7:]), r"E:\竞赛\冠状病毒肺炎X光检测\dataset\dataset\测试集\2" + '\\'+ name[i][7:])
    if label[i] == 3:
        shutil.move(os.path.join(folderName,name[i][7:]), r"E:\竞赛\冠状病毒肺炎X光检测\dataset\dataset\测试集\3" + '\\'+ name[i][7:])


