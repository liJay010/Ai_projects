"""
加载网络无需参数
"""
# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_name('efficientnet-b0')

"""
加载网络与参数
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
"""

#修改最后1层
from efficientnet_pytorch import EfficientNet
from torch import nn
model = EfficientNet.from_pretrained('efficientnet-b5')
feature = model._fc.in_features
model._fc = nn.Linear(in_features=feature,out_features=40,bias=True)

print(model)

