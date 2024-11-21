from thop import profile
import torch.nn as nn
import torch
# Note: Please import the model first when testing.
# Eg. from xxx import xxx


# 实例化模型
model = ....
model = model.to('cuda:0')
model.train()

# 计算模型的FLOPs
flops, params = profile(model, inputs=(torch.randn(1,1, 32, 128).to('cuda:0'),))  # 提供一个符合模型输入形状的随机张量
print('模型的Flops:{} '.format(flops))
print('模型的参数量:{}'.format(params))
