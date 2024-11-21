from thop import profile
import torch
from LGTSNet import LGTSNet


# 实例化模型
model = LGTSNet()
model = model.to('cuda:0')
model.train()

# 计算模型的FLOPs
_, params = profile(model, inputs=(torch.randn(1,1,14,128).to('cuda:0'),))  # 提供一个符合模型输入形状的随机张量
print('模型的参数量:{}'.format(params))
