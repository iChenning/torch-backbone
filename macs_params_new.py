import torch
from thop import profile
from backbone.xception import xception


x = torch.rand(1, 3, 224, 224)

# xception
net = xception(pretrained=None)
macs, params = profile(net, inputs=(x,))
print('xception: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))