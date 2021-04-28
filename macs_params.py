import torch
import torchvision.models as models
from thop import profile


x = torch.rand(1, 3, 224, 224)

# AlexNet
net = models.alexnet()
macs, params = profile(net, inputs=(x,))
print('alexnet: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# vgg11
net = models.vgg11()
macs, params = profile(net, inputs=(x,))
print('vgg11: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# vgg13
net = models.vgg13()
macs, params = profile(net, inputs=(x,))
print('vgg13: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# vgg16
net = models.vgg16()
macs, params = profile(net, inputs=(x,))
print('vgg16: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# vgg19
net = models.vgg19()
macs, params = profile(net, inputs=(x,))
print('vgg19: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# resnet18
net = models.resnet18()
macs, params = profile(net, inputs=(x,))
print('resnet18: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# resnet34
net = models.resnet34()
macs, params = profile(net, inputs=(x,))
print('resnet34: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# resnet50
net = models.resnet50()
macs, params = profile(net, inputs=(x,))
print('resnet50: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# resnet101
net = models.resnet101()
macs, params = profile(net, inputs=(x,))
print('resnet101: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# resnet152
net = models.resnet152()
macs, params = profile(net, inputs=(x,))
print('resnet152: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# squeezenet1_0
net = models.squeezenet1_0()
macs, params = profile(net, inputs=(x,))
print('squeezenet1_0: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# squeezenet1_1
net = models.squeezenet1_1()
macs, params = profile(net, inputs=(x,))
print('squeezenet1_1: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# densenet121
net = models.densenet121()
macs, params = profile(net, inputs=(x,))
print('densenet121: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# densenet169
net = models.densenet169()
macs, params = profile(net, inputs=(x,))
print('densenet169: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# densenet201
net = models.densenet201()
macs, params = profile(net, inputs=(x,))
print('densenet201: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# densenet161
net = models.densenet161()
macs, params = profile(net, inputs=(x,))
print('densenet161: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# inception_v3
net = models.inception_v3()
macs, params = profile(net, inputs=(x,))
print('inception_v3: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# googlenet
net = models.googlenet()
macs, params = profile(net, inputs=(x,))
print('googlenet: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# shufflenet_v2_x1_0
net = models.shufflenet_v2_x1_0()
macs, params = profile(net, inputs=(x,))
print('shufflenet_v2_x1_0: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# shufflenet_v2_x0_5
net = models.shufflenet_v2_x0_5()
macs, params = profile(net, inputs=(x,))
print('shufflenet_v2_x0_5: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# mobilenet_v2
net = models.mobilenet_v2()
macs, params = profile(net, inputs=(x,))
print('mobilenet_v2: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# mobilenet_v3_large
net = models.mobilenet_v3_large()
macs, params = profile(net, inputs=(x,))
print('mobilenet_v3_large: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# mobilenet_v3_small
net = models.mobilenet_v3_small()
macs, params = profile(net, inputs=(x,))
print('mobilenet_v3_small: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# resnext50_32x4d
net = models.resnext50_32x4d()
macs, params = profile(net, inputs=(x,))
print('resnext50_32x4d: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# resnext101_32x8d
net = models.resnext101_32x8d()
macs, params = profile(net, inputs=(x,))
print('resnext101_32x8d: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# wide_resnet50_2
net = models.wide_resnet50_2()
macs, params = profile(net, inputs=(x,))
print('wide_resnet50_2: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# wide_resnet101_2
net = models.wide_resnet101_2()
macs, params = profile(net, inputs=(x,))
print('wide_resnet101_2: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# mnasnet1_0
net = models.mnasnet1_0()
macs, params = profile(net, inputs=(x,))
print('mnasnet1_0: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

# mnasnet0_5
net = models.mnasnet0_5()
macs, params = profile(net, inputs=(x,))
print('mnasnet0_5: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))