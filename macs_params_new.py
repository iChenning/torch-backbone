import torch
from thop import profile
from backbone_encoder.senet import se_resnext101_32x4d


# x = torch.rand(1, 3, 112, 112)
#
# # se_resnext50_32x4d
# net = se_resnext101_32x4d()
# # print(net)
# macs, params = profile(net, inputs=(x,))
# print('se_resnext101_32x4d: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))

from backbone.iresnet import se_iresnet50
x = torch.rand(1, 3, 112, 112)
net = se_iresnet50()
# print(net)
macs, params = profile(net, inputs=(x,))
print('se_iresnet50: macs'.rjust(20), round(macs / 1e9, 2), 'G; parmas', round(params / 1e6, 2))


import math
from torch.optim.lr_scheduler import LambdaLR


__all__ = ['warm_cos']


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_epoch, t_total, len_trainloader, cycles=.5, last_epoch=-1):
        self.warmup_epoch = warmup_epoch * len_trainloader
        self.t_total = t_total * len_trainloader
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_epoch:
            return float(step) / float(max(1.0, self.warmup_epoch))
        # progress after warmup
        progress = float(step - self.warmup_epoch) / float(max(1, self.t_total - self.warmup_epoch))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def warm_cos(optimizer, warmup_epoch, t_total, len_trainloader, cycles=.5, last_epoch=-1):
    return WarmupCosineSchedule(optimizer, warmup_epoch, t_total, len_trainloader, cycles=cycles, last_epoch=last_epoch)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
scheduler = WarmupCosineSchedule(optimizer, 0, 10, 100)
a = torch.load('scheduler.tar')
scheduler.load_state_dict(a)
for i_epoch in range(10):
    for i_iter in range(100):
        optimizer.step()
        scheduler.step()
        print(i_epoch, i_iter, scheduler.state_dict()['last_epoch'], optimizer.state_dict()['param_groups'][0]['lr'])
    torch.save(scheduler.state_dict(), 'scheduler.tar')
    if i_epoch >= 2:
        break