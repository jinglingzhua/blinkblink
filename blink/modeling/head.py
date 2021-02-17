from mmdet.models import HEADS
import torch
from torch import nn
from mmcv.cnn import normal_init, constant_init
from task.common.base import StableSoftmax
from task.common.modeling import *

@HEADS.register_module()
class BlinkHead(nn.Module):
    def __init__(self, in_channels, train_cfg, test_cfg, actModule='relu',
                 dense_layers=[128], nclass=2):
        super(BlinkHead, self).__init__()
        if actModule.lower() == 'relu':
            actModule = nn.ReLU
        elif actModule.lower() == 'mish':
            actModule = Mish
        else:
            raise Exception('actModule {} is unknown'.format(actModule))
        layers = [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        for dl in dense_layers:
            layers.append(nn.Linear(in_channels, dl, bias=False))
            layers.append(nn.BatchNorm1d(dl))
            layers.append(actModule())
            in_channels = dl
        layers.append(nn.Linear(in_channels, nclass))
        self.layers = nn.Sequential(*layers)        
        self.softmax = StableSoftmax(dim=1)
                
    def init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
            if isinstance(m, nn.BatchNorm1d):
                constant_init(m, 1)
                
    def forward_train(self, x, img_metas, *args):
        x = self.layers(x[-1])
        y = torch.cat([x['gt_labels'].data for x in img_metas])
        y = y.to(x.device)
        loss = nn.functional.cross_entropy(x, y)
        return {'loss': loss}        
    
    def simple_test_rpn(self, x, img_metas):
        x = self.layers(x[-1])
        return self.softmax(x)