from mmdet.models import HEADS
import torch
from torch import nn
from mmcv.cnn import normal_init, constant_init
from task.common.base import StableSoftmax
from task.common.modeling import *
import numpy as np

@HEADS.register_module()
class FaceTrackerHead(nn.Module):
    def __init__(self, in_channels, train_cfg, test_cfg, actModule='relu',
                 dense_layers=[128], nclass=2, npoints=4):
        super(FaceTrackerHead, self).__init__()
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
        self.layers = nn.Sequential(*layers)        
        self.classify = nn.Linear(in_channels, nclass)
        self.softmax = StableSoftmax(dim=1)
        self.regress = nn.Linear(in_channels, npoints*2)
                
    def init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
            if isinstance(m, nn.BatchNorm1d):
                constant_init(m, 1)
                
    def forward_train(self, x, img_metas, *args):
        x = self.layers(x[-1])
        pred_cls = self.classify(x)
        pred_reg = self.regress(x)
        
        pts = np.stack([x['gt_pts'] for x in img_metas])
        pts = torch.from_numpy(pts).to(x.device)
        cls = torch.cat([x['gt_labels'].data for x in img_metas])
        cls = cls.to(x.device)
        
        loss_cls = nn.functional.cross_entropy(pred_cls, cls)
        loss_pts = nn.functional.smooth_l1_loss(pred_reg, pts, reduction='none')
        loss_pts = torch.mean(loss_pts, dim=1)
        loss_pts = torch.sum(loss_pts*cls) / torch.clamp(torch.sum(cls), 1)
        return {'loss_cls': loss_cls, 'loss_pts': loss_pts}
    
    def simple_test_rpn(self, x, img_metas):
        x = self.layers(x[-1])
        return self.softmax(self.classify(x)), self.regress(x)