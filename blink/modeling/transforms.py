import imgaug.augmenters as iaa
from mmdet.datasets.pipelines.transforms import *
import cv2
from MyUtils.my_utils import *


def get_training_augmentation(**kwargs):
    seq = iaa.Sequential([
        iaa.Resize({'height': kwargs['crop_sz'], 'width': kwargs['crop_sz']}),
        iaa.flip.Fliplr(p=0.5),
        iaa.OneOf([
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.MotionBlur(k=(3,5))
        ]),
        iaa.OneOf([iaa.GammaContrast((0.8, 1.0)),
                   iaa.LinearContrast((0.75, 1.5)),
                   iaa.LogContrast((0.8, 1.0))]),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        iaa.Crop(px=(0, 2*(kwargs['crop_sz']-kwargs['inp_sz']))),
        iaa.Resize({'height': kwargs['inp_sz'], 'width': kwargs['inp_sz']})
    ])
    return seq

def get_valid_augmentation(**kwargs):
    seq = iaa.Sequential([
        iaa.Resize({'height': kwargs['crop_sz'], 'width': kwargs['crop_sz']}),
        iaa.CenterCropToFixedSize(height=kwargs['inp_sz'], width=kwargs['inp_sz'])
    ])
    return seq

@PIPELINES.register_module()
class BlinkTransform(object):
    def __init__(self, training=True, **kwargs):
        if training:
            self.augmentation = get_training_augmentation(**kwargs)
        else:
            self.augmentation = get_valid_augmentation(**kwargs)
        self.training = training
    
    def __call__(self, results):
        results['img'] = self.augmentation(images=results['img'][None])[0]
        return results






