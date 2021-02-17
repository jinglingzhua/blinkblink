import imgaug.augmenters as iaa
from mmdet.datasets.pipelines.transforms import *
from .geometry import *
from MyUtils.my_utils import safe_crop
import cv2

def get_fg_augmentation():
    seq = iaa.Sequential([
        iaa.Resize({'height': 'keep', 'width': (0.9, 1.1)}),
        iaa.KeepSizeByResize(iaa.Rotate((-15,15), fit_output=True))
    ])
    return seq

def get_common_augmentation(inp_hw):
    seq = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0.0, 0.5)),
        iaa.OneOf([iaa.GammaContrast((0.8, 1.0)),
                   iaa.LinearContrast((0.75, 1.5)),
                   iaa.LogContrast((0.8, 1.0))]),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02 * 255), per_channel=0.5),
        iaa.Resize({'height': inp_hw[0], 'width': inp_hw[1]}),
    ])
    return seq

@PIPELINES.register_module()
class FaceTrackerTransform(object):
    def __init__(self, inp_hw):
        self.fg_augmentation = get_fg_augmentation()
        self.common_augmentation = get_common_augmentation(inp_hw)
        
    @staticmethod
    def _get_bbox_by_pts(pts):
        return face_bbox(pts[0], pts[1], pts[3])
       
    def _crop_bg(self, img, pts_list):
        for pts in pts_list:
            l, t, r, b = self._get_bbox_by_pts(pts)
            img[max(0,t):b, max(0,l):r] = [np.random.randint(256) 
                                           for _ in range(3)]
        pts = pts_list[np.random.randint(len(pts_list))]
        l, t, r, b = self._get_bbox_by_pts(pts)
        w = int((r-l)*np.random.uniform(0.8,1.2)+0.5)
        h = int((b-t)*np.random.uniform(0.8,1.2)+0.5)
        x = max(0, img.shape[1]-w)
        if x > 0:
            x = np.random.randint(x)
        y = max(0, img.shape[0] - h)
        if y > 0:
            y = np.random.randint(y)
        return safe_crop(img, x, y, x+w, y+h), np.zeros((4,2), 'f4')
    
    @staticmethod
    def _crop_fg_rand(l, t, r, b):
        def _scale_bbx(l, t, r, b):
            ctx, cty = (r+l)/2, (b+t)/2
            sz = (r-l)/2 * np.random.uniform(0.8,1.2)
            return ctx - sz, cty - sz, ctx + sz, cty + sz
        
        def _move_bbx(l, t, r, b):
            shift = int((r - l) * 0.1 + 0.5)
            dx = np.random.randint(-shift, shift)
            dy = np.random.randint(-shift, shift)
            return l+dx, t+dy, r+dx, b+dy
        
        def _to_int(l, t, r, b):
            return [int(x+0.5) for x in [l, t, r, b]]
        
        if np.random.randint(2):
            bbx = _move_bbx(l, t, r, b)
            return _to_int(*_scale_bbx(*bbx))
        else:
            bbx = _scale_bbx(l, t, r, b)
            return _to_int(*_move_bbx(*bbx))
        
    def _crop_fg(self, img, pts):
        img, pts = self.fg_augmentation(image=img, keypoints=pts[None])
        pts = pts[0]
        l, t, r, b = self._get_bbox_by_pts(pts)
        l, t, r, b = self._crop_fg_rand(l, t, r, b)
        return safe_crop(img, l, t, r, b), pts - (l, t)
    
    def _flip_lr(self, img, pts):
        if np.random.randint(2):
            img = img[:,::-1]
            pts[:, 0] = img.shape[1] - pts[:, 0]
            pts[[0,1]] = pts[[1,0]]
        return img, pts
    
    def __call__(self, results):
        pts_list = results['gt_pts']
        pts_idx = results['pts_idx']
        img = results['img']
        if pts_idx == -1:
            img, gt_pts = self._crop_bg(img, pts_list)
        else:
            img, gt_pts = self._crop_fg(img, pts_list[pts_idx])
        img, gt_pts = self._flip_lr(img, gt_pts)
        img, gt_pts = self.common_augmentation(image=img, keypoints=gt_pts[None])
        gt_pts = gt_pts[0]
        results['img'] = img
        results['gt_pts'] = gt_pts.flatten().astype(np.float32)
        results['gt_labels'] = int(pts_idx >= 0)
        
        return results






