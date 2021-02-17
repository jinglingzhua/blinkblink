from torch.utils.data import Dataset
from utils.utils import *
import cv2
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
from .geometry import *

@DATASETS.register_module()
class Caltech(Dataset):
    CLASSES = []
    def __init__(self, datadir, pipeline, bg_fg_ratio=3):
        self.bg_fg_ratio = max(0, int(bg_fg_ratio+0.5))
        self.gt = self._load_gt(datadir)
        self.pipeline = Compose(pipeline)
        self.data = self._gen_key_pts_idx(self.gt, bg_fg_ratio)
        self.flag = np.zeros(len(self), 'u1')
        
    def _gen_key_pts_idx(self, gt, bg_fg_ratio):
        data = []
        for k, pts_list in gt.items():
            for i in range(len(pts_list)):
                data.append((k,i))
            data.extend([(k,-1)]*self.bg_fg_ratio)
        return data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        k, pts_idx = self.data[index]
        results = dict()
        Img = cv2.imread(k, 1)
        results['img_info'] = {'file_name': k, 'filename': k,
                               'height': Img.shape[0],
                               'width': Img.shape[1]}
        results['img'] = Img
        results['gt_pts'] = self.gt[k]
        results['pts_idx'] = pts_idx
        return self.pipeline(results)
       
    def _load_gt(self, datadir):
        gt = dict()
        with open(opj(datadir, 'WebFaces_GroundThruth.txt'), 'r') as f:
            for s in f.read().strip().split('\n'):
                s = s.split()
                k = opj(datadir, s[0])
                pts = np.array([float(x) for x in s[1:]]).reshape((-1,2))
                if not self._is_valid_pts(pts):
                    continue
                if k not in gt:
                    gt[k] = []
                gt[k].append(pts)
        return gt
    
    def _is_valid_pts(self, pts):
        MIN_EYE_DIST = 20
        return point_dist(pts[0], pts[1]) >= MIN_EYE_DIST       
    
@DATASETS.register_module()
class BlinkDataset(Dataset):
    CLASSES = None

    def __init__(self, datadir, pipeline, stage='train', repeat=1):
        self.images = get_spec_files(datadir, ext=IMG_EXT, oswalk=True)
        if stage == 'train':
            self.images = self.images * repeat
        self.pipeline = Compose(pipeline)
        self.flag = np.zeros(len(self), 'u1')
        self.stage = stage

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        results = dict()
        img = self.images[index]
        Img = cv2.imread(img, 1)
        results['img_info'] = {'file_name': self.images[index],
                               'filename': self.images[index],
                               'height': Img.shape[0],
                               'width': Img.shape[1]}
        _, fname = os.path.split(img)
        results['img'] = Img
        results['gt_labels'] = int(fname.split('_')[4])
        return self.pipeline(results)
    