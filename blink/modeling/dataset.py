import mmcv
from torch.utils.data import Dataset
from MyUtils.my_utils import *
import cv2
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose
import json


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
    