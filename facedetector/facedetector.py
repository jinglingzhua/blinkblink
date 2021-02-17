import cv2
import numpy as np
from .nn import *
from .hyp import hyp
from .utils.utils import non_max_suppression
import os
from utils.faceobj import *
from utils.landmark import *
curdir = os.path.dirname(os.path.abspath(__file__))

class Preprocessor:
    def __init__(self, sz_multiple=32):
        self.sz_multiple = sz_multiple
        
    @staticmethod
    def _round_sz(sz, multiple):
        return int(np.round(sz/multiple)*multiple)
    
    def forward_img(self, img):
        ori_h, ori_w = img.shape[:2]
        tar_h, tar_w = [self._round_sz(x, self.sz_multiple) for x in [ori_h, ori_w]]
        self.scale_h, self.scale_w = ori_h / tar_h, ori_w / tar_w
        if ori_h != tar_h or ori_w != ori_w:
            img = cv2.resize(img, (tar_w, tar_h))
        return img
    
    def backward_points(self, bboxes, landmarks):
        bboxes = bboxes * ([self.scale_w, self.scale_h]*2)
        landmarks = landmarks * [self.scale_w, self.scale_h]
        return np.int32(bboxes+0.5), np.int32(landmarks+0.5)
        
class FaceDetector:
    DEFAULT_MODEL_PATH = os.path.join(curdir, 'mbv3_small_75_light_final.pt')
    def __init__(self, model_path='', device='cpu'):
        model_path = self.DEFAULT_MODEL_PATH if model_path == '' else model_path
        backone = mobilenetv3_small( width_mult=0.75)
        self.net = DarknetWithShh(backone, hyp, light_head=True).to(device)
        self.net.load_state_dict(torch.load(model_path, map_location=device)['model'])
        self.net.eval()        
        self.preproc = Preprocessor()
        self.device = device
        self.n_points = 5
        
    @staticmethod
    def show(bgr, facearr, title='FaceDetector', show=-1):
        if show >= 0:
            bgr = facearr.draw_on(bgr)
            cv2.imshow(title, bgr)
            cv2.waitKey(show)
        
    def __call__(self, bgr, score_thresh=0.3, show=-1):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = self.preproc.forward_img(rgb)
        inp = torch.from_numpy(np.float32(np.moveaxis(rgb,2,0))/255).to(self.device)
        with torch.no_grad():
            pred = self.net(inp[None])[0]

        pred = non_max_suppression(pred, score_thresh, 0.35, multi_label=False, classes=0,
                                   agnostic=False, land=True, point_num=self.n_points)[0]
        facearr = FaceObjArray()
        if pred is not None:
            det = pred.cpu().numpy()
            bboxes = det[:, :4]
            landmarks = det[:, 5: 5+self.n_points*2].reshape((-1,self.n_points,2))
            bboxes, landmarks = self.preproc.backward_points(bboxes, landmarks)
            for bbox, lm in zip(bboxes, landmarks):
                facearr.add(FaceObj(bbox, Landmark5(lm)))

        self.show(bgr, facearr, show=show)
        return facearr
        