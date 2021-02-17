from utils.utils import *
import torch
from utils.normalize import mmdet_normalize
from utils.geometry import *
from utils.landmark import *
curdir = os.path.dirname(os.path.abspath(__file__))

class Forward:
    INP_SZ = 64
    NORM_CFG = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)
    def __init__(self, script_path=opj(curdir, 'model.script'), device='cpu'):
        self.model = torch.jit.load(open(script_path, 'rb'), map_location=device)
        self.model.eval()
        self.device = device
        
    def _scale_resize(self, bgr, pts):
        l, t, r, b = face_bbox(pts[0], pts[1], pts[3])
        C = safe_crop(bgr, l, t, r, b)
        scale = C.shape[0] / self.INP_SZ
        C = cv2.resize(C, (self.INP_SZ, self.INP_SZ))
        self.offset, self.scale = (l, t), scale
        return C
    
    def _scale_resize_back(self, pts):
        return pts * self.scale + self.offset
    
    def __call__(self, bgr, lm):
        if isinstance(lm, Landmark5):
            lm = lm.to_lm4()
        pts = lm.pts
        bgr = self._scale_resize(bgr, pts)
        #cvshow('', bgr)
        
        I = mmdet_normalize(bgr, **self.NORM_CFG)
        I = np.moveaxis(I, 2, 0)
        I = torch.from_numpy(I).to(self.device)
        with torch.no_grad():
            score, pts = [x.cpu().numpy() for x in self.model(I[None])]
        score = score[0,1]
        pts = pts[0].reshape((-1,2))
        pts = self._scale_resize_back(pts)
        
        return score, Landmark4(np.int32(pts+0.5))
    