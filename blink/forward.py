import torch
from utils.normalize import mmdet_normalize
from utils.utils import *
import numpy as np
curdir = os.path.dirname(os.path.abspath(__file__))

class Forward:
    CROP_SZ = 72
    INP_SZ = 64
    NORM_CFG = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375], to_rgb=True)
    def __init__(self, script_path=opj(curdir, 'model.script'), device='cpu'):
        self.model = torch.jit.load(open(script_path, 'rb'), map_location=device)
        self.model.eval()
        self.device = device
        
    def eyeopen_score(self, bgr, left_eye, right_eye):
        left_crop, right_crop = self._crop(bgr, left_eye, right_eye)
        return (self(left_crop) + self(right_crop)) / 2
        
    @staticmethod
    def _crop(bgr, left_eye, right_eye):
        left_eye = np.asarray(left_eye)
        right_eye = np.asarray(right_eye)
        dist = np.sqrt(np.sum(np.square(left_eye-right_eye)))
        sz = max(1, int(dist*0.3 + 0.5))
        
        def _crop_one(bgr, xy, sz):
            l, t = xy[0] - sz, xy[1] - sz
            r, b = xy[0] + sz, xy[1] + sz
            return safe_crop(bgr, l, t, r, b)
        return _crop_one(bgr, left_eye, sz), _crop_one(bgr, right_eye, sz)

    def __call__(self, bgr):
        img = cv2.resize(bgr, (self.CROP_SZ, self.CROP_SZ))
        start = (self.CROP_SZ-self.INP_SZ)//2
        img = img[start:start+self.INP_SZ, start:start+self.INP_SZ]
        I = mmdet_normalize(img, **self.NORM_CFG)
        I = np.moveaxis(I, 2, 0)
        I = torch.from_numpy(I).to(self.device)
        with torch.no_grad():
            out = self.model(I[None])[0]
        return out.cpu().numpy()[1]

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    args = parser.parse_args()

    predictor = Forward()
    bgr = cv2.imread(args.src)
    print(predictor(bgr))







