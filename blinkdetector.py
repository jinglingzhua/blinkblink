from facedetector.facedetector import FaceDetector
from facetracker.forward import Forward as FaceTracker
from blink.forward import Forward as Blink
import cv2
from utils.utils import cvshow

class BlinkDetector:
    NO_FACE = 1
    EYE_OPEN = 2
    EYE_CLOSE = 3
    def __init__(self, eyeopen_thresh=0.5, facedet_every_nframes=30):
        self.fd = FaceDetector()
        self.ft = FaceTracker()
        self.bk = Blink()
        self.facedet_every_nframes = max(1, facedet_every_nframes)
        self.iframe = self.facedet_every_nframes - 1
        self.lm = None
        self.eyeopen_thresh = eyeopen_thresh
        
    def _det_full(self, bgr):
        faceobjs = self.fd(bgr)
        if not faceobjs.empty():
            self.lm = faceobjs.biggest.landmark
    
    def _track(self, bgr, thresh=0.5):
        score, lm = self.ft(bgr, self.lm)
        self.lm = lm if score >= thresh else None
        
    def show(self, bgr, status, title='FaceTracker', show=-1):
        if show >= 0:
            cv2.putText(bgr, str(status), (10,100), cv2.FONT_HERSHEY_PLAIN,
                        2, (0,255,0), thickness=2)
            #show = 0 if status == self.EYE_CLOSE else show
            k = cvshow(title, bgr, show)
            if k == ord('q'):
                exit()
        
    def get_status(self, bgr, show=-1):
        self.iframe += 1
        if self.iframe >= self.facedet_every_nframes and self.lm is None:
            self.iframe = 0
            self._det_full(bgr)            
        elif self.lm:
            self._track(bgr)  
            
        if self.lm:
            eyeopen_score = self.bk.eyeopen_score(
                bgr, self.lm.left_eye, self.lm.right_eye)
            status = self.EYE_OPEN if eyeopen_score >= self.eyeopen_thresh \
                else self.EYE_CLOSE
        else:
            status = self.NO_FACE
            
        self.show(bgr, status, show=show)
        return status

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_id', default=0, type=int)
    args = parser.parse_args()
    
    bkdet = BlinkDetector()
    cap = cv2.VideoCapture(args.camera_id)
    while True:
        _, bgr = cap.read()
        bkdet.get_status(bgr, show=1)