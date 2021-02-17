import cv2
import numpy as np

class FaceObj:
    def __init__(self, bbox, landmark):
        self.bbox, self.landmark = bbox, landmark
        
    def xy_shift(self, left, top):
        self.bbox += (left, top, left, top)
        self.landmark += (left, top)
        
    def draw_on(self, bgr, color=(0,255,0), thickness=2):
        canvas = bgr.copy()
        l, t, r, b = self.bbox
        cv2.rectangle(canvas, (l,t), (r,b), color, thickness=thickness)
        for x, y in self.landmark:
            cv2.circle(canvas, (x,y), 2, color, thickness=thickness)
        return canvas

    @property
    def area(self):
        l, t, r, b = self.bbox
        return (r-l)*(b-t)
    
class FaceObjArray:
    def __init__(self):
        self.faceobjarray = []
        
    def add(self, faceobj):
        self.faceobjarray.append(faceobj)
        
    def empty(self):
        return len(self.faceobjarray) == 0
    
    def draw_on(self, bgr, color=(0,255,0), thickness=2):
        for faceobj in self.faceobjarray:
            bgr = faceobj.draw_on(bgr, color=color, thickness=thickness)
        return bgr
       
    @property
    def biggest(self):
        idx = np.argmax([x.area for x in self.faceobjarray])
        return self.faceobjarray[idx]
    
    def __iter__(self):
        self._iter = 0
        return self
    
    def __next__(self):
        if self._iter >= len(self.faceobjarray):
            raise StopIteration
        _iter = self._iter
        self._iter += 1
        return self.faceobjarray[_iter]
