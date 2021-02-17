import numpy as np

class Base:
    def __init__(self):
        self.left_eye = None
        self.right_eye = None
        self.nose = None
        self.left_mouse = None
        self.right_mouse = None
        self.center_mouse = None
        
class Landmark4(Base):
    def __init__(self, lm4):
        super().__init__()
        self.left_eye, self.right_eye, self.nose, self.center_mouse = lm4
        
    @property
    def pts(self):
        return np.array([self.left_eye, self.right_eye, self.nose, self.center_mouse])
        
class Landmark5(Base):
    def __init__(self, lm5):
        super().__init__()
        self.left_eye, self.right_eye, self.nose, self.left_mouse, self.right_mouse = lm5
        self.center_mouse = (self.left_mouse + self.right_mouse) / 2
        
    def to_lm4(self):
        return Landmark4(np.array([self.left_eye, self.right_eye, self.nose, self.center_mouse]))        
    
    @property
    def pts(self):
        return np.array([self.left_eye, self.right_eye, self.nose, self.left_mouse, self.right_mouse])