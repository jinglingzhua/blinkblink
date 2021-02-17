import numpy as np

def point_dist(a, b):
    return np.sqrt(np.sum(np.square(a-b)))
    
def point_center(a, b):
    return (a+b)/2

def face_sz(l_eye, r_eye, mouse):
    return point_dist(mouse, point_center(l_eye, r_eye))    

def face_bbox(l_eye, r_eye, mouse):
    sz = face_sz(l_eye, r_eye, mouse)
    center = point_center(mouse, point_center(l_eye, r_eye))
    left = center[0] - sz
    right = center[0] + sz
    top = center[1] - sz
    bottom = center[1] + sz
    return [int(x+0.5) for x in [left, top, right, bottom]]    

