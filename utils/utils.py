import os
from os.path import join as opj
from os.path import split as ops
import cv2

def cvshow(title, I, delay=0):
    cv2.imshow(title, I)
    return cv2.waitKey(delay)

def safe_crop(I, left, top, right, bottom, border=cv2.BORDER_REPLICATE, border_value=0):
    ''' safely crop to I[top:bottom, left:right], auto add border '''
    h, w = I.shape[:2]
    if left < 0 or top < 0 or right > w or bottom > h:
        b_left   = max(0, -left)
        b_top    = max(0, -top)
        b_right  = max(0, right-w)
        b_bottom = max(0, bottom-h)
        I = cv2.copyMakeBorder(I, b_top, b_bottom, b_left, b_right, 
                               borderType=border, value=border_value)
        left   += b_left
        right  += b_left
        top    += b_top
        bottom += b_top
    return I[top:bottom, left:right].copy()