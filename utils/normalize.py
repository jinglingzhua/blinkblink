import cv2
import numpy as np
import torch

def mmdet_normalize(img, mean, std, to_rgb=True):
    """

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    mean = np.array(mean, 'f4')
    stdinv = 1 / np.array(std, 'f4')
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img - mean) * stdinv
    return np.float32(img)

class StableSoftmax(torch.nn.Module):
    def __init__(self, dim=None):
        super(StableSoftmax, self).__init__()
        self.dim = dim
        
    def forward(self, input):
        x = input - torch.max(input, self.dim, keepdim=True)[0]
        x = torch.exp(x)
        return x / torch.sum(x, self.dim, keepdim=True)        
