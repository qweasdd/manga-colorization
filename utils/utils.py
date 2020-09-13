import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
import cv2

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight.data)
        
def weights_init_spectr(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight_bar.data)
        
def generate_mask(height, width, mu = 1, sigma = 0.0005, prob = 0.5, full = True, full_prob = 0.01):
    X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)

    if full:
        if (np.random.binomial(1, p = full_prob) == 1):
            return torch.ones(1, maskS, maskS).float() 
        
    if np.random.binomial(1, p = prob) == 1:
        mask = torch.rand(1, height, width).ge(X.rvs(1)[0]).float() 
    else:
        mask = torch.zeros(1, height, width).float() 

    return mask

def resize_pad(img, size = 512):
            
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)
        
    if img.shape[2] == 1:
        img = np.repeat(img, 3, 2)
        
    if img.shape[2] == 4:
        img = img[:, :, :3]

    pad = None        
            
    if (img.shape[0] < img.shape[1]):
        height = img.shape[0]
        ratio = height / size
        width = int(np.ceil(img.shape[1] / ratio))
        img = cv2.resize(img, (width, size), interpolation = cv2.INTER_AREA)
        
        new_width = width
        while (new_width % 32 != 0):
            new_width += 1
            
        pad = (0, new_width - width)
        
        img = np.pad(img, ((0, 0), (0, pad[1]), (0, 0)), 'maximum')
    else:
        width = img.shape[1]
        ratio = width / size
        height = int(np.ceil(img.shape[0] / ratio))
        img = cv2.resize(img, (size, height), interpolation = cv2.INTER_AREA)

        new_height = height
        while (new_height % 32 != 0):
            new_height += 1
            
        pad = (new_height - height, 0)
        
        img = np.pad(img, ((0, pad[0]), (0, 0), (0, 0)), 'maximum')
        
    if (img.dtype == 'float32'):
        np.clip(img, 0, 1, out = img)

    return img, pad

