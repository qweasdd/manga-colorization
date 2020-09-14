from cv2 import resize, INTER_LANCZOS4, INTER_AREA
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt

class XDoGSketcher:
    
    def __init__(self, gamma = 0.95, phi = 89.25, eps = -0.1, k = 8, sigma = 0.5, mult = 1):
        self.params = {}
        self.params['gamma'] = gamma
        self.params['phi'] = phi
        self.params['eps'] = eps
        self.params['k'] = k
        self.params['sigma'] = sigma
        
        self.params['mult'] = mult
        
    def _xdog(self, im, **transform_params):
        # Source : https://github.com/CemalUnal/XDoG-Filter
        # Reference : XDoG: An eXtended difference-of-Gaussians compendium including advanced image stylization
        # Link : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.365.151&rep=rep1&type=pdf
        
        if im.shape[2] == 3:
            im = rgb2gray(im)

        imf1 = gaussian_filter(im, transform_params['sigma'])
        imf2 = gaussian_filter(im, transform_params['sigma'] * transform_params['k'])
        imdiff = imf1 - transform_params['gamma'] * imf2
        imdiff = (imdiff < transform_params['eps']) * 1.0 \
            + (imdiff >= transform_params['eps']) * (1.0 + np.tanh(transform_params['phi'] * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()

       
        th = threshold_otsu(imdiff)
        imdiff = imdiff >= th

        imdiff = imdiff.astype('float32')

        return imdiff
        
        
    def get_sketch(self, image, **kwargs):
        current_params = self.params.copy()
        
        for key in kwargs.keys():
            if key in current_params.keys():
                current_params[key] = kwargs[key]
                
        result_image = self._xdog(image, **current_params)
        
        return result_image
    
    def get_sketch_with_resize(self, image, **kwargs):
        if 'mult' in kwargs.keys():
            mult = kwargs['mult']
        else:
            mult = self.params['mult']
        
        temp_image = resize(image, (image.shape[1] * mult, image.shape[0] * mult), interpolation = INTER_LANCZOS4)
        temp_image = self.get_sketch(temp_image, **kwargs)
        image = resize(temp_image, (image.shape[1], image.shape[0]), interpolation = INTER_AREA)
        
        return image
        
    