import numpy as np
import matplotlib.pyplot as plt
import cv2
import snowy
import os


def get_resized_image(img, size):
    if len(img.shape) == 2:
        img = np.repeat(np.expand_dims(img, 2), 3, 2)

    if (img.shape[0] < img.shape[1]):
        height = img.shape[0]
        ratio = height / size
        width = int(np.ceil(img.shape[1] / ratio))
        img = cv2.resize(img, (width, size), interpolation = cv2.INTER_AREA)
    else:
        width = img.shape[1]
        ratio = width / size
        height = int(np.ceil(img.shape[0] / ratio))
        img = cv2.resize(img, (size, height), interpolation = cv2.INTER_AREA)
        
    if (img.dtype == 'float32'):
        np.clip(img, 0, 1, out = img)    
        
    return img


def get_sketch_image(img, sketcher, mult_val):
    
    if mult_val:
        sketch_image = sketcher.get_sketch_with_resize(img, mult = mult_val)
    else:
        sketch_image = sketcher.get_sketch_with_resize(img)
        
    return sketch_image


def get_dfm_image(sketch):
    dfm_image = snowy.unitize(snowy.generate_sdf(np.expand_dims(1 - sketch, 2) != 0)).squeeze()
    return dfm_image

def get_sketch(image, sketcher, dfm, mult = None):
    sketch_image = get_sketch_image(image, sketcher, mult)

    dfm_image = None

    if dfm:
        dfm_image = get_dfm_image(sketch_image)

    sketch_image = (sketch_image * 255).astype('uint8')

    if dfm:
        dfm_image = (dfm_image * 255).astype('uint8')

    return sketch_image, dfm_image

def get_sketches(image, sketcher, mult_list, dfm):
    for mult in mult_list:
        yield get_sketch(image, sketcher, dfm, mult)


def create_resized_dataset(source_path, target_path, side_size):
    images = os.listdir(source_path)
    
    for image_name in images:
        
        new_image_name = image_name[:image_name.rfind('.')] + '.png'
        new_path = os.path.join(target_path, new_image_name)
        
        if not os.path.exists(new_path):
            try:
                image = cv2.imread(os.path.join(source_path, image_name))
                
                if image is None:
                    raise Exception()
                
                image = get_resized_image(image, side_size)
               
                cv2.imwrite(new_path, image)                
            except:
                print('Failed to process {}'.format(image_name))
    

def create_sketches_dataset(source_path, target_path, sketcher, mult_list, dfm = False):
    
    images = os.listdir(source_path)
    for image_name in images:
        try:
            image = cv2.imread(os.path.join(source_path, image_name))

            if image is None:
                raise Exception()
            
            for number, (sketch_image, dfm_image) in enumerate(get_sketches(image, sketcher, mult_list, dfm)):
                new_sketch_name = image_name[:image_name.rfind('.')] + '_' + str(number) + '.png'
                cv2.imwrite(os.path.join(target_path, new_sketch_name), sketch_image)
                
                if dfm:
                    dfm_name = image_name[:image_name.rfind('.')] + '_' + str(number) + '_dfm.png'
                    cv2.imwrite(os.path.join(target_path, dfm_name), dfm_image)
            
        except:
            print('Failed to process {}'.format(image_name))
    
    
def create_dataset(source_path, target_path, sketcher, mult_list, side_size, dfm = False):
    images = os.listdir(source_path)
    
    color_path = os.path.join(target_path, 'color')
    sketch_path = os.path.join(target_path, 'bw')
    
    if not os.path.exists(color_path):
        os.makedirs(color_path)
        
    if not os.path.exists(sketch_path):
        os.makedirs(sketch_path)
        
    for image_name in images:
        new_image_name = image_name[:image_name.rfind('.')] + '.png'
        
        try:
            image = cv2.imread(os.path.join(source_path, image_name))
            
            if image is None:
                raise Exception()
                
            resized_image = get_resized_image(image, side_size)
            cv2.imwrite(os.path.join(color_path, new_image_name), resized_image)
            
            for number, (sketch_image, dfm_image) in enumerate(get_sketches(resized_image, sketcher, mult_list, dfm)):
                new_sketch_name = image_name[:image_name.rfind('.')] + '_' + str(number) + '.png'
                cv2.imwrite(os.path.join(sketch_path, new_sketch_name), sketch_image)
                
                if dfm:
                    dfm_name = image_name[:image_name.rfind('.')] + '_' + str(number) + '_dfm.png'
                    cv2.imwrite(os.path.join(sketch_path, dfm_name), dfm_image)
                    
        except:
            print('Failed to process {}'.format(image_name)) 
          