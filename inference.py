import torch
import torch.nn as nn
import numpy as np
from utils.dataset_utils import get_sketch, extract_cbr, create_cbz, sorted_alphanumeric, subfolder_image_search, remove_folder
from utils.utils import resize_pad, generate_mask
from torchvision.transforms import ToTensor
import os
import matplotlib.pyplot as plt
import argparse
from model.models import Colorizer, Generator
from model.extractor import get_seresnext_extractor
from utils.xdog import XDoGSketcher
from utils.utils import open_json

def colorize_without_hint(inp, colorizer, device = 'cpu', auto_hint = False, auto_hint_sigma = 0.003):
    i_hint = torch.zeros(1, 4, inp.shape[2], inp.shape[3]).float().to(device)
    
    with torch.no_grad():
        fake_color, _ = colorizer(torch.cat([inp, i_hint], 1))
    
    if auto_hint:
        mask = generate_mask(fake_color.shape[2], fake_color.shape[3], full = False, prob = 1, sigma = auto_hint_sigma).unsqueeze(0)
        mask = mask.to(device)
        i_hint = torch.cat([fake_color * mask, mask], 1)
        
        with torch.no_grad():
            fake_color, _ = colorizer(torch.cat([inp, i_hint], 1))
        
    return fake_color


def process_image(image, sketcher, colorizer, auto_hint, auto_hint_sigma = 0.003, dfm = True, device = 'cpu', to_tensor = ToTensor()):
    image, pad = resize_pad(image)
    bw, dfm = get_sketch(image, sketcher, dfm)
    
    bw = to_tensor(bw).unsqueeze(0).to(device)
    dfm = to_tensor(dfm).unsqueeze(0).to(device)
    
    output = colorize_without_hint(torch.cat([bw, dfm], 1), colorizer, device = device, auto_hint = auto_hint)
    result = output[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    
    if pad[0] != 0:
        result = result[:-pad[0]]
    if pad[1] != 0:
        result = result[:, :-pad[1]]
        
    return result

def colorize_single_image(file_path, save_path, sketcher, colorizer, auto_hint, auto_hint_sigma = 0.003, dfm = True, device = 'cpu'):
    try:
        image = plt.imread(file_path)

        colorization = process_image(image, sketcher, colorizer, auto_hint, auto_hint_sigma, dfm, device)

        plt.imsave(save_path, colorization)
    except:
        print('Failed to colorize {}'.format(image_name))

def colorize_images(source_path, target_path, sketcher, colorizer, auto_hint, auto_hint_sigma = 0.003, dfm = True, device = 'cpu'):
    images = os.listdir(source_path)
    
    for image_name in images:
        try:
            image = plt.imread(os.path.join(source_path, image_name))
            
            colorization = process_image(image, sketcher, colorizer, auto_hint, auto_hint_sigma, dfm, device)
            
            plt.imsave(os.path.join(target_path, image_name), colorization)
        except:
            print('Failed to colorize {}'.format(image_name))
            
def colorize_cbr(file_path, sketcher, colorizer, auto_hint, auto_hint_sigma = 0.003, dfm = True, device = 'cpu'):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    temp_path = 'temp_colorization'
    
    if not  os.path.exists(temp_path):
        os.makedirs(temp_path)
    extract_cbr(file_path, temp_path)
    
    images = subfolder_image_search(temp_path)
    for image_path in images:
        try:
            image = plt.imread(image_path)
            
            colorization = process_image(image, sketcher, colorizer, auto_hint, auto_hint_sigma, dfm, device)
            
            plt.imsave(image_path, colorization)
        except:
            print('Failed to colorize {}'.format(image_name))
    
    result_name = os.path.join(os.path.dirname(file_path), file_name + '_colorized.cbz')
    
    create_cbz(result_name, images)
    
    remove_folder(temp_path)
    

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", required=True)
parser.add_argument("-gen", "--generator", default = 'model/biggan.pth')
parser.add_argument("-ext", "--extractor", default = 'model/extractor.pth')
parser.add_argument("-s", "--sigma", type = float, default = 0.003)
parser.add_argument('-g', '--gpu', dest = 'gpu', action = 'store_true')
parser.add_argument('-ah', '--auto', dest = 'autohint', action = 'store_true')
parser.set_defaults(gpu = False)
parser.set_defaults(autohint = False)
args = parser.parse_args()
    
    
if __name__ == "__main__":
    
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
        
    generator = Generator()
    generator.load_state_dict(torch.load(args.generator))
    
    extractor = get_seresnext_extractor()
    extractor.load_state_dict(torch.load(args.extractor))
    
    colorizer = Colorizer(generator, extractor)
    colorizer = colorizer.eval().to(device)
    
    sketcher = XDoGSketcher()
    xdog_config = open_json('utils/xdog_config.json')
    for key in xdog_config.keys():
        if key in sketcher.params:
            sketcher.params[key] = xdog_config[key]
    
    if os.path.isdir(args.path):
        colorization_path = os.path.join(args.path, 'colorization')
        if not os.path.exists(colorization_path):
            os.makedirs(colorization_path)
            
        colorize_images(args.path, colorization_path, sketcher, colorizer, args.autohint, args.sigma, device = device)
    elif os.path.isfile(args.path):
        split = os.path.splitext(args.path)
        if split[1].lower() in ('.cbr', '.cbz', '.rar', '.zip'):
            colorize_cbr(args.path, sketcher, colorizer, args.autohint, args.sigma, device = device)
        elif split[1].lower() in ('.jpg', '.png'):
            new_image_path = split[0] + '_colorized' + split[1]
            
            colorize_single_image(args.path, new_image_path, sketcher, colorizer, args.autohint, args.sigma, device = device)
        else:
            print('Wrong format')
    else:
        print('Wrong path')
    