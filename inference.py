import torch
import torch.nn as nn
import numpy as np
from utils.dataset_utils import get_sketch
from utils.utils import resize_pad, generate_mask, extract_cbr, create_cbz, sorted_alphanumeric, subfolder_image_search, remove_folder
from torchvision.transforms import ToTensor
import os
import matplotlib.pyplot as plt
import argparse
from model.models import Colorizer, Generator
from model.extractor import get_seresnext_extractor
from utils.xdog import XDoGSketcher
from utils.utils import open_json
import sys
from denoising.denoiser import FFDNetDenoiser

def colorize_without_hint(inp, color_args):
    i_hint = torch.zeros(1, 4, inp.shape[2], inp.shape[3]).float().to(color_args['device'])
    
    with torch.no_grad():
        fake_color, _ = color_args['colorizer'](torch.cat([inp, i_hint], 1))
    
    if color_args['auto_hint']:
        mask = generate_mask(fake_color.shape[2], fake_color.shape[3], full = False, prob = 1, sigma = color_args['auto_hint_sigma']).unsqueeze(0)
        mask = mask.to(color_args['device'])
        
        
        if color_args['ignore_gray']:
            diff1 = torch.abs(fake_color[:, 0] - fake_color[:, 1])
            diff2 = torch.abs(fake_color[:, 0] - fake_color[:, 2])
            diff3 = torch.abs(fake_color[:, 1] - fake_color[:, 2])
            mask = ((mask + ((diff1 + diff2 + diff3) > 60 / 255).float().unsqueeze(1)) == 2).float()
        
        
        i_hint = torch.cat([fake_color * mask, mask], 1)
        
        with torch.no_grad():
            fake_color, _ = color_args['colorizer'](torch.cat([inp, i_hint], 1))
        
    return fake_color


def process_image(image, color_args, to_tensor = ToTensor()):
    image, pad = resize_pad(image)
    
    if color_args['denoiser'] is not None:
        image = color_args['denoiser'].get_denoised_image(image, color_args['denoiser_sigma'])
    
    bw, dfm = get_sketch(image, color_args['sketcher'], color_args['dfm'])
    
    bw = to_tensor(bw).unsqueeze(0).to(color_args['device'])
    dfm = to_tensor(dfm).unsqueeze(0).to(color_args['device'])
    
    output = colorize_without_hint(torch.cat([bw, dfm], 1), color_args)
    result = output[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    
    if pad[0] != 0:
        result = result[:-pad[0]]
    if pad[1] != 0:
        result = result[:, :-pad[1]]
        
    return result

def colorize_with_hint(inp, color_args):
    with torch.no_grad():
        fake_color, _ = color_args['colorizer'](inp)
        
    return fake_color
    
def process_image_with_hint(bw, dfm, hint, color_args, to_tensor = ToTensor()):
    bw = to_tensor(bw).unsqueeze(0).to(color_args['device'])
    dfm = to_tensor(dfm).unsqueeze(0).to(color_args['device'])
    
    i_hint = (torch.FloatTensor(hint[..., :3]).permute(2, 0, 1) - 0.5) / 0.5
    mask = torch.FloatTensor(hint[..., 3:]).permute(2, 0, 1)
    i_hint = torch.cat([i_hint * mask, mask], 0).unsqueeze(0).to(color_args['device'])
    
    output = colorize_with_hint(torch.cat([bw, dfm, i_hint], 1), color_args)
    result = output[0].cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    
    return result
    
def colorize_single_image(file_path, save_path, color_args):
    try:
        image = plt.imread(file_path)

        colorization = process_image(image, color_args)

        plt.imsave(save_path, colorization)
        
        return True
    except KeyboardInterrupt:
        sys.exit(0)
    except:
        print('Failed to colorize {}'.format(file_path))
        return False

def colorize_images(source_path, target_path, color_args):
    images = os.listdir(source_path)
    
    for image_name in images:
        file_path = os.path.join(source_path, image_name)
        
        name, ext = os.path.splitext(image_name)
        if (ext != '.png'):
            image_name = name + '.png'
        
        save_path = os.path.join(target_path, image_name)
        colorize_single_image(file_path, save_path, color_args)
            
def colorize_cbr(file_path, color_args):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    temp_path = 'temp_colorization'
    
    if not  os.path.exists(temp_path):
        os.makedirs(temp_path)
    extract_cbr(file_path, temp_path)
    
    images = subfolder_image_search(temp_path)
    
    result_images = []
    for image_path in images:
        save_path = image_path
        
        path, ext = os.path.splitext(save_path)
        if (ext != '.png'):
            save_path = path + '.png'
        
        res_flag = colorize_single_image(image_path, save_path, color_args)
        
        result_images.append(save_path if res_flag else image_path)
        
    
    result_name = os.path.join(os.path.dirname(file_path), file_name + '_colorized.cbz')
    
    create_cbz(result_name, result_images)
    
    remove_folder(temp_path)
    
    return result_name
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True)
    parser.add_argument("-gen", "--generator", default = 'model/generator.pth')
    parser.add_argument("-ext", "--extractor", default = 'model/extractor.pth')
    parser.add_argument("-s", "--sigma", type = float, default = 0.003)
    parser.add_argument('-g', '--gpu', dest = 'gpu', action = 'store_true')
    parser.add_argument('-ah', '--auto', dest = 'autohint', action = 'store_true')
    parser.add_argument('-ig', '--ignore_grey', dest = 'ignore', action = 'store_true')
    parser.add_argument('-nd', '--no_denoise', dest = 'denoiser', action = 'store_false')
    parser.add_argument("-ds", "--denoiser_sigma", type = int, default = 25)
    parser.set_defaults(gpu = False)
    parser.set_defaults(autohint = False)
    parser.set_defaults(ignore = False)
    parser.set_defaults(denoiser = True)
    args = parser.parse_args()
    
    return args

    
if __name__ == "__main__":
    
    args = parse_args()
    
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
    xdog_config = open_json('configs/xdog_config.json')
    for key in xdog_config.keys():
        if key in sketcher.params:
            sketcher.params[key] = xdog_config[key]

    denoiser = None
    if args.denoiser:
        denoiser = FFDNetDenoiser(device, args.denoiser_sigma)
    
    color_args = {'colorizer':colorizer, 'sketcher':sketcher, 'auto_hint':args.autohint, 'auto_hint_sigma':args.sigma,\
                 'ignore_gray':args.ignore, 'device':device, 'dfm' : True, 'denoiser':denoiser, 'denoiser_sigma' : args.denoiser_sigma}
    
    
    if os.path.isdir(args.path):
        colorization_path = os.path.join(args.path, 'colorization')
        if not os.path.exists(colorization_path):
            os.makedirs(colorization_path)
            
        colorize_images(args.path, colorization_path, color_args)
        
    elif os.path.isfile(args.path):
        
        split = os.path.splitext(args.path)
        
        if split[1].lower() in ('.cbr', '.cbz', '.rar', '.zip'):
            colorize_cbr(args.path, color_args)
        elif split[1].lower() in ('.jpg', '.png', ',jpeg'):
            new_image_path = split[0] + '_colorized' + '.png'
            
            colorize_single_image(args.path, new_image_path, color_args)
        else:
            print('Wrong format')
    else:
        print('Wrong path')
    
