import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import albumentations as albu
import argparse

from utils.utils import open_json, weights_init, weights_init_spectr, generate_mask
from model.models import Colorizer, Generator, Content, Discriminator
from model.extractor import get_seresnext_extractor
from dataset.datasets import TrainDataset, FineTuningDataset



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help = "dataset path")
    parser.add_argument('-ft', '--fine_tuning', dest = 'fine_tuning', action = 'store_true')
    parser.add_argument('-g', '--gpu', dest = 'gpu', action = 'store_true')
    parser.set_defaults(fine_tuning = False)
    parser.set_defaults(gpu = False)
    args = parser.parse_args()
    
    return args

def get_transforms():
    return albu.Compose([albu.RandomCrop(512, 512, always_apply = True), albu.HorizontalFlip(p = 0.5)], p = 1.)

def get_dataloaders(data_path, transforms, batch_size, fine_tuning):
    train_dataset = TrainDataset(data_path, transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    
    if fine_tuning:
        finetuning_dataset = FineTuningDataset(data_path, transforms)
        finetuning_dataloader = torch.utils.data.DataLoader(finetuning_dataset, batch_size = batch_size, shuffle = True)
    
    return train_dataloader, finetuning_dataloader

def get_models(device):
    generator = Generator()
    extractor = get_seresnext_extractor()
    colorizer = Colorizer(generator, extractor)
    
    colorizer.extractor_eval()
    colorizer = colorizer.to(device)
    
    discriminator = Discriminator().to(device)
    
    content = Content('model/vgg16-397923af.pth').eval().to(device)
    for param in content.parameters():
        param.requires_grad = False
    
    return colorizer, discriminator, content

def set_weights(colorizer, discriminator):
    colorizer.generator.apply(weights_init)
    colorizer.load_extractor_weights(torch.load('model/extractor.pth'))
    
    discriminator.apply(weights_init_spectr)
    
def get_losses():
    L1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    return L1_loss, bce_loss, mse_loss
    
def get_optimizers(colorizer, discriminator, generator_lr, discriminator_lr):
    optimizerG = optim.Adam(colorizer.generator.parameters(), lr = generator_lr, betas=(0.5, 0.9))
    optimizerD = optim.Adam(discriminator.parameters(), lr = discriminator_lr, betas=(0.5, 0.9))
    
    return optimizerG, optimizerD



if __name__ == '__main__':
    args = parse_args()
    config = open_json('configs/train_config.json')
    
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'
        
    augmentations = get_transforms()
    
    train_dataloader, ft_dataloader = get_dataloaders(args.path, augmentations, config['batch_size'], args.fine_tuning)
    
    colorizer, discriminator, content = get_models(device)
    set_weights(colorizer, discriminator)
    
    l1_loss, bce_loss, mse_loss = get_losses()
    
    gen_optimizer, disc_optimizer = get_optimizers(colorizer, discriminator, config['generator_lr'], config['discriminator_lr'])
    
    