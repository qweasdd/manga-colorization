import torch
import os
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import generate_mask


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform = None, mults_amount = 1):
        self.data = os.listdir(os.path.join(data_path, 'color'))
        self.data_path = data_path
        self.transform = transform
        self.mults_amount = mults_amount
        
        self.ToTensor = transforms.ToTensor()
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_name = self.data[idx]
        
        color_img = plt.imread(os.path.join(self.data_path, 'color', image_name))
        

        if self.mults_amount > 1:
            mult_number = np.random.choice(range(self.mults_amount))
            
            bw_name = image_name[:image_name.rfind('.')] + '_' + str(mult_number) + '.png'
            dfm_name = image_name[:image_name.rfind('.')] + '_' + str(mult_number) + '_dfm.png'
        else:
            bw_name = self.data[idx]
            dfm_name = 'dfm_' + self.data[idx]
            
            
        bw_img =  np.expand_dims(plt.imread(os.path.join(self.data_path, 'bw', bw_name)), 2)
        dfm_img =  np.expand_dims(plt.imread(os.path.join(self.data_path, 'bw', dfm_name)), 2)
        
        bw_img = np.concatenate([bw_img, dfm_img], axis = 2)
        
        if self.transform:
            result = self.transform(image = color_img, mask = bw_img)
            color_img = result['image']
            bw_img = result['mask']
          
        dfm_img = bw_img[:, :, 1]
        bw_img = bw_img[:, :, 0]
        
        color_img = self.ToTensor(color_img)
        bw_img = self.ToTensor(bw_img)
        
        dfm_img = self.ToTensor(dfm_img)
        
        color_img = (color_img - 0.5) / 0.5
        
        mask = generate_mask(bw_img.shape[1], bw_img.shape[2])
        hint = torch.cat((color_img * mask, mask), 0)
        
        return bw_img, color_img, hint, dfm_img
    
class FineTuningDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform = None):
        self.data = [x for x in os.listdir(os.path.join(data_path, 'real_manga')) if x.find('dfm_') == -1] * 8
        self.color_data = [x for x in os.listdir(os.path.join(data_path, 'color')) if x.find('left') == -1 and x.find('right') == -1] * 6
        self.data_path = data_path
        self.transform = transform
        
        np.random.shuffle(self.color_data)
        
        self.ToTensor = transforms.ToTensor()
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        color_img = plt.imread(os.path.join(self.data_path, 'color', self.color_data[idx]))
        bw_img =  np.expand_dims(plt.imread(os.path.join(self.data_path, 'real_manga', self.data[idx])), 2)
        dfm_img =  np.expand_dims(plt.imread(os.path.join(self.data_path, 'real_manga', 'dfm_' + self.data[idx])), 2)
        
        if self.transform:
            result = self.transform(image = color_img)
            color_img = result['image']
            
            result = self.transform(image = bw_img,  mask = dfm_img)
            bw_img = result['image']
            dfm_img = result['mask']
        
        color_img = self.ToTensor(color_img)
        bw_img = self.ToTensor(bw_img)
        dfm_img = self.ToTensor(dfm_img)
        
        color_img = (color_img - 0.5) / 0.5
        
        return bw_img, dfm_img, color_img
