"""
Created on Fri Nov 17 16:59:17 2023

@author: LYT
"""

import torch
from torch.utils import data
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
from unet import UNet
from diffusion import GaussianDiffusion, Trainer
from scipy.io import savemat
from scipy.io import loadmat


"""Dataloader"""

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, mode): 
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.mode = mode
                                  
        self.transform = transforms.Compose([  
            # transforms.Resize(image_size),
            # transforms.CenterCrop(image_size),
            # transforms.ToTensor(),
            # transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        
    def __len__(self):        
        dir_path = self.folder+"data/"
        res = len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))])
        return res


    # def __getitem__(self, index):

    #     data = self.folder+"data/"+str(index)+".png"
    #     img_data = Image.open(data)

    #     if self.mode == "demultiple":
    #         label = self.folder+"labels/"+str(index)+".png"
    #         img_label = Image.open(label)
    #         return self.transform(img_data), self.transform(img_label)
    #     elif self.mode == "interpolation":
    #         return self.irregular_mask(self.transform(img_data)), self.transform(img_data)
    #     elif self.mode == "denoising":
    #         rate = 0.1 #0.5
    #         img = self.transform(img_data)
    #         mean = torch.mean(img)
    #         std = torch.std(img)
    #         noise = rate*torch.normal(mean, std, size =(img.shape[0], img.shape[1], img.shape[2]))
    #         img_ = img + noise
    #         return img_, img  
    #     else:
    #         print("ERROR MODE DATASET")
    
    def __getitem__(self, index):
        # print("index")
        data = self.folder + "data/" + str(index) + ".mat"
        # print("data",data)
        
        img_data = loadmat(data)['data']
        img_data = torch.tensor(np.array(img_data))
        img_data = torch.unsqueeze(img_data, 0)
        # img_data = img_data[:,::2,::2]
        # print('img_data',img_data.shape)

        if self.mode == "demultiple":
            
            label = self.folder+"labels/"+str(index)+".mat"
            img_label = loadmat(label)['data']
            img_label = torch.tensor(np.array(img_label))
            img_label = torch.unsqueeze(img_label, 0)
            # img_label = img_label[:,::2,::2]
            
            return img_data, img_label
        else:
            print("ERROR MODE")      
            
            
            


"""Create Model"""
SNR = -2 #5
mode = "demultiple" 
# folder = './dataset/'+mode+'/data_test/'
folder = "data/marmousi2/SNR={}".format(SNR)+"/data_test/"

image_size = (64,128)

model = UNet(
        in_channel=2, #2
        out_channel=1
).cuda()

diffusion = GaussianDiffusion(
    model,
    mode = mode,
    channels = 1,
    image_size = image_size,
    timesteps = 2000, #2000
    loss_type = 'l1', # L1 or L2
).cuda()

ds = Dataset(folder, image_size=image_size, mode=mode)


"""Load diffusion Model"""
name = "5" #5 "test" 
parameters = torch.load("results_demultiple/model-steps=5000-SNR={}.pt".format(name))['model']



del parameters['betas']
del parameters['alphas_cumprod']
del parameters['alphas_cumprod_prev']
del parameters['sqrt_alphas_cumprod']
del parameters['sqrt_one_minus_alphas_cumprod']
del parameters['log_one_minus_alphas_cumprod']
del parameters['sqrt_recip_alphas_cumprod']
del parameters['sqrt_recipm1_alphas_cumprod']
del parameters['posterior_variance']
del parameters['posterior_log_variance_clipped']
del parameters['posterior_mean_coef1']
del parameters['posterior_mean_coef2']


def change_key(self, old, new):
    #copy = self.copy()
    for _ in range(len(self)):
        k, v = self.popitem(False)
        self[new if old == k else k] = v
        
keys = []
for key, value in parameters.items():
    keys.append(key)
    
for i in range(len(keys)):
    change_key(parameters, keys[i], keys[i][11:])
    
model.load_state_dict(parameters)

num = 4
in_samples = np.zeros([num,image_size[0],image_size[1]])
out_samples = np.zeros([num,image_size[0],image_size[1]])

for i, (x_in) in enumerate(ds):
    print('i',i)
    print('(x_in)',len(x_in))
    x_start = x_in[0]
    print('x_start',x_start)
    
    x_start = torch.unsqueeze(x_start, dim=0)
    print('x_start',x_start)
    x_ = x_in[1]
    print('x_',x_)
    x_ = torch.unsqueeze(x_, dim=0)

    if mode == "interpolation":
        out = diffusion.inference(x_in=x_.cuda(), mask=x_start.cuda())
        print('out1',out.shape)
    else:
        out = diffusion.inference(x_in=x_start.cuda())
        print('out',out.shape)

    in_samples[i] = out[0,0].cpu().detach().numpy()
    out_samples[i] = out[1,0].cpu().detach().numpy()
    if mode == "interpolation":
        out_samples[i] = out[2,0].cpu().detach().numpy()
    
    if i == num-1:
        break
    
   
in_samples = in_samples.transpose(0,2,1)
in_samples = np.expand_dims(in_samples, axis=2)

out_samples = out_samples.transpose(0,2,1)
out_samples = np.expand_dims(out_samples, axis=2)


np.save('data/diffusion_result_SNR={}.npy'.format(SNR), {'in_samples': in_samples,'out_samples': out_samples})


"""Plotting"""
fig, axs = plt.subplots(1,2* num, figsize=(20,8))

max_ = x_start.max()
min_ = x_start.min()
cont = 0
for i in range(num):
    axs[cont].imshow(in_samples[i,:,0,:].T, vmin=min_,vmax=max_, cmap="Greys")
    axs[cont].set_title("Input "+str(i))
    cont = cont+1
    axs[cont].imshow(out_samples[i,:,0,:].T, vmin=min_,vmax=max_, cmap="Greys")
    axs[cont].set_title("Output "+str(i))
    cont = cont+1
[axi.set_axis_off() for axi in axs.ravel()]

