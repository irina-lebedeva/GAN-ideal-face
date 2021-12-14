import os
import glob
import argparse
import random
import numpy as np
from shutil import copyfile
import shutil
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import cv2
from model import Generator, Encoder
from train_encoder import VGGLoss
import matplotlib.pyplot as plt
import pandas as pd

def image2tensor(image):
    image = torch.FloatTensor(image).permute(2,0,1).unsqueeze(0)/255.
    return (image-0.5)/0.5

def tensor2image(tensor):
    tensor = tensor.clamp_(-1., 1.).detach().squeeze().permute(1,2,0).cpu().numpy()
    return tensor*0.5 + 0.5

def imshow(img, size=5, cmap='jet'):
    plt.figure(figsize=(size,size))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()
    
def prepare_folder(file, folder):
    
    # create folder with the images top rated by each user
    
    if os.path.exists(folder):
           shutil.rmtree(folder) 
    os.mkdir(folder)    
     
    df = pd.read_csv(file)
 
    for rater in df['rater'].unique():
        
       if os.path.exists(folder +'/' + rater):
           shutil.rmtree(folder +'/' + rater) 
       os.mkdir(folder + '/' + rater)  
        
       images = df[df['rater']==rater]
    
       for img in images['image']:
            
            #print(img)
            
            try: 
                os.mkdir(folder +'/'+rater+'/'+img.rsplit('/', 1)[-1])
                
            except: 
                continue
                
            try:
                copyfile(img, folder + '/'+rater+'/'+img.rsplit('/', 1)[-1]+'/'+img.rsplit('/', 1)[-1])
                
            except:
                
                print(img, "file was removed from the dataset")

    print(folder, ' - the folder with top rated images has been created')
    
    
def generate_ideal():    
    
    device = 'cuda'
    image_size=256
    
    if os.path.exists('ideals'):
           shutil.rmtree('ideals') 
    os.mkdir('ideals')
    
    g_model_path = '/home/ubuntu/gen.pt'
    g_ckpt = torch.load(g_model_path, map_location=device)

    latent_dim = g_ckpt['args'].latent

    generator = Generator(image_size, latent_dim, 8).to(device)
    generator.load_state_dict(g_ckpt["g_ema"], strict=False)
    generator.eval()
    print('[generator loaded]')

    e_model_path = '/home/ubuntu/encoder.pt'
    e_ckpt = torch.load(e_model_path, map_location=device)

    encoder = Encoder(image_size, latent_dim).to(device)
    encoder.load_state_dict(e_ckpt['e'])
    encoder.eval()
    print('[encoder loaded]')

    truncation = 0.7
    trunc = generator.mean_latent(4096).detach().clone()
    
    batch_size = 5

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    
    
        
    
    dataset = datasets.ImageFolder(root='best/caucasian_male_39', transform=transform)
    loader = iter(torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True))

    imgs, _ = next(loader)
    imgs = imgs.to(device)

    with torch.no_grad():
        z0 = encoder(imgs)
        imgs_gen, _ =  generator([z0], 
                           input_is_latent=True,
                           truncation=truncation,
                           truncation_latent=trunc,
                           randomize_noise=False)

    imgs_real = torch.cat([img for img in imgs], dim=1)
    imgs_fakes = torch.cat([img_gen for img_gen in imgs_gen], dim=1)

    vgg_loss = VGGLoss(device)

    z = z0.detach().clone()

    z.requires_grad = True
    optimizer = torch.optim.Adam([z], lr=0.01)

    for step in range(500):
        imgs_gen, _ = generator([z], 
                           input_is_latent=True, 
                           truncation=truncation,
                           truncation_latent=trunc, 
                           randomize_noise=False)

    z_hat = encoder(imgs_gen)
    
    loss = F.mse_loss(imgs_gen, imgs) + vgg_loss(imgs_gen, imgs) + F.mse_loss(z0, z_hat)*2.0
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 
   
    n_interp = 5
    latent_interp = torch.zeros(1, z.shape[1], z.shape[2]).to(device)

    with torch.no_grad():
        for j in range(n_interp):
            latent_interp[0] = (float(1) * z[0] + float(1) * z[1]+ float(1) * z[2]+ float(1) * z[3]+ float(1) * z[4])/(5)
        #latent_interp[0] = (float(1) * z[0] + float(1) * z[1])/(2)
    
            imgs_gen, _ = generator([latent_interp],
                                input_is_latent=True,                                     
                                truncation=truncation,
                                truncation_latent=trunc,
                                randomize_noise=False)

    cv2.imwrite('ideals/caucasian_male_39.jpg', cv2.cvtColor(tensor2image(torch.cat([img_gen for img_gen in imgs_gen], dim=2))*255, cv2.COLOR_RGB2BGR))
    print('The file ideals/caucasian_male_39.jpg '+' has benn created')    

        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='the folder where save top rated images',
                         default = 'best' )
    parser.add_argument('--file', type=str, help='csv file with top rated images',
                         default = '../ME-beautydatabase/personal_best_images.csv' )
    args = parser.parse_args()
    
    folder = args.folder
    file = args.file
    
    prepare_folder(file, folder)
    generate_ideal()  