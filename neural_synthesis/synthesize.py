from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import copy, os
import pdb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
print(device)
from loss import *

# create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

def get_model(layer_name = 'pool4'):
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    i=j=1
    model = nn.Sequential()
    layer_names = []
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            name = 'conv{}_{}'.format(i, j)
            j+=1
        elif isinstance(layer, nn.ReLU):
            name = 'relu{}_{}'.format(i, j)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool{}'.format(i)
            i+=1; j=1;
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)
        if name == layer_name:
            break
    return model
    
def get_synth_model_and_losses(layer, input_image, device=device, area='IT', neuron_id = None, loss_func=NeuralInterpolation_Loss, 
                               rand_vec=None, vec_mag=None, interpol_image=None):
    
    # Get dCNN model features in response to target image.
    model = get_model(layer).to(device)
    resp = model(input_image) # Get response of model up through specified layer.
    target_resp = resp.view(resp.shape[0], -1).detach()
    if interpol_image is not None:
        resp2 = model(interpol_image)
        interpol_image = resp2.view(resp2.shape[0], -1).detach()
    
    synth_loss = loss_func(target_resp, layer=layer, area=area, neuron_id = neuron_id, rand_vec=rand_vec, 
                           vec_mag=vec_mag, image2_response=interpol_image)
    model.add_module("synth_loss", synth_loss)

    return model, synth_loss

def get_input_optimizer(init_image):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([init_image.requires_grad_()], lr=.1)
    return optimizer

def run_synthesis(layer, input_image, init_image, num_steps=300, saveLoc=None, saveName=None, saveInterval=500, area='IT', 
                  neuron_id=1, loss_func=NeuralInterpolation_Loss, rand_vec=None, vec_mag=None, interpol_image=None, verbose=True, tv_weight=1e-3):
    
    print('(run_synthesis) Building the style transfer model.')
    model, synth_loss = get_synth_model_and_losses(layer = layer, input_image = input_image, neuron_id=neuron_id, area=area,
                                                   loss_func=loss_func, rand_vec=rand_vec, vec_mag=vec_mag, interpol_image = interpol_image)
    optimizer = get_input_optimizer(init_image)

    print('(run_synthesis) Beginning optimization for image synthesis.')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            init_image.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(init_image)
            synth_score = synth_loss.loss

            loss = synth_score + tv_loss(init_image, tv_weight)
            loss.backward()

            run[0] += 1
            if run[0] % 100 == 0 and verbose==True:
                print(f'Step #{run[0]} synth loss: {synth_score.item()}')
            # If you want to save out intermediate steps.
            if run[0] % saveInterval == 0 and saveLoc is not None:
                tmp = init_image.clone()
                tmp.data.clamp_(0,1)
                if not os.path.isdir('{}/iters'.format(saveLoc[0])):
                    os.makedirs('{}/iters'.format(saveLoc[0]))
                imsave(tmp, f'{saveLoc[0]}/iters/{saveLoc[1].split(".")[0]}_step{run[0]}.png')

            return synth_score 

        optimizer.step(closure)

    # a last correction...
    init_image.data.clamp_(0, 1)

    return init_image.detach()
 

# Plotting and saving functions.
def imsave(tensor, savepath=None):
    unloader = transforms.ToPILImage()  # reconvert into PIL image

    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    if savepath is not None:
        image.save(savepath)
    return image

def image_loader(image_name):
    imsize = 256 if torch.cuda.is_available() else 128  # use small size if no gpu

    loader = transforms.Compose([
            transforms.Resize(imsize),  # scale imported image
            transforms.CenterCrop(imsize),
            transforms.ToTensor()])  # transform it into a torch tensor

    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

if __name__ == '__main__':
    LAYER = 'pool4'
    INPUT_IMAGE_PATH = '/home/gru/akshay/textures/input/leopard.jpg'
    
    input_image = image_loader(INPUT_IMAGE_PATH).to(device)
    init_image = torch.randn(input_image.data.size(), device=device)

    output_image = run_synthesis(LAYER, input_image, init_image)
