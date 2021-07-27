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
import synthesize

ROOTDIR = '/home/gru/akshay/neurint'
device = synthesize.device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_fit_weights(layer, area, ver='V6'):
    if area == 'V1':
        filepath = f'{ROOTDIR}/model_fits/{area}_vgg19_{layer}-fit_5-components.pickle'
    else:
        filepath = f'{ROOTDIR}/model_fits/{area}_vgg19_{layer}-fit_5-components_{ver}.pickle'
    print(filepath)
    if os.path.isfile(filepath):
        results = np.load(filepath, allow_pickle=True)
        weights = np.array(results[f'{area.lower()}_weights']).squeeze()
        return weights
    else:
        print(f'Weight file not found : {filepath}')
        return None

def tv_loss(img, tv_weight):
    ''' tv_loss
        - Purpose: Compute total variation loss.
        - Inputs:
            - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
            - tv_weight: Scalar giving the weight w_t to use for the TV loss.
        - Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss for img weighted by tv_weight.
    '''
    w_variance = torch.sum(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.sum(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = tv_weight * (h_variance + w_variance)
    return loss


class Activation_Loss(nn.Module):
    ''' Activation_Loss
        - Purpose: match activations from a particular layer (i.e. Feather et al., Neurips)
        - Inputs: 
            - target_response: vector of activations from some layer to match.
    '''
    def __init__(self, target_response, layer, **kwargs):
        super(Activation_Loss, self).__init__()
        self.target = target_response

    def forward(self, input):
        self.loss = F.mse_loss(input.view(input.shape[0],-1), self.target)
        return input


class IT_Loss(nn.Module):
    ''' IT_Loss
        - Purpose: match the predicted IT neural response (i.e. the linear transform of a particular layer which best predicts IT) to a target image.
    '''
    def __init__(self, target_response, layer, area='IT', rand_vec=None, vec_mag=None, **kwargs):
        super(IT_Loss, self).__init__()
        self.target_response = target_response
        
        # Load weights and multiply by model features to get the target feature.
        weights = load_fit_weights(layer=layer, area=area)
        
        self.W = torch.from_numpy(weights.T).to(device)
        self.target = torch.matmul(self.target_response.float(), self.W.float()).detach()
        if rand_vec is not None:
            self.target = self.target + vec_mag * rand_vec

    def forward(self, input):
        G = torch.matmul(input.view(input.shape[0], -1).float(), self.W.float())
        self.loss = F.mse_loss(G, self.target)
        return input

class MEI_Loss(nn.Module):
    ''' MEI_Loss
        - Purpose: Generate an image which maximizally drives a single neuron's response.
        - this is the loss used in the Bashivan et al Science paper
    '''
    def __init__(self, target_response, layer, area='IT', neuron_id=1, **kwargs):
        super(MEI_Loss, self).__init__()
        self.target_response = target_response
        
        # Load weights and multiply by model features to get the target feature.
        weights = load_fit_weights(layer=layer, area=area)
        
        self.neuron_id = neuron_id
        self.W = torch.from_numpy(weights.T).to(device)
        self.target = torch.matmul(self.target_response.float(), self.W.float()).detach()

    def forward(self, input):
        G = torch.matmul(input.view(input.shape[0], -1).float(), self.W.float())
        #print(G.shape)
        self.loss = (-G[0,self.neuron_id])
        return input


class NeuralInterpolation_Loss(nn.Module):
    ''' NeuralInterpolation_Loss
        - Purpose: match the predicted IT neural response (i.e. linear transform of a particular layer) as you interpolate between two images 
    '''
    def __init__(self, image1_response, layer, area='IT', image2_response=None, vec_mag=None, **kwargs):
        super(NeuralInterpolation_Loss, self).__init__()
        
        # Load weights and multiply by model features to get the target feature.
        weights = load_fit_weights(layer=layer, area=area)
        
        self.W = torch.from_numpy(weights.T).to(device)
        print(image1_response.shape, self.W.shape)
        print(layer, area)
        self.image1_resp = torch.matmul(image1_response.float(), self.W.float()).detach()
        self.image2_resp = torch.matmul(image2_response.float(), self.W.float()).detach()
        distance_vec = self.image2_resp - self.image1_resp
        
        self.target = self.image1_resp + vec_mag * distance_vec

    def forward(self, input):
        G = torch.matmul(input.view(input.shape[0], -1).float(), self.W.float())
        self.loss = F.l1_loss(G, self.target)
        return input

class dCNNInterpolation_Loss(nn.Module):
    ''' dCNNInterpolation_Loss
          - Purpose: match the predicted IT neural response as you interpolate between two images 
              (i.e. the linear transform of a particular layer which best predicts IT)
    '''
    def __init__(self, image1_response, layer, area='IT', image2_response=None, vec_mag=None, **kwargs):
        super(dCNNInterpolation_Loss, self).__init__()
        
        # Interpolate through dCNN feature space.
        self.image1_resp = image1_response.detach()
        self.image2_resp = image2_response.detach()
        distance_vec = self.image2_resp - self.image1_resp
        
        self.target = self.image1_resp + vec_mag * distance_vec

    def forward(self, input):
        G = input.view(input.shape[0],-1)
        self.loss = F.l1_loss(G, self.target)