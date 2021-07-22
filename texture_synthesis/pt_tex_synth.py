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

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature, **kwargs):
        super(StyleLoss, self).__init__()
        self.activation = target_feature.detach()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class StyleLossPCA(nn.Module):
    def __init__(self, target_feature, layerName, which_pc = None, pc_step_size=None, pc_layer = 'pool2'):
        super(StyleLossPCA, self).__init__()
        self.pca = np.load('/scratch/groups/jlg/texpca/{}_dims_histmatch.npy'.format(layerName)).item()['pca']
        self.target = gram_matrix(target_feature).detach().to(device)
        self.components = torch.from_numpy(self.pca.components_.T).to(device) # nFeatures x nComponents
        self.pca_mean = torch.from_numpy(self.pca.mean_).to(device).view(1,-1) # (nFeatures,)
        self.target_pca = self.transform(self.target.view(1,-1))
        if which_pc is not None and layerName == pc_layer:
            self.target_pca[0,which_pc] = self.target_pca[0,which_pc] + pc_step_size

    def forward(self, input):
        G = self.transform(gram_matrix(input).view(1,-1))
        self.loss = F.mse_loss(G, self.target_pca)
        return input

    def transform(self, mtx):
        mtx_pca = torch.mm(mtx - self.pca_mean, self.components)
        return mtx_pca

class StyleLossNMF(nn.Module):
    def __init__(self, target_feature, layerName, which_pc = None, pc_step_size=None, pc_layer='pool2', **kwargs):
        super(StyleLossNMF, self).__init__()
        self.nmf = np.load('/scratch/groups/jlg/texpca/{}_dims_nmf.npy'.format(layerName)).item()['nmf']
        self.target = gram_matrix(target_feature).detach().to(device)
        self.components = torch.from_numpy(self.nmf.components_.T).to(device) # nFeatures x nDimensions
        self.target_nmf = self.transform(self.target.view(1,-1))
        print(self.target_nmf.shape)
        if which_pc is not None and layerName == pc_layer:
          self.target_nmf[0,which_pc] = self.target_nmf[0,which_pc] + pc_step_size

    def forward(self, input):
        G = self.transform(gram_matrix(input).view(1,-1))
        self.loss = F.mse_loss(G, self.target_nmf)
        return input

    def transform(self, mtx):
        mtx_lda = torch.mm(torch.abs(mtx), self.components.float())
        return mtx_lda

class StyleLossDiag(nn.Module):
    def __init__(self, target_feature, **kwargs):
        super(StyleLossDiag, self).__init__()
        self.target = torch.diag(gram_matrix(target_feature).detach())

    def forward(self, input):
        G = torch.diag(gram_matrix(input))
        self.loss = F.mse_loss(G, self.target)
        return input

class StyleLossPool2(nn.Module):
    def __init__(self, target_feature, layerName=None):
        super(StyleLossPool2, self).__init__()
        if layerName == 'pool2':
            self.target = 10000*torch.ones_like(target_feature)
        else:
            self.target = torch.zeros_like(target_feature)

    def forward(self, input):
        pdb.set_trace()
        assert (input >= 0. & input <= 1.).all()

        self.loss = F.mse_loss(input, self.target)
        return input

# create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std
    

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, style_layers, device=device, style_loss_func=StyleLoss,
                               which_pc = None, pc_step_size = None):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 1  # increment every time we see a pool layer
    j = 1  # increment for each conv layer within a block
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

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = style_loss_func(target_feature, layerName=name, which_pc=which_pc, 
                                          pc_step_size = pc_step_size)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], style_loss_func):
            break

    model = model[:(i + 1)]

    return model, style_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=.1)
    return optimizer

def run_texture_synthesis(cnn, normalization_mean, normalization_std,
                       style_img, input_img, num_steps=300, saveLoc=None, saveName=None, saveInterval=500,
                       style_weight=1000000, style_layers=style_layers_default, 
                       style_loss_func=StyleLoss, which_pc=None, pc_step_size=None):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, 
                                                     style_img, style_layers=style_layers, 
                                                     style_loss_func=style_loss_func, which_pc=which_pc,
                                                     pc_step_size=pc_step_size)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0

            for sl in style_losses:
                style_score += sl.loss

            style_score *= style_weight

            loss = style_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
              print('Step #{} style loss: {:4f}'.format(
                    run[0], style_score.item()))
            if run[0] % saveInterval == 0 and saveLoc is not None:
              tmp = input_img.clone()
              tmp.data.clamp_(0,1)
              if not os.path.isdir('{}/iters'.format(saveLoc[0])):
                os.makedirs('{}/iters'.format(saveLoc[0]))
              imsave(tmp, '{}/iters/{}_step{}.png'.format(saveLoc[0], saveLoc[1].split('.')[0], run[0]))

            return style_score 

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

def get_layer_features(cnn, normalization_mean, normalization_std,
                       style_img, style_layers=style_layers_default):
    model, style_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, 
                                                     style_img, style_layers=style_layers)
    sl = {};
    for i in range(len(style_layers)):
        sl[style_layers[i]] = style_losses[i].target.cpu().detach().numpy();

    return sl

def get_layer_activations(cnn, normalization_mean, normalization_std,
                       style_img, style_layers=style_layers_default):
    model, style_losses = get_style_model_and_losses(cnn, normalization_mean, normalization_std, 
                                                     style_img, style_layers=style_layers)
    sl = {};
    for i in range(len(style_layers)):
        sl[style_layers[i]] = style_losses[i].activation.cpu().detach().numpy();

    return sl


# Plotting and saving functions.
unloader = transforms.ToPILImage()  # reconvert into PIL image

def imsave(tensor, savepath=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(savepath)
  
