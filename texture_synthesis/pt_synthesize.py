from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms
import torchvision.models as models

import copy, os, time
from pt_tex_synth import *
import argparse
import pdb

# desired size of the output image
imsize = 256 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    
def imsave(tensor, savepath=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(savepath)    

########## MAIN: SPECIFY OPTIONS:
if __name__ == "__main__":
  ### Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--stim_dir", default="/home/gru/akshay/textures", help="specifies the base directory in which the stimuli (both input and outputs) will be located")
  parser.add_argument("-i", "--img_path", default="input/tulips.jpg", help="specifies the path of the original image")
  parser.add_argument("-o", "--out_dir", default="out_color/pool2", help="specifies the path of the output directory")
  parser.add_argument("-l", "--layer", default="pool4", help="specifies the layer to match statistics through")
  parser.add_argument("-s", "--nSplits", default=1, help="specifies the number of sections to split each dimension into (e.g. 1x1, 2x2, nxn)")
  parser.add_argument('-n', '--nSteps', type=int, default=5000, help="specifies the number of steps to run the gradient descent for")
  parser.add_argument('-g', '--gramLoss', default='gram', help='specifies the type of gram matrix loss function (choices: "gram", "diag", "pca", or "lda")')
  parser.add_argument('-z', '--pc_step_size', default=None, type=int, help='step size along PC (only works if gramLoss==pca)')
  parser.add_argument('-p', '--which_pc', default=None, type=int, help='which PC to step along - starts from 0 (only works if gramLoss==pca)')
  parser.add_argument('-x', '--sample', default=1, type=int, help='sample number to add onto end of ssave name')
  args = parser.parse_args()

  ### PyTorch Models (VGG)
  # Load the model (VGG19)
  cnn = models.vgg19(pretrained=True).features.to(device).eval()

  # This is the normalization mean and std for VGG19.
  cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
  cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

  ### Specify texture synthesis parameters.
  # Load the original image
  style_img = image_loader(args.stim_dir + '/' + args.img_path)
  img_name = args.img_path.split('/')[-1].split('.')[0] # get just the name (e.g. cherries) without path or extension

  # Specify which layers to match
  all_layers = ['conv1_1', 'pool1', 'pool2', 'pool3', 'pool4'];
  this_layers = all_layers[:all_layers.index(args.layer)+1]
  #this_layers = ['conv1_1', 'pool1', 'pool2', 'pool3', 'pool4'];
  print(this_layers)

  ## Run Texture Synthesis
  # Randomly initialize white noise input image
  input_img = torch.randn(style_img.data.size(), device=device)

  # Make directory if it doesn't already exist
  if not os.path.isdir(args.stim_dir + '/' + args.out_dir):
    os.makedirs(args.stim_dir + '/' + args.out_dir) 

  # Save as: e.g. 1x1_pool2_cherries.png
  saveName = '{}x{}_{}_{}_smp{}.png'.format(args.nSplits, args.nSplits, args.layer, img_name, args.sample)
  saveInterval=100

  # Specify which loss function to use.
  if args.gramLoss == 'diag':
    print('Using diagonal of gram matrix to compute style loss')
    style_loss_func = StyleLossDiag
  elif args.gramLoss == 'pca':
    print('Using PCA of gram matrix to compute style loss')
    style_loss_func = StyleLossPCA
    if args.pc_step_size is not None:
      saveName = '{}x{}_{}_{}_PC{}_{}.png'.format(args.nSplits, args.nSplits, args.layer, img_name, args.which_pc, args.pc_step_size)
  elif args.gramLoss == 'lda':
    print('Using LDA of gram matrix to compute style loss')
    style_loss_func = StyleLossLDA
  elif args.gramLoss == 'pool2':
    print('Maximizing pool2 features, minimizing all others')
    style_loss_func = StyleLossPool2
  elif args.gramLoss == 'nmf':
    print('Using NMF of gram matrix to compute style loss')
    style_loss_func = StyleLossNMF
    if args.pc_step_size is not None:
      saveName = '{}x{}_{}_{}_NMF{}_{}.png'.format(args.nSplits, args.nSplits, args.layer, img_name, args.which_pc, args.pc_step_size)
  else:
    print('Using full gram matrix to compute style loss')
    style_loss_func = StyleLoss

  # Check if we're on GPU or CPU, then run! 
  gpu_str = "Using GPU" if torch.cuda.is_available() else "Using CPU"
  print("{} to synthesize textures at layer {}, nSplits: {}, image: {}, numSteps: {}".format(gpu_str, args.layer, args.nSplits, img_name, args.nSteps))
  tStart = time.time()
  output_leaves = run_texture_synthesis(cnn, cnn_normalization_mean, cnn_normalization_std, style_img, 
                                        input_img, num_steps=args.nSteps, style_layers=this_layers, 
                                        saveLoc=[args.stim_dir + '/' + args.out_dir, saveName], 
                                        style_loss_func=style_loss_func, which_pc=args.which_pc, 
                                        pc_step_size = args.pc_step_size, saveInterval=saveInterval)

  tElapsed = time.time() - tStart
  print('Done! {} steps took {} seconds. Saving as {} now.'.format(args.nSteps, tElapsed, saveName))
  # Save final product to output directory
  imsave(output_leaves, args.stim_dir + '/' + args.out_dir + '/' + saveName);

