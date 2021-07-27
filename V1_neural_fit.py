import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torchvision.transforms as transforms
import torchvision.models as models

import copy, os, time, sys, h5py, pickle, argparse
from sklearn.cross_decomposition import PLSRegression
from skimage.transform import resize as imresize
from tqdm import tqdm

# desired size of the output image
imsize = 256 if torch.cuda.is_available() else 128  # use small size if no gpu
loader = transforms.Compose([transforms.ToPILImage(),
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor
unloader = transforms.ToPILImage()  # reconvert into PIL image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
print(f'Using device: {device}')

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

def rgb_from_grey(grey):
    wide, high = grey.shape
    rgb = np.empty((wide, high, 3), dtype=np.uint8)
    rgb[:, :, 2] =  rgb[:, :, 1] =  rgb[:, :, 0] =  grey
    return rgb


def image_loader(input_image):
    image = rgb_from_grey(input_image)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def get_model_response(images, model):
    layer_responses = [] 
    for i_image in tqdm(range(len(images)), desc='Extracting model responses'): 
        input_image = image_loader(images[i_image])
        i_responses = model(input_image).detach().cpu().numpy()
        layer_responses.append(i_responses.flatten())
    return np.array(layer_responses)

def load_v1_data(v1_data_path):
    with open(v1_data_path, 'rb') as f:
        v1_data = pickle.load(f)
    
    # responses: 7250 (nImages) x 166 (nNeurons)
    responses = np.nanmean(v1_data['responses'], axis=0) 
    
    # images: 7250 (nImages) x 140 (width) x 140 (height)
    images = v1_data['images']
    
    return responses, images

def generate_shuffle(n, ratio=0.75):
    sh = np.random.permutation(range( n ))
    return sh[:int(len(sh)*(ratio))], sh[int(len(sh)*(ratio)):]

def fit_response(model_, neural_, train_, test_, pls): 
    # find mapping between model responses and population
    print('starting fitting')
    print(model_[train_].shape, neural_[train_].shape)
    pls.fit(model_[train_], neural_[train_] )
    print('fit complete')
    # coeffs
    coef = pls.coef_ # nFeatures x 1
    # extract best fit to training data
    pred_train = pls.predict(model_[train_]).flatten()
    # extract predictions to testing data
    pred_test = pls.predict(model_[test_]).flatten()
    # compute correlation between model and neural responses in training data 
    train_r =  np.corrcoef(pred_train, neural_[train_].flatten())
    # compute correlation between model and neural responses in testing data
    test_r = np.corrcoef(pred_test, neural_[test_].flatten())
    return train_r[0, 1], test_r[0, 1], coef


def model_neural_map(layer_, neural_responses,  n_components=25, per_neuron=True): 
    # define pls model 
    pls = PLSRegression(n_components=n_components, scale=False)    
    # we'll use the same split across regions 
    train_, test_ = generate_shuffle(layer_.shape[0], ratio=0.1)
    # initialize data types 
    fits_ = {'v1': {'train':[], 'test':[]}}    
    coeffs_ = {'v1': []}
    print(f'---MODELING {["POPULATION", "SINGLE UNIT"][per_neuron*1]} DATA') 
    for region in neural_responses: 
        print(f'---- {region} ----')
        # define region of interest 
        neural_ = neural_responses[region]
        if per_neuron: 
            for i_neuron in range( neural_.shape[1]): 
                print(f'------ {region.upper()} NEURON {i_neuron}')
                # single neuron's response 
                neuron_ = neural_[:, i_neuron]
                nan_idx = np.where(np.isnan(neuron_))[0]
                if len(nan_idx) > 0:
                    print(f'Found {len(nan_idx)} NaNs for this neuron {i_neuron}')
                # find mapping between model responses and single neuron
                r_train, r_test, coeffs = fit_response( layer_, neuron_, 
                                                       np.setdiff1d(train_,nan_idx), 
                                                       np.setdiff1d(test_, nan_idx), pls )
                print('done fitting')
                # store 
                fits_[region]['train'].append( r_train )
                fits_[region]['test'].append( r_test )
                coeffs_[region].append(coeffs)
                print(coeffs.shape)
                print(f'----NEURON {i_neuron} CORRELATION: {r_test:.02f}')
            print(np.array(coeffs_[region]).shape)
        else: 
            # fit population 
            train_r, test_r = fit_response(layer_, neural_, train_, test_, pls)
            # store correlations
            fits_[region]['train'].append( train_r )
            fits_[region]['test'].append( test_r )            
    return fits_, coeffs_


if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layer", default = "pool4", help="specifies which layer to get features from to fit IT/V4 responses")
    parser.add_argument('-n', '--nComponents', default=5, type=int, help='specifies how many components to use in the PLS fit')
    args = parser.parse_args()
    
    LAYER = args.layer
    N_COMPONENTS= args.nComponents
    #path_='/home/gru/akshay/ventral_neural_data.hdf5'
    
    path_='/home/gru/akshay/neurint/neural_data/cadena_ploscb_data.pkl'
    save_location = '/home/gru/akshay/neurint/model_fits'

    # Get neural data 
    print('Extracting neural data')
    v1_data, images = load_v1_data( path_ )
    neural_ = {'v1': v1_data}

    print('Loading dCNN model')
    model = get_model(layer_name = args.layer)
    model_response = get_model_response(images, model)
    print('MODEL RESPONSES SHAPE:', model_response.shape)  
    
    print('-BEGINNING LAYER-NEURAL FIT') 
    np.random.seed(3)
    v1_test, v1_train, v1_weights = [], [], []

    fits_, coeffs_ = model_neural_map(model_response, neural_, args.nComponents, per_neuron=True)
    v1_test.append(fits_['v1']['test']  ) 
    v1_train.append(fits_['v1']['train'])
    v1_weights.append(coeffs_['v1'])
    print(np.array(coeffs_['v1']).shape)
    save_file_name = f'{save_location}/V1_vgg19_{args.layer}-fit_{args.nComponents}-components.pickle'

    
    save_data = {'v1_test':v1_test, 'v1_train':v1_train, 'v1_weights': v1_weights}
    
    print('SAVING...') 
    with open(save_file_name, 'wb') as handle: 
        pickle.dump(save_data, handle) 
    print(':D') 
