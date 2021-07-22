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

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')


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
    return image

def get_model_response(images, model):
    layer_responses = [] 
    for i_image in tqdm(range(len(images)), desc='Extracting model responses'): 
        input_image = image_loader(images[i_image])
        i_responses = model(input_image).detach().numpy()
        layer_responses.append(i_responses.flatten())
    return np.array(layer_responses)

def extract_neural_data(path): 
    # load our dataset
    data = h5py.File(path, 'r')
    # extract neural data
    neural_data = np.array(data['time_averaged_trial_averaged'])
    # extract variation labels
    variation_data = np.array(data['image_meta']['variation_level'])
    # extract IT data
    it_data = neural_data[:, data['neural_meta']['IT_NEURONS'] ]
    # extract V4 data 
    v4_data = neural_data[:, data['neural_meta']['V4_NEURONS'] ]
    # extract images 
    images = np.array(data['images'])
    # close h5py file 
    data.close() 
    return v4_data, it_data, variation_data, images


def generate_shuffle(n, ratio=3/4):
    sh = np.random.permutation(range( n ))
    return sh[:int(len(sh)*(ratio))], sh[int(len(sh)*(ratio)):]


def fit_response(model_, neural_, train_, test_, pls): 
    # find mapping between model responses and population
    pls.fit(model_[train_], neural_[train_] )
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
    train_, test_ = generate_shuffle(layer_.shape[0])
    # initialize data types 
    fits_ = {'it': {'train':[], 'test':[]}, 'v4': {'train':[], 'test':[]}}    
    coeffs_ = {'it': [], 'v4': []}
    print('---MODELING %s DATA'%['POPULATION', 'SINGLE UNIT'][per_neuron*1]) 
    for region in neural_responses: 
        print('---- %s'%region)
        # define region of interest 
        neural_ = neural_responses[region]
        if per_neuron: 
            for i_neuron in range( neural_.shape[1]): 
                print('------ %s NEURON %d'%(region.upper(), i_neuron))
                # single neuron's response 
                neuron_ = neural_[:, i_neuron]
                # find mapping between model responses and single neuron
                r_train, r_test, coeffs = fit_response( layer_, neuron_, train_, test_, pls )
                # store 
                fits_[region]['train'].append( r_train )
                fits_[region]['test'].append( r_test )
                coeffs_[region].append(coeffs)
                print('----NEURON %d CORRELATION: %.02f'%(i_neuron, r_test))
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
    parser.add_argument('-v', '--variation_type', default='all', help='specifies which variation level (V0, V3, V6, or all)')
    args = parser.parse_args()
    
    LAYER = args.layer
    N_COMPONENTS= args.nComponents
    variation_type = args.variation_type
    path_='/home/gru/akshay/ventral_neural_data.hdf5'
    save_location = '/home/gru/akshay/synthesis/model_fits'

    # Get neural data 
    print('Extracting neural data')
    v4_data, it_data, variation_, images = extract_neural_data( path_ ) 
    if args.variation_type == 'all': 
        select_indices = variation_ !=  b'V0'
    else: 
        select_indices = variation_ == args.variation_type.encode() 
    neural_ = {'v4': v4_data[select_indices], 'it': it_data[select_indices]}
    images = images[select_indices]

    print('Loading dCNN model')
    model = get_model(layer_name = args.layer)
    model_response = get_model_response(images, model)
    print('MODEL RESPONSES SHAPE:', model_response.shape)  
    
    print('-BEGINNING LAYER-NEURAL FIT') 
    np.random.seed(3)
    it_test, it_train, v4_test, v4_train = [] , [] , [] , [] 
    it_weights, v4_weights = [], []

    fits_, coeffs_ = model_neural_map(model_response, neural_, args.nComponents, per_neuron=True)
    it_test.append(fits_['it']['test']  ) 
    it_train.append(fits_['it']['train']) 
    v4_test.append(fits_['v4']['test']  ) 
    v4_train.append(fits_['v4']['train'])

    it_weights.append(coeffs_['it'])
    v4_weights.append(coeffs_['v4'])
    
    save_file_name = f'{save_location}/vgg19_{args.layer}-fit_{args.nComponents}-components_{args.variation_type}.pickle'

    
    save_data = {'it_test':it_test, 'it_train':it_train, 'v4_test':v4_test, 'v4_train':v4_train,
                 'it_weights': it_weights, 'v4_weights': v4_weights}
    
    print('SAVING...') 
    with open(save_file_name, 'wb') as handle: 
        pickle.dump(save_data, handle) 
    print(':D') 
