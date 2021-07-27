from synthesize import *
from loss import *
from PIL import ImageOps
import os
import argparse

print(device)

def run_neural_interpolation(image1_name, image2_name, n_intervals=5, uid='s1', save_path = '/home/gru/akshay/neurint/outputs', num_steps=1000,
                             layer='pool4', area='IT', input_image_dir='/home/gru/akshay/textures/input', tv_weight=1e-2, 
                             loss_func = NeuralInterpolation_Loss):
    if save_path is not None:
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
            os.mkdir(save_path + '/color')
            os.mkdir(save_path + '/bw')
    
    tv_weight_str= f'{tv_weight:.0e}'.replace('+', '')

    image1 = image_loader(f'{input_image_dir}/{image1_name}.jpg').to(device)
    image2 = image_loader(f'{input_image_dir}/{image2_name}.jpg').to(device)

    outputs = []

    intervals = np.linspace(0,1,n_intervals)
    for i in range(len(intervals)):
        print(f'---Generating image {intervals[i]*100}% of the way from {image1_name} to {image2_name}---')
        init_image = torch.randn(image1.data.size(), device=device)
        output_image = run_synthesis(layer, image1, init_image, num_steps=num_steps, area = area, tv_weight=tv_weight,
                                     loss_func = loss_func, vec_mag=intervals[i], interpol_image=image2);
        if save_path is not None:
            if loss_func is NeuralInterpolation_Loss:
                savename = f'{image1_name}_{image2_name}_{int(100*intervals[i])}_{layer}-{area}_{uid}.png'
            else:
                savename = f'{image1_name}_{image2_name}_{int(100*intervals[i])}_{layer}_{uid}.png'
            output = imsave(output_image, savepath=f'{save_path}/color/{savename}', grayscale=False)
            output = imsave(output_image, savepath=f'{save_path}/bw/{savename}', grayscale=True)
        outputs.append(output_image)
        
    return outputs
    
# Plotting and saving functions.
def imsave(tensor, savepath=None,  grayscale=True):
    unloader = transforms.ToPILImage()  # reconvert into PIL image

    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    if grayscale==True:
        image = ImageOps.grayscale(image)
        #image = image.convert('L')
    if savepath is not None:
        image.save(savepath)
    return image

if __name__ == '__main__':
    ### Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layer", default="pool4", help="specifies the base directory in which the stimuli (both input and outputs) will be located")
    parser.add_argument('-a', '--area', default='IT', help='which brain area')
    args = parser.parse_args()

    input_image_dir='/home/gru/akshay/textures/input'
    for i in range(5):
        print(f'--- Synthesizing sample {i+1} ---')
        outputs = run_neural_interpolation('rocks', 'leaves', uid=f's{i+1}', num_steps=5000, input_image_dir = input_image_dir, 
                                           loss_func = NeuralInterpolation_Loss, layer=args.layer, area=args.area);
        
