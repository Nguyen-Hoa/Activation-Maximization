import torch
import torch.nn as nn
import torchvision.models as models

# Showing images
import cv2

# Reading images
from torchvision import transforms
from PIL import Image
from numpy import asarray, percentile, tile

# Gaussian Kernel
from scipy.ndimage import gaussian_filter

# https://medium.com/analytics-vidhya/deep-dream-visualizing-the-features-learnt-by-convolutional-networks-in-pytorch-b7296ae3b7f
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
denormalize = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225], std = [1/0.229, 1/0.224, 1/0.225] )
def image_converter(im):
    
    # move the image to cpu
    im_copy = im.cpu()
    
    # for plt.imshow() the channel-dimension is the last
    # therefore use transpose to permute axes
    im_copy = denormalize(im_copy.clone().detach()).numpy()
    im_copy = im_copy.transpose(1,2,0)
    
    # clip negative values as plt.imshow() only accepts 
    # floating values in range [0,1] and integers in range [0,255]
    im_copy = im_copy.clip(0, 1) 
    
    return im_copy

# https://stackoverflow.com/questions/50420168/how-do-i-load-up-an-image-and-convert-it-to-a-proper-tensor-for-pytorch
def get_image(img_path):
    img = Image.open(img_path) # use pillow to open a file
    img = img.resize((256, 256)) # resize the file to 256x256
    img = img.convert('RGB') #convert image to RGB channel

    img = asarray(img).transpose(-1, 0, 1) # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
    img = img/255
    img = torch.from_numpy(img) # create the image tensor
    return img

"""
Create a hook into target layer
    Example to hook into classifier 6 of Alexnet:
        alexnet.classifier[6].register_forward_hook(layer_hook('classifier_6'))
"""
def layer_hook(act_dict, layer_name):
    def hook(module, input, output):
        act_dict[layer_name] = output
    return hook

"""
Reguarlizer, crop by absolute value of pixel contribution
"""
def abs_contrib_crop(img, threshold=0):

    abs_img = torch.abs(img)
    smalls = abs_img < percentile(abs_img, threshold)
    
    return img - img*smalls

"""
Regularizer, crop if norm of pixel values below threshold
"""
def norm_crop(img, threshold=0):

    norm = torch.norm(img, dim=0)
    norm = norm.numpy()

    # Create a binary matrix, with 1's wherever the pixel falls below threshold
    smalls = norm < percentile(norm, threshold)
    smalls = tile(smalls, (3,1,1))

    # Crop pixels from image
    crop = img - img*smalls
    return crop

"""
Optimizing Loop
    Dev: maximize layer vs neuron
"""
def act_max(network, 
    input, 
    layer_activation, 
    layer_name, 
    unit, 
    steps=5, 
    alpha=torch.tensor(100), 
    generate_gif=False,
    path_to_gif='./',
    L2_Decay=False, 
    theta_decay=0.1,
    Gaussian_Blur=False,
    theta_every=4,
    theta_width=1,
    verbose=False,
    Norm_Crop=False,
    theta_n_crop=30,
    Contrib_Crop=False,
    theta_c_crop=30,
    ):

    best_activation = -float('inf')
    best_img = input

    for k in range(steps):

        input.retain_grad() # non-leaf tensor
        # network.zero_grad()
        
        # Propogate image through network,
        # then access activation of target layer
        network(input)
        layer_out = layer_activation[layer_name]

        # compute gradients w.r.t. target unit,
        # then access the gradient of input (image) w.r.t. target unit (neuron) 
        layer_out[0][unit].backward(retain_graph=True)
        img_grad = input.grad

        # Gradient Step
        # input = input + alpha * dimage_dneuron
        input = torch.add(input, torch.mul(img_grad, alpha))

        # regularization does not contribute towards gradient
        """
        DEV:
            Detach input here
        """
        with torch.no_grad():

            # Regularization: L2
            if L2_Decay:
                input = torch.mul(input, (1.0 - theta_decay))

            # Regularization: Gaussian Blur
            if Gaussian_Blur and k % theta_every is 0:
                temp = input.squeeze(0)
                temp = temp.detach().numpy()
                for channel in range(3):
                    cimg = gaussian_filter(temp[channel], theta_width)
                    temp[channel] = cimg
                temp = torch.from_numpy(temp)
                input = temp.unsqueeze(0)

            # Regularization: Clip Norm
            if Norm_Crop:
                input = norm_crop(input.detach().squeeze(0), threshold=theta_n_crop)
                input = input.unsqueeze(0)

            # Regularization: Clip Contribution
            if Contrib_Crop:
                input = abs_contrib_crop(input.detach().squeeze(0), threshold=theta_c_crop)
                input = input.unsqueeze(0)

        input.requires_grad_(True)

        if verbose:
            print('step: ', k, 'activation: ', layer_out[0][unit])

        if generate_gif:
            frame = input.detach().squeeze(0)
            frame = image_converter(frame)
            frame = frame * 255
            cv2.imwrite(path_to_gif+str(k)+'.jpg', frame)

        # Keep highest activation
        if best_activation < layer_out[0][unit]:
            best_activation = layer_out[0][unit]
            best_img = input

    return best_img

"""
Prepare Input from Image
"""
def load_image(path_to_image, device=False):
    tensor_image = get_image(path_to_image)
    tensor_image = normalize(tensor_image)
    tensor_image = tensor_image.unsqueeze(0)
    tensor_image.requires_grad = True
    if device:
        tensor_image = tensor_image.type(torch.cuda.FloatTensor)
    else:
        tensor_image = tensor_image.type(torch.FloatTensor)
    return tensor_image

"""
Prepare Dummy Input
"""
def load_dummy_image(device=False):
    dummy_image = torch.randn(3, 256, 256, requires_grad=True)
    dummy_image = normalize(dummy_image)
    if device:
        tensor_image = tensor_image.type(torch.cuda.FloatTensor)
    else:
        tensor_image = tensor_image.type(torch.FloatTensor)
    dummy_image = dummy_image.unsqueeze(0)
    return dummy_image
