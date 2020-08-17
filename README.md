# Activation Maximization

Activation maximization with PyTorch.

Regularizers from [Yosinski et al.](https://arxiv.org/abs/1506.06579)

## Overview

Activation maximization is a technique to visualize the features learned by a neural network. This is done via gradient ascent, or finding pixel values that maximally activate a particular neuron. In the following example, we will visualize a neuron in the final layer of Alexnet, trained on Imagenet, which activates for the flamingo class.

## Files

* act_max_util.py - Functions for activation maximization

* requirements.txt - Package dependencies

## Installation
<!-- Try alternative without Anaconda -->

The necessary packages are listed in *requirements.txt*. This file was produced within an Conda environment, so one simply has to run:

```conda create -n myenv --file requirements.txt```

to have all dependencies installed in a new conda environment.

## Usage

To begin, one needs a trained neural network and a neuron to visualize. In this example, we will use a pretrained Alexnet model from [torchvision](https://pytorch.org/docs/stable/torchvision/models.html), trained on *Imagenet* for image classification. We will visualize the neurons in the final layer (classification) since they produce easily interpretible results. 

---
Import packages.
```python
import torch                        # tensor creation
import torchvision.models as models # get pretrained model
import act_max_util as amu          # activation maximization tools
import cv2                          # display results
```

Load pretrained alexnet.

```python
alexnet = models.alexnet(pretrained=True)
```

Initialize a tensor with 3 channels for color, and height and width to match the size of data Alexnet was trained on, 227 by 227, with random values. We unsqueeze because PyTorch models receive data in batches, and so an additional dimmension is added. We also need the input to store gradient information to perform gradient ascent.

```python
input = torch.randn(3, 227, 227)
input = input.unsqueeze(0)
# input dimmensions become (1, 3, 227, 227)
input.requires_grad_(True)
```

Create a hook into the desired layer, this allows access to the activation value of the desired neuron during gradient ascent. To do so, we initialize an empty dictionary to store the layer's activations, and attach the *layer_hook()* function from *act_max_util.py* to the layer's *register_forward_hook()* call.

```python
activation_dictionary = {}
layer_name = 'classifier_final'

alexnet.classifier[-1].register_forward_hook(amu.layer_hook(activation_dictionary, layer_name))
```

The *amu.layer_hook()* function copies the layer's activations into *activation_dictionary*. The *layer_name* is used to access the activations in the dictionary. Since we registered *amu.layer_hook()* function to a forward hook, each forward pass will overwrite the layer's activation in the dictionary. The gradient ascent loop can then perform a backward pass on the target neuron in the target layer using the activation value.

Initialize activation maximization parameters.

```python
steps = 100                 # perform 100 iterations
unit = 130                  # flamingo class of Imagenet
alpha = torch.tensor(100)   # learning rate (step size) 
verbose = True              # print activation every step
L2_Decay = True             # enable L2 decay regularizer
Gaussian_Blur = True        # enable Gaussian regularizer
Norm_Crop = True            # enable norm regularizer
Contrib_Crop = True         # enable contribution regularizer
```

Perform activation maximization, storing result in *output*.

```python
output = amu.act_max(network=alexnet,
                input=input,
                layer_activation=activation_dictionary,
                layer_name=layer_name,
                unit=unit,
                steps=steps,
                alpha=alpha,
                verbose=verbose,
                L2_Decay=L2_Decay,
                Gaussian_Blur=Gaussian_Blur,
                Norm_Crop=Norm_Crop,
                Contrib_Crop=Contrib_Crop,
                )
```

Display output. The result of *amu.act_max()* is a tensor of the same dimensions as input, so we need to squeeze it back to 3x227x227. We will use OpenCV to display the image, which does not take torch tensors, so we will convert the output to a NumPy array. This is done with the utility function *amu.image_converter()*. We then convert all values in the final image from [0,1] to [0,255] for RGB channels.

```python
final_image = amu.image_converter(output.squeeze(0))
final_image = final_image * 255
# cv2.imshow('final image', final_image)
# cv2.waitKey(0)

# uncomment to save the image
# cv2.imwrite(path_to_save_dir + '.jpg', final_image)

```
gif tutorial in progress...

![flamingoA](./example_results/gifs/130_0.gif)
![flamingoB](./example_results/gifs/130_5.gif)
![flamingoC](./example_results/gifs/130_6.gif)