import torch                        # tensor creation
import torchvision.models as models # get pretrained model
import act_max_util as amu          # activation maximization tools
import cv2                          # display results

alexnet = models.alexnet(pretrained=True)

input = torch.randn(3, 227, 227)
input = input.unsqueeze(0)
input.requires_grad_(True)
# input dimmensions become (1, 3, 227, 227)

activation_dictionary = {}
layer_name = 'classifier_6'

alexnet.classifier[-1].register_forward_hook(amu.layer_hook(activation_dictionary, layer_name))

steps = 250                 # perform 100 iterations
unit = 130                  # flamingo class of Imagenet
alpha = torch.tensor(100)   # learning rate (step size) 
verbose = True              # print activation every step
L2_Decay = True             # enable L2 decay regularizer
Gaussian_Blur = True        # enable Gaussian regularizer
Norm_Crop = True            # enable norm regularizer
Contrib_Crop = True         # enable contribution regularizer

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

final_image = amu.image_converter(output.squeeze(0))
final_image = final_image * 255
# cv2.imshow('final image', final_image)
# cv2.waitKey(0)

# uncomment to save the image
cv2.imwrite('./example_result_0-2.jpg', final_image)

