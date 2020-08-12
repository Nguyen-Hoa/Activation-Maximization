import torch
from numpy import percentile, tile

def norm_crop(img, threshold=0):

    norm = torch.norm(img, dim=0)
    norm = norm.numpy()

    # Create a binary matrix, with 1's wherever the pixel falls below threshold
    smalls = norm < percentile(norm, threshold)
    smalls = tile(smalls, (3,1,1))

    # Crop pixels from image
    crop = img - img*smalls
    return crop