import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale


def distance(x, X):
    #Make sure x is a tensor, not just an array:
    #tensor([[ 35.8645, -14.4873,  19.7296]], dtype=torch.float64)

    return torch.cdist(torch.unsqueeze(x, 0), X)
    

def distance_batch(x, X):
    return torch.cdist(torch.unsqueeze(x, 0), X)

def gaussian(dist, bandwidth):
    
    #Bandwith parameter is squared, then doubled and divides the distance, all wrapped in an exponent function.
    
    #Each other point has a weight now
    return torch.squeeze(torch.exp(-(2/bandwidth)**2 * dist))

def update_point(weight, X):
    
    weighted_sum = torch.matmul(weight,X)
    weight_sum = torch.sum(weight)
    normalised_weight = weighted_sum / weight_sum
    
    return normalised_weight

def update_point_batch(weight, X):
    weighted_sum = torch.transpose(torch.matmul(weight,X), 0, 1)
    
    #print('weighted_sum should be 3675 x 3', weighted_sum.shape)

    weight_sum = torch.sum(weight, dim = 1)

    #print('weight_sum, should be 3675 x 1', weight_sum.shape)

    #print('normalised_weight should be 3675 x 3')
    normalised_weight = torch.div(weighted_sum, weight_sum)
    
    normalised_weight = torch.transpose(normalised_weight, 0, 1)

    return normalised_weight

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    
    dist = distance_batch(X, X)
    #Distance matrix is ok, 0 diagonal

    weight = gaussian(dist, bandwidth)

    X_ = update_point_batch(weight, X)
    return X_

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        #X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
        print(f'step {_}')
    return X




scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab).cuda()).detach().cpu().numpy()

#X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
