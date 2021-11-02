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

    return torch.cdist(x, X)
    

def distance_batch(x, X):
    return torch.cdist(x, X)

def gaussian(dist, bandwidth):
    #Bandwith parameter is squared, then doubled and divides the distance, all wrapped in an exponent function.
    
    return torch.exp(-(2/bandwidth)**2 * dist)

def update_point(weight, X):

    return torch.sum(weight*X) / torch.sum(weight)

def update_point_batch(weight, X):
    return torch.sum(weight*X) / torch.sum(weight)

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    pass

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        X = meanshift_step(X)   # slow implementation
        # X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()

print(X)
# X = meanshift(torch.from_numpy(data).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
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


#Pseudo Code
""" 
while not_converged():
    for i, point in enumerate(points):
        #distance for the given point to all points
        distances = distance(point, points)

        #turn distance into wights using a gaussian
        weights = gaussian(dist, bandwidth = 2.5)

        # update the points by calculating weightd mean of all points
        points[i] = update_point(weight, X)

return points """