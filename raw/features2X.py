# convert features to the format of X
# input 
#   features, patch images' feature extracted from ResNet;

# output
#   X, the input of the fully connected regress network with the dimension of M x 1000, which 
#   M is the number of the training image patches in formula (1);


import numpy as np


def features2X(features):
    n = 0
    for c in features:
        n = n + c.size

    n = int(n/1000)
    X = np.zeros((n, 1000))
    k = 0
    for patch_feature in features:
        h = int(patch_feature.size/1000)
        X[k:k + h, :] = patch_feature.reshape(h, 1000)
        k = k + h

    return X
