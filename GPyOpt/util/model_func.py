import numpy as np

def model_predict(x):
    a = np.copy(x)
    a[x <= -5] = .8
    a[(x > -5) & (x <= 0)] = .6
    a[(x >0) & (x <= 5 )] = .4
    a[x > 5]= .1
    return a