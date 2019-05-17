import numpy as np
import boto3
import botocore
import lightgbm as lgb
from numba import jit

# def model_predict(x):
#     a = np.copy(x)
#     a[x <= -5] = .8
#     a[(x > -5) & (x <= 0)] = .6
#     a[(x >0) & (x <= 5 )] = .4
#     a[x > 5]= .1
#     return a

@jit
def model_predict(x,data,model):
    n = np.empty((x.shape[0],x.shape[1]))
    col = data.columns
    for counter in range(0,x.shape[0]):
        if 'IPO_optimise_factor_Bronze' in col:
            data['IPO_optimise_factor_Bronze'] = x[counter,:,0]
        if 'IPO_optimise_factor_Silver' in col:
            data['IPO_optimise_factor_Silver'] = x[counter,:,1]
        if 'IPO_optimise_factor_Gold' in col:
            data['IPO_optimise_factor_Gold'] = x[counter,:,2]
        n[counter,:] =  model.predict_proba(data.values)[:,0]
    return n



