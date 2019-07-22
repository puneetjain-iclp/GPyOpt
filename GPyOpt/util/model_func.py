import numpy as np
from numba import jit
import boto3
import botocore
import lightgbm as lgb
# from numba import jit

# def model_predict(x):
#     a = np.copy(x)
#     a[x <= -5] = .8
# #     a[(x > -5) & (x <= 0)] = .6
# #     a[(x >0) & (x <= 5 )] = .4
# #     a[x > 5]= .1
# #     return a

def model_predict(x,mod_data,model):
    elasticity_vars =["ipo_elasticity_factor_bronze","ipo_elasticity_factor_silver","ipo_elasticity_factor_gold"]
    col = mod_data.columns
    n = np.empty((x.shape[0],x.shape[1]))
    for counter in range(0,x.shape[0]):
        mod_data = create_features(mod_data,counter,x,elasticity_vars,col)        
        n[counter,:] =  model.predict_proba(mod_data.values)[:,1]
        # print(model.predict_proba(mod_data.values)[:,1].sum()/len(model.predict_proba(mod_data.values)[:,1]))
    return n


def create_features(mod_data,counter,x,elasticity_vars,col):
    if 'ipo_elasticity_factor_bronze' in col:
        mod_data['ipo_elasticity_factor_bronze'] = x[counter,:,0]
    if 'ipo_elasticity_factor_silver' in col:
        mod_data['ipo_elasticity_factor_silver'] = x[counter,:,1]
    if 'ipo_elasticity_factor_gold' in col:
        mod_data['ipo_elasticity_factor_gold'] = x[counter,:,2]
    if 'b_s_diff' in col:
        mod_data['b_s_diff'] = x[counter,:,0] - \
            x[counter,:,1]
    if 'b_g_diff' in col:
        mod_data['b_g_diff'] = x[counter,:,0] - \
            x[counter,:,2]
    if 's_g_diff' in col:
        mod_data['s_g_diff'] = x[counter,:,1] - \
            x[counter,:,2]
    # if 'elas_mean' in col:
    #     mod_data['elas_mean'] = mod_data[elasticity_vars].mean(axis=1)
    # if 'elas_sum' in col:
    #     mod_data['elas_sum'] = mod_data[elasticity_vars].sum(axis=1)
    # if 'elas_stdev' in col:
    #     mod_data['elas_stdev'] = mod_data[elasticity_vars].std(axis=1)
    return mod_data

