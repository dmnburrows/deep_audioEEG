import numpy as np
import pandas as pd
import os
import glob

#====================
def mat_load(path):
#===================
    """ 
    This function loads .mat EEG files and processes them into python dictionaries.

    Inputs:
        path (str): path to .mat file

    Outputs:
        dic (dictionary): dict containing ID name and time series npy array.
    
    """
    import scipy.io

    dic = scipy.io.loadmat(path)
    dic['data'] = dic[os.path.basename(path).split('.')[0]]
    dic['ID'] = os.path.basename(path).split('.')[0]
    dic.pop(os.path.basename(path).split('.')[0], None)
    return(dic)