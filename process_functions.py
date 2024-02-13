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


#====================
def lowpassfilt(signal, cutoff, fs, N=4):
#===================
    """ 
    This function applies a low pass filter to time series data.

    Inputs:
        signal (np array): 
        cutoff (float): cutoff frequency for lowpass filter, all frequencies greater will be removed
        fs (float): frames per second sampling rate
        N (int): order of butterworth filter

    Outputs:
        filt (np array): filtered data
    
    """

    from scipy.signal import butter, filtfilt
    # Normalize the cutoff frequency with respect to the Nyquist frequency
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq

    # Design the Butterworth low-pass filter (Nth order, where N=4 is a common choice)
    N = 4  # Filter order
    b, a = butter(N, norm_cutoff, btype='low', analog=False)
    filt = filtfilt(b, a, signal)
    return(filt)


#====================
def notchfilt(signal, fs, f0, Q):
#===================
    """ 
    This function applies a notch filter to time series data to remove specific frequencies. 

    Inputs:
        signal (np array)
        fs (float): frames per second sampling rate
        f0 (float): freq to be removed
        Q (float): quality factor, (a higher Q means a narrower notch)
    Outputs:
        filt (np array): filtered data
    
    """

    from scipy.signal import iirnotch, filtfilt

    # Design the notch filter
    b, a = iirnotch(f0, Q, fs)
    # Assuming `signal` is your input signal
    filt = filtfilt(b, a, signal)
    return(filt)



#====================
def filt_loop(dic):
#===================
    signal=dic['data']
    out = np.zeros(signal.shape)
    assert signal.shape[1] == 19, 'Incorrect number of channels'
    for s in range(signal.shape[1]):
        lpfilt = lowpassfilt(signal[:,s], lp_cutoff, fs, N=4)
        notchfilt = notchfilt(lpfilt, fs, f0, Q)
        out[:,s] = notchfilt
    dic['filt'] = out
    return(dic)

