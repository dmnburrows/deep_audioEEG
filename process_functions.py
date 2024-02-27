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
def set_load(path, id, sex, age, group, mmse):
#===================
    """ 
    This function loads .set EEG files and processes them into python dictionaries with included metadata.

    Inputs:
        path (str): path to .set file
        id (str): sample id
        sex (str): sample sex
        group (str): condition, A = alzheimer, F = FTD, C = control
        mmse (float): Mini-Mental State Examination  

    Outputs:
        dic (dictionary): dict containing ID name and time series npy array.
    
    """
    import mne
    raw = mne.io.read_raw_eeglab(path, preload=True)
    dic = {}
    dic['id'] = id
    dic['sex'] = sex
    dic['age'] = age
    dic['group'] = group
    dic['mmse'] = mmse
    dic['data'] = raw._data
    dic['ch_names'] = raw.ch_names
    return(dic)

def PSD(signal, fs):
    """
    This function calculates the power spectral density (PSD) of the given signal by performing
    a Fast Fourier Transform (FFT) and returns the frequency bins and their corresponding power values.
    The PSD is computed independently for each channel in the signal.

    Inputs:
        signal (np.ndarray): Input array containing the signal data, 
                             structured as channels (rows) x samples (columns).

    Outputs:
        psd (np.ndarray): The power spectral density of the signal for each channel. 
                          This array has the same number of rows as `signal` and half the number of columns,
                          since PSD is symmetric and only positive frequencies are returned.
        frequency (np.ndarray): An array of frequency bins corresponding to the PSD values,
                                ranging from 0 to the Nyquist frequency (half the sampling rate).

    Note:
        This function assumes the sampling frequency `fs` is defined outside its scope. If `fs` is not defined,
        include it as a parameter to the function and pass it accordingly.

    Example:
        fs = 256  # Sampling frequency in Hz
        psd, freq = PSD(eeg_signal)
        plt.plot(freq, psd[0, :])  # Plot PSD for the first channel
    """

    from scipy.fft import fft
    import numpy as np

    # Perform FFT and get the magnitude (absolute value) of the FFT for each channel
    fft_result = fft(signal, axis=1)
    fft_magnitude = np.abs(fft_result)

    # Calculate power spectral density (PSD) as the square of the magnitude of the FFT components
    psd = fft_magnitude ** 2 / signal.shape[1]

    # Frequency axis: Create a frequency array from 0 to the Nyquist frequency
    n = signal.shape[1]
    frequency = np.linspace(0.0, fs / 2, n // 2)

    # Return only the first half of the PSD (positive frequencies) and the frequency array
    return (psd[:, :n // 2], frequency)


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

#===================
def highpassfilt(signal, cutoff, fs, N=4):
#===================

    """
    This function applies a high-pass filter to time series data.

    Inputs:
        signal (np.array): The input signal to be filtered.
        cutoff (float): Cutoff frequency for the high-pass filter, all frequencies lower will be attenuated.
        fs (float): Sampling rate of the input signal, in Hz.
        N (int): Order of the Butterworth filter.

    Outputs:
        filt (np.array): The high-pass filtered data.
    """

    from scipy.signal import butter, filtfilt

    # Normalize the cutoff frequency with respect to the Nyquist frequency
    nyq = 0.5 * fs
    norm_cutoff = cutoff / nyq

    # Design the Butterworth high-pass filter (Nth order, where N=4 is a common choice)
    b, a = butter(N, norm_cutoff, btype='high', analog=False)
    
    # Apply the filter to the signal using filtfilt for zero-phase filtering
    filt = filtfilt(b, a, signal)
    
    return filt


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


def split_array_into_bins(array, m):
    """
    Split an array into n bins, each bin having a length as close as possible to m.
    
    Parameters:
        array (np.array): The input array to split.-> Must be channelsxsamples
        m (int): The desired approximate size for each bin.
    
    Returns:
        list of np.array: A list containing the resulting bins.
    """
    total_length = array.shape[1]
    n = int(np.ceil(total_length / m))  # Calculate the number of bins
    
    bins = []
    for i in range(n):
        start_index = i * m
        # Ensure the last bin gathers all remaining elements
        end_index = min(start_index + m, total_length)
        bins.append(array[:,start_index:end_index])
    
    return bins


def zeromean_unitvar(array):
    mean = np.mean(array)
    std = np.std(array)
    # Standardize the array
    return((array - mean) / std)



def split_array(arr, fractions, seed=3):
    """
    Split an array into multiple parts by fractions.
    Fractions should be a list of values that sum to 1.
    """
    assert sum(fractions) == 1, "Fractions must sum to 1"
    np.random.seed(seed)
    # Shuffle indices
    indices = np.arange(arr.shape[0])
    np.random.shuffle(indices, )
    
    # Calculate split sizes
    n = arr.shape[0]
    split_sizes = [int(f * n) for f in fractions[:-1]]  # Exclude last fraction
    split_sizes.append(n - sum(split_sizes))  # Ensure the last split takes the remainder
    
    # Split indices by calculated sizes
    split_indices = np.split(indices, np.cumsum(split_sizes)[:-1])
    
    # Return split arrays
    return [arr[indices] for indices in split_indices]