import statistics
import pandas as pd
import numpy as np
from EntropyHub import FuzzEn

# returns data chunked into size n
def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# reutns one three dimensionsal entropy vector per chunk, excluding ps1 and ps2
def get_entropy_from_chunks(_chunks, _chanSD):
    entropy_of_chunks = []
    for chunk in _chunks: 
        chunk_entropy, Ps1, Ps2 = FuzzEn(chunk, m=3, tau=1, Fx = 'Gudermannian', r=(_chanSD), Logx = np.exp(1))
        entropy_of_chunks.append(chunk_entropy)
    return entropy_of_chunks

# returns a scalar of the third dimension for each three dimensional fuzzy entropy vector
def get_FuzzEn_third_dimension(_FuzzEn_m3):
    _FuzzEn_third_dimension = []
    for dimension in _FuzzEn_m3:
        third_dimension = dimension[2]
        _FuzzEn_third_dimension.append(third_dimension)
    return _FuzzEn_third_dimension

# takes a timeseries eeg dataframe and returns the last five minutes downsampled to 250hz  
def downsample_and_trim(_df):
    _df = _df.tail(300000)
    _df = _df.iloc[::2, :]
    _df = _df.iloc[::2, :]
    return _df

# returns df with selected channels only
def get_channels(_df, _chan1, _chan2):
    _df_cut = _df[[_chan1,_chan2]]
    return _df_cut


# takes a df of raw EEG and tranforms it into a df of Fuzzy Entropy values.
def df_raw_to_entropy(_df,_chan1,_chan2):

    df_chan1 = _df.loc[:,_chan1]
    df_chan2 = _df.loc[:,_chan2]

    chan1_arr = df_chan1.to_numpy()
    chan2_arr = df_chan2.to_numpy()

    chan1_sd = statistics.stdev(chan1_arr)*.2
    chan2_sd = statistics.stdev(chan2_arr)*.2

    chan1_arr_chunked = list(chunk(chan1_arr,2500))
    chan2_arr_chunked = list(chunk(chan2_arr,2500))

    chan1_FuzEn_3d = get_entropy_from_chunks(chan1_arr_chunked,chan1_sd)
    chan2_FuzEn_3d = get_entropy_from_chunks(chan2_arr_chunked,chan2_sd)

    chan1_FuzEn = get_FuzzEn_third_dimension(chan1_FuzEn_3d)
    chan2_FuzEn = get_FuzzEn_third_dimension(chan2_FuzEn_3d)

    _df_FuzEn = pd.DataFrame({_chan1: chan1_FuzEn, _chan2: chan2_FuzEn})
    return _df_FuzEn