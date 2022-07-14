import mne
import datashaper
import warnings
import pandas as pd
import numpy as np
from alphawaves.dataset import AlphaWaves
from scipy.stats import zscore
from scipy.signal import welch
warnings.filterwarnings("ignore")

def calc_theta_alpha_bandpower(_data):
    n_channels = 16
    fs = 256
    # Define EEG bands
    eeg_band_fft = np.zeros([2, n_channels])
    eeg_bands = {'Theta': (4, 8), 'Alpha': (8, 12)}
    for channel in range(n_channels):
        data = np.transpose(_data[channel,:])
        # Get real amplitudes of FFT (only in postive frequencies)
        fft_vals = np.absolute(np.fft.rfft(data))
        # Get frequencies for amplitudes in Hz
        fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)
        # Take the mean of the fft amplitude for each EEG band
        for band in eeg_bands:  
            # print(np.where((fft_freq >= eeg_bands[band][0])))
            freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                            (fft_freq <= eeg_bands[band][1]))[0]
            if band == 'Theta':
                eeg_band_fft[0,channel] = np.mean(fft_vals[freq_ix])
            else:
                eeg_band_fft[1,channel] = np.mean(fft_vals[freq_ix])
    return eeg_band_fft

# AlphaWaves DATASET AND INITIAL DATA HANDLING CODE ADAPTED FROM THE ALPHA WAVES GITHUB:
# https://github.com/plcrodrigues/py.ALPHA.EEG.2017-GIPSA
# specifically the file example_classification.py

# define the dataset instance
dataset = AlphaWaves(useMontagePosition = False) # use useMontagePosition = False with recent mne versions

# #initialise and set variables
X = np.array([])
Xt = np.array([])
Xa = np.array([])
Xtdiva = np.array([])
participant_n = 19
channel_n = 16

# get the data from subject of interest. range(20) gets 0 to 19.
for p in range(participant_n):
    subject = dataset.subject_list[p]
    raw = dataset._get_single_subject_data(subject) #<RawArray  |  None, n_channels x n_times : 17 x 119808

    # filter data and resample
    fmin = 4
    fmax = 8
    raw.filter(fmin, fmax, verbose=False)
    # raw.resample(sfreq=128, verbose=False)

    # detect the events and cut the signal into epochs
    events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
    event_id = {'closed': 1, 'open': 2}
    epochs = mne.Epochs(raw, events, event_id, tmin=2.0, tmax=8.0, baseline=None,
                        verbose=False, preload=True)
    epochs.pick_types(eeg=True)

    # get trials and labels
    #y = events[:, -1] # this labels eyes open or eyes closed condition. don't need as we are taking both.
    X_w = epochs.get_data()
    #initialise array, shape = (n features, n channels)
    X_w_entropy = np.zeros((X_w.shape[0], channel_n)) 
    X_w_bp_t = np.zeros((X_w.shape[0], channel_n)) 
    X_w_bp_a = np.zeros((X_w.shape[0], channel_n)) 
    X_w_bp_t_div_a = np.zeros((X_w.shape[0], channel_n)) 
    #loop through each FEATURE of the data
    for x, element in enumerate(X_w):
        chunk = X_w[x]
        #z-score normalise and clamp at 3 standard deviations to account for outliers
        chunk_z = zscore(chunk)
        chunk_z = np.clip(chunk_z, -3, 3) 
        #rescale values between -1 and 1
        chunk_n = 2*(chunk - chunk.min())/ (chunk.max() - chunk.min()) - 1
        #bandpower calcs
        bp = calc_theta_alpha_bandpower(chunk_n)
        X_w_bp_t[x] = bp[0,:]
        X_w_bp_a[x] = bp[1,:]
        X_w_bp_t_div_a[x] = bp[0,:] / bp[1,:]
        #calculate fuzzy entropy for each channel
        #X_w_entropy[x] = datashaper.df_norm_to_entropy(chunk_n, channel_n)
    Xt = np.vstack([Xt,X_w_bp_t]) if Xt.size else X_w_bp_t
    Xa = np.vstack([Xa,X_w_bp_a]) if Xa.size else X_w_bp_a
    Xtdiva = np.vstack([Xtdiva,X_w_bp_t_div_a]) if Xtdiva.size else X_w_bp_t_div_a
#     #concatenate participants into a big array
#     X = np.vstack([X,X_w_entropy]) if X.size else X_w_entropy
#     print('finished calculating entropy values for participant #', (p+1))
print(Xt.shape)
# save the features to be loaded and classified in using_alpha_waves_classify.py
dft = pd.DataFrame(Xt)
dfa = pd.DataFrame(Xa)
dftdiva = pd.DataFrame(Xtdiva)
dft.to_pickle("bandpower_theta.pkl")
dfa.to_pickle("bandpower_alpha.pkl")
dftdiva.to_pickle("bandpower_theta_div_alpha.pkl")

# df = pd.DataFrame(X)
# df.to_pickle("FuzEn_4-8Hz.pkl")

