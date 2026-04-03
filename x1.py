import scipy.io # for loading the .mat file into python
from scipy.signal import butter, filtfilt, find_peaks, detrend, sosfiltfilt 
import matplotlib.pyplot as plt
import numpy as np

data = scipy.io.loadmat('X_1.mat') # python dictionary
# print(data.keys()) # dict_keys(['__header__', '__version__', '__globals__', 'X_1'])

signal = data['X_1'] # nested array
# print(type(signal))
# print(signal)
biosignal = signal[0, 0]
# print(biosignal.shape) # (120000, 4) -> 120000 samples and 4 channels

# 120000 samples in  4 ??? minutes -> 200Hz
fs = 500 
nyquist = fs / 2

# RAW SIGNALS PLOT
time = np.arange(len(biosignal)) / fs # Time in seconds
labels = ['ECG', 'EMG', 'RESP', 'EDA']
colors = ['b', 'r', 'y', 'm']
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

for i in range(4):
    axs[i].plot(time[0:30000], biosignal[0:30000, i], color=colors[i])
    axs[i].set_title(f"Raw {labels[i]}")
axs[3].set_xlabel('Seconds')
plt.tight_layout()
plt.show()


#### PREPROCESSING ####
### ECG ###
lowcut = 0.5
highcut = 40
low = lowcut / nyquist
high = highcut / nyquist
b, a = butter(4, [low, high], btype='band')
ecg = filtfilt(b, a, biosignal[:, 0])

### EMG ###
emg_high = 20 / nyquist
emg_b, emg_a = butter(4, emg_high, btype='high')
emg = filtfilt(emg_b, emg_a, biosignal[:, 1])
# rectification
emg = np.abs(emg)
# envelope extraction (low-pass ~5 Hz)
b, a = butter(4, 5 / nyquist, btype='low')
emg = filtfilt(b, a, emg)


### RESPIRATION ###
resp_raw = biosignal[:, 2]
# detrend
resp = detrend(resp_raw)
# remove very low frequencies (~0.2 Hz) + low pass filter
sos = butter(4, [0.05/nyquist, 0.1/nyquist], btype='band', output='sos')
resp = sosfiltfilt(sos, resp)

### EDA ###
eda_low = 0.5 / nyquist
eda_b, eda_a = butter(4, eda_low, btype='low')
eda = filtfilt(eda_b, eda_a, biosignal[:, 3])

# PREPROCESSED SIGNALS PLOT
signals = [ecg, emg, resp, eda]
labels = ['ECG', 'EMG', 'RESP', 'EDA']
colors = ['b', 'r', 'g', 'm']

fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

for i in range(4):
    axs[i].plot(time[:30000], signals[i][:30000], color=colors[i])
    axs[i].set_title(labels[i])
    axs[i].grid(True)
axs[3].set_xlabel("Time (sec)")
plt.tight_layout()
plt.show()


#### FEATURES #####
### ECG ###
# simple version  R-PEAK
peaks, _ = find_peaks(ecg, distance=0.6*fs, height=np.mean(ecg))
r_peaks = peaks
# HRV 
# RR 
rr_intervals = np.diff(r_peaks) / fs
hrv_features = { "mean_hr": 60 / np.mean(rr_intervals),
    "sdnn": np.std(rr_intervals),
    "rmssd": np.sqrt(np.mean(np.diff(rr_intervals)**2)),
    "min_rr": np.min(rr_intervals),
    "max_rr": np.max(rr_intervals)}


### EMG ###
emg_rms = np.sqrt(np.mean(emg**2))
emg_std = np.std(emg)
emg_max = np.max(emg)
emg_features = {"emg_rms": emg_rms,
    "emg_std": emg_std,
    "emg_max": emg_max}


### RESP ###
resp_peaks, _ = find_peaks(resp, distance=fs*2)
breaths = len(resp_peaks) / (len(resp)/fs)
resp_features = { "resp_rate": breaths * 60,
    "resp_std": np.std(resp),
    "resp_mean": np.mean(resp),
    "resp_amplitude": np.max(resp) - np.min(resp)}


### EDA ###
eda_diff = np.diff(eda)
eda_features = { "eda_mean": np.mean(eda),
    "eda_std": np.std(eda),
    "eda_max": np.max(eda),
    "eda_min": np.min(eda),
    "eda_slope": np.mean(eda_diff)}
