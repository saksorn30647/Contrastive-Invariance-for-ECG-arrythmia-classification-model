"""STEP 1b of 4 -- extract INCART beats.

DATA DOWNLOAD (run once, before this script):
    python -c "import wfdb; wfdb.dl_database('incartdb', 'data/incart')"
  or download https://physionet.org/content/incartdb/1.0.0/ into data/incart/

CONSUMES : data/incart/*.dat|.hea|.atr   (75 records)
PRODUCES : data/X_incart.npy   (175811, 256) float, per-beat z-scored
           data/y_incart.npy   (175811,)     AAMI labels as CHARACTERS
           data/records.npy    (175811,)     INCART record id per beat -- IN data/

*** data/records.npy, NOT records.npy. ***
The root records.npy holds MIT-BIH record ids (109,451 beats). This script used
to write to that path, which overwrote the MIT-BIH array and silently broke
every MIT-BIH split. Fixed below; keep the two datasets on separate paths.

PIPELINE : beat_extrac_label.py -> beat_extrac_label_INCART.py
           -> CWT_preprocess.py -> CWT_preprocess_incart.py

Signal chain: lead II (channel 1, verified II in all 75 records), resampled
257 -> 360 Hz to match MIT-BIH (annotation indices rescaled by 360/257),
bandpass 4-40 Hz, 256-sample R-centred window, per-beat z-score.

Expected: 175,811 beats / 75 records. INCART contains ZERO paced beats, so the
Q class there is 6 unclassifiable beats only.
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import resample

def resample_ecg(signal, original_fs=257, target_fs=360):

    target_length = int(len(signal) * target_fs / original_fs)

    resampled_signal = resample(signal, target_length)

    return resampled_signal

def bandpass_filter(signal, lowcut=4, highcut=40, fs=360, order=4):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')

    filtered = filtfilt(b, a, signal)

    return filtered
NASYMBOLS = []
VALID_SYMBOLS = [
    "N","L","R","e","j",
    "A","a","J","S",
    "V","E",
    "F",
    "/","f","Q"
]

AAMI_MAP = {
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    "V": "V",
    "E": "V",

    "F": "F",
      
    "/": "Q",
    "f": "Q",
    "Q": "Q"
}

def annotators(symbol):
    
    return AAMI_MAP.get(symbol)

def zscore_normalize(signal):

    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
       return signal
    return (signal - mean) / std

X = []
y = []

records = [
    "I01", "I02", "I03", "I04", "I05", "I06", "I07", "I08", "I09", "I10",
    "I11", "I12", "I13", "I14", "I15", "I16", "I17", "I18", "I19", "I20",
    "I21", "I22", "I23", "I24", "I25", "I26", "I27", "I28", "I29", "I30",
    "I31", "I32", "I33", "I34", "I35", "I36", "I37", "I38", "I39", "I40",
    "I41", "I42", "I43", "I44", "I45", "I46", "I47", "I48", "I49", "I50",
    "I51", "I52", "I53", "I54", "I55", "I56", "I57", "I58", "I59", "I60",
    "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69", "I70",
    "I71", "I72", "I73", "I74", "I75"
]   
record_list = []
beat_index_list = []
window_size = 256
half_window = window_size // 2

# lead_index = wfdb.rdrecord("incart/files/I01").sig_name.index("II")
# print(f"Lead II index: {lead_index}")

for record_name in records:

    record = wfdb.rdrecord(f"data/incart/files/{record_name}") #record100 data
    annotation = wfdb.rdann(f"data/incart/files/{record_name}", "atr") #record100 annotation data

    ecg = record.p_signal[:,1] #Lead II is the second channel in INCART dataset, so we use index 1 to access it. The first channel (index 0) is usually Lead I.
    ecg = resample_ecg(ecg, original_fs=257, target_fs=360) # Resample to 360 Hz
    ecg = bandpass_filter(ecg)
    scale_factor = 360 / 257

    
    for idx in range(len(annotation.sample)):
    
        r_peak = round(annotation.sample[idx] * scale_factor)
        symbol = annotation.symbol[idx]
        if symbol not in VALID_SYMBOLS:
           NASYMBOLS.append(symbol)
           continue
        label = annotators(symbol)

        # skip invalid boundary beats
        if r_peak-half_window < 0:
            continue

        if r_peak+half_window > len(ecg):
            continue

        beat = ecg[r_peak-half_window : r_peak+half_window]
        beat = zscore_normalize(beat)
        X.append(beat)
        y.append(label)
        record_list.append(record_name)
        beat_index_list.append(idx)

# WAS np.save("records.npy", ...) -- that is the MIT-BIH record array's path,
# so running this script would OVERWRITE MIT-BIH record ids with INCART ones
# and silently break every MIT-BIH split. INCART belongs in data/.
np.save("data/records.npy", record_list)
np.save("data/beat_indices.npy", beat_index_list)


print("Total beats:", len(X))
print("Total labels:", len(y))
from collections import Counter
print(Counter(y))
# print(f"unknown: {Counter(NASYMBOLS)}")


def plot_sample_beats(X, y, shift=0):
#Sample beat visualization
    n, q, f, s, v = 0, 0, 0, 0, 0
    for i, label in enumerate(y):
    
        if label == "N" and n == 0:
            plt.subplot(2,3,1)
            plt.plot(X[i+shift])
            plt.title("N beat")
            n += 1
        
        elif label == "Q" and q == 0:
            plt.subplot(2,3,2)
            plt.plot(X[i+shift])
            plt.title("Q beat")
            q   += 1
        
        elif label == "F" and f == 0:
            plt.subplot(2,3,3)
            plt.plot(X[i+shift])
            plt.title("F beat")
            f += 1
        
        elif label == "S" and s == 0:
            plt.subplot(2,3,4)
            plt.plot(X[i+shift])
            plt.title("S beat")
            s += 1
        
        elif label == "V" and v == 0:
            plt.subplot(2,3,5)
            plt.plot(X[i+shift])
            plt.title("V beat")
        
            v += 1
        elif label == "S" and s == 1:
            plt.subplot(2,3,6)
            plt.plot(X[i+shift])
            plt.title("S beat2")
            s += 1
        if n > 0 and q > 0 and f > 0 and s > 1 and v > 0:
            plt.show()
            break

def plot_n_beats(X, y, num_samples=5, num_cols=5, num_rows=1,label="N", shift=0):
    count = 0
    plt.figure(figsize=(15, 3))
    for i, beat_label in enumerate(y):
        if beat_label == label:
            plt.subplot(num_rows, num_cols, count + 1)
            plt.plot(X[i+shift])
            plt.title(f"{label} beat {count + 1}")
            # if count == 8 or count == 9:
            #     print(i, beat_label)
            count += 1
            if count >= num_samples:
                break
    plt.tight_layout()
    plt.show()

# plot_n_beats(X, y, num_samples=10, num_cols=5, num_rows=2,label="S", shift=1)
# plot_sample_beats(X, y)

X = np.array(X)
y = np.array(y)
print(X.shape)
print(y.shape)
np.save("data/X_incart.npy", X)
np.save("data/y_incart.npy", y)

LABEL_MAP = {
    "N":0,
    "S":1,
    "V":2,
    "F":3,
    "Q":4
}

# y_numeric = []
# for i in range(len(y)):
#     y_numeric.append(LABEL_MAP[y[i]])
# y_numeric = np.array(y_numeric)
# np.save("y_incart_numeric.npy", y_numeric)
# print(y_numeric[:100])
# print("All Data saved ")