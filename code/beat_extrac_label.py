"""STEP 1a of 4 -- extract MIT-BIH beats.

DATA DOWNLOAD (run once, before this script):
    python -c "import wfdb; wfdb.dl_database('mitdb', 'data/mit-bih')"
  or download https://physionet.org/content/mitdb/1.0.0/ into data/mit-bih/

CONSUMES : data/mit-bih/*.dat|.hea|.atr   (48 records)
PRODUCES : data/X.npy      (109451, 256) float, per-beat z-scored
           data/y.npy      (109451,)     AAMI labels as CHARACTERS 'N','S','V','F','Q'
           records.npy     (109451,)     MIT-BIH record id per beat -- AT THE ROOT

PIPELINE : beat_extrac_label.py -> beat_extrac_label_INCART.py
           -> CWT_preprocess.py -> CWT_preprocess_incart.py

Signal chain: channel 0, bandpass 4-40 Hz (Butterworth order 4), 256-sample
window centred on the annotated R peak, per-beat z-score. Beats whose window
would cross a record boundary are dropped.

NOTE on lead: channel 0 is MLII for 45 of 48 records. It is V5 for records 102,
104 (which have no MLII at all) and 114 (whose MLII is channel 1).

Expected class counts: N 90,593  S 2,781  V 7,235  F 802  Q 8,040  = 109,451.
"""

import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from collections import Counter

def bandpass_filter(signal, lowcut=4, highcut=40, fs=360, order=4):

    nyquist = 0.5 * fs

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')

    filtered = filtfilt(b, a, signal)

    return filtered

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
    "100", "101", "102", "103", "104","105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115","116", "117", "118", "119","121", "122", "123", "124", 
    "200","201", "202", "203", "205","207", "208", "209", "210","212", "213", "214", "215",
    "217", "219", "220", "221","222", "223", "228", "230","231", "232", "233","234"
]   
train_records = [101,106,108,109,112,114,115,116,118,119,
122,124,201,203,205,207,208,209,215,220,
223,230]
test_records = [100,103,105,111,113,117,121,123,200,202,
210,212,213,214,219,221,222,228,231,232,
233,234]
record_list = []
window_size = 256
half_window = window_size // 2

for record_name in records:
    record_y = []
    record = wfdb.rdrecord(f"data/mit-bih/{record_name}")
    annotation = wfdb.rdann(f"data/mit-bih/{record_name}", "atr")

    ecg = record.p_signal[:,0]
    ecg = bandpass_filter(ecg)

    for idx in range(len(annotation.sample)):
        
        r_peak = annotation.sample[idx]
        symbol = annotation.symbol[idx]
        if symbol not in VALID_SYMBOLS:
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
        record_y.append(label)
        record_list.append(record_name)
    print(f"{record_name} : {Counter(record_y)}")


print("Total beats:", len(X))
print("Total labels:", len(y))
print(Counter(y))
# print("Unique labels:", set(y))


def plot_sample_beats(X, y):
#Sample beat visualization
    for i, label in enumerate(y):
    
        if label == "N" and n == 0:
            plt.subplot(2,3,1)
            plt.plot(X[i])
            plt.title("N beat")
            n += 1 # Now it stays 1 and won't plot 'N' again
        
        elif label == "Q" and q == 0:
            plt.subplot(2,3,2)
            plt.plot(X[i])
            plt.title("Q beat")
            q   += 1
        
        elif label == "F" and f == 0:
            plt.subplot(2,3,3)
            plt.plot(X[i])
            plt.title("F beat")
            f += 1
        
        elif label == "S" and s == 0:
            plt.subplot(2,3,4)
            plt.plot(X[i])
            plt.title("S beat")
            s += 1
        
        elif label == "V" and v == 0:
            plt.subplot(2,3,5)
            plt.plot(X[i])
            plt.title("V beat")
        
            v += 1
        elif label == "S" and s == 1:
            plt.subplot(2,3,6)
            plt.plot(X[i])
            plt.title("S beat2")
            s += 1
        if n > 0 and q > 0 and f > 0 and s > 1 and v > 0:
            plt.show()
            break

# plot_sample_beats(X, y)
def plot_n_beats(X, y, num_samples=5, num_cols=5, num_rows=1,label="N"):
    count = 0
    plt.figure(figsize=(15, 3))
    for i, beat_label in enumerate(y):
        if beat_label == label:
            plt.subplot(num_rows, num_cols, count + 1)
            plt.plot(X[i])
            plt.title(f"{label} beat {count + 1}")
            count += 1
            if count >= num_samples:
                break
    plt.tight_layout()
    plt.show()

# plot_n_beats(X, y, num_samples=10, num_cols=5, num_rows=2,label="Q")
X = np.array(X)
y = np.array(y)
records = np.array(record_list)
print(X.shape)
print(y.shape)
print(records.shape)
np.save("data/X.npy", X)
np.save("data/y.npy", y)
# MIT-BIH record ids go to the ROOT records.npy -- that is where every
# MIT-BIH training/analysis script reads them from. INCART record ids go to
# data/records.npy instead (see beat_extrac_label_INCART.py). Same filename,
# two different datasets: do not merge these two paths.
np.save("records.npy", records)


# LABEL_MAP = {
#     "N":0,
#     "S":1,
#     "V":2,
#     "F":3,
#     "Q":4
# }

# y_numeric = []
# for i in range(len(y)):
#     y_numeric.append(LABEL_MAP[y[i]])
# y_numeric = np.array(y_numeric)
# np.save("y_numeric.npy", y_numeric)
# print(y_numeric[:100])
# print("All Data saved ")

def plot_CWT(X,y):
    n, q, f, s, v = 0, 0, 0, 0, 0
    import matplotlib.pyplot as plt
    from utils.cwt import compute_cwt
    for i, label in enumerate(y):
        if label == "N" and n == 0:
            cwt_img = compute_cwt(X[i])
            plt.subplot(2,3,1)
            plt.imshow(
            cwt_img,
            aspect='auto',
            cmap='jet'
            )
            plt.title("N beat")
            plt.colorbar()
        
            n += 1 # Now it stays 1 and won't plot 'N' again
        
        elif label == "Q" and q == 0:
            cwt_img = compute_cwt(X[i])
            plt.subplot(2,3,2)
            plt.imshow(
            cwt_img,
            aspect='auto',
            cmap='jet'
            )
            plt.colorbar()
            plt.title("Q beat")
            q   += 1
        
        elif label == "F" and f == 0:
            cwt_img = compute_cwt(X[i])
            plt.subplot(2,3,3)
            plt.imshow(
            cwt_img,
            aspect='auto',
            cmap='jet'
            )
            plt.colorbar()
            plt.title("F beat")
            f += 1
        
        elif label == "S" and s == 0:
            cwt_img = compute_cwt(X[i])
            plt.subplot(2,3,4)
            plt.imshow(
            cwt_img,
            aspect='auto',
            cmap='jet'
            )
            plt.colorbar()
            plt.title("S beat")
            s += 1
        
        elif label == "V" and v == 0:
            cwt_img = compute_cwt(X[i])
            plt.subplot(2,3,5)
            plt.imshow(
            cwt_img,
            aspect='auto',
            cmap='jet'
            )
            plt.colorbar()
            plt.title("V beat")
        
            v += 1
        elif label == "S" and s == 1:
            cwt_img = compute_cwt(X[i])
            plt.subplot(2,3,6)
            plt.imshow(
            cwt_img,
            aspect='auto',
            cmap='jet'
            )
            plt.colorbar()
            plt.title("S beat2")
            s += 1
        if n > 0 and q > 0 and f > 0 and s > 1 and v > 0:
            plt.show()
            break

plot_CWT(X, y)    

    

