import os
import urllib.request
import mne
import numpy as np

PHYSIONET_BASE_URL = "https://physionet.org/files/chbmit/1.0.0/"

CANONICAL_BIPOLAR_18 = [
    ("FP1", "F7"), ("F7", "T7"), ("T7", "P7"), ("P7", "O1"),  # Left Temporal
    ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),  # Left Parasagittal
    ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),  # Right Parasagittal
    ("FP2", "F8"), ("F8", "T8"), ("T8", "P8"), ("P8", "O2"),  # Right Temporal
    ("FZ", "CZ"),  ("CZ", "PZ")                               # Midline
]

def download_edf(patient_id, file_name, download_dir="data"):
    os.makedirs(download_dir, exist_ok=True)
    file_path = os.path.join(download_dir, file_name)
    if not os.path.exists(file_path):
        url = f"{PHYSIONET_BASE_URL}{patient_id}/{file_name}"
        print(f"Downloading {url} to {file_path}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            return None
    else:
        print(f"File {file_path} already exists. Skipping download.")
    return file_path

def preprocess_edf(file_path, window_sec=4, overlap_sec=2, target_fs=256):
    print(f"Loading {file_path}...")
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    except Exception as e:
        print(f"Error reading EDF: {e}")
        return None, None, None
    
    # Standardize channel names (strip 'EEG ', remove '-REF' etc.)
    rename_dict = {}
    for ch in raw.ch_names:
        clean = ch.replace('EEG ', '').replace('-REF', '').replace('-LE', '').strip().upper()
        rename_dict[ch] = clean
    raw.rename_channels(rename_dict)
    
    # Filtering
    print("Applying filters (0.5-45 Hz bandpass, 60 Hz notch)...")
    raw.filter(l_freq=0.5, h_freq=45.0, fir_design='firwin', verbose=False)
    raw.notch_filter(freqs=60.0, verbose=False)
    
    # Create bipolar channels
    print("Creating 18 canonical bipolar channels...")
    bipolar_data = []
    ch_names = []
    
    for anode, cathode in CANONICAL_BIPOLAR_18:
        target_bipolar = f"{anode}-{cathode}"
        # 1. Check if the bipolar channel already exists in the file (CHB-MIT standard)
        if target_bipolar in raw.ch_names:
            idx = raw.ch_names.index(target_bipolar)
            data, _ = raw[idx, :]
            bipolar_data.append(data[0])
            ch_names.append(target_bipolar)
        # 2. Otherwise, check if referential channels exist to subtract them
        elif anode in raw.ch_names and cathode in raw.ch_names:
            idx_a = raw.ch_names.index(anode)
            idx_c = raw.ch_names.index(cathode)
            data_a, _ = raw[idx_a, :]
            data_c, _ = raw[idx_c, :]
            bipolar_data.append(data_a[0] - data_c[0])
            ch_names.append(target_bipolar)
        # 3. If neither exists, fill with zeros
        else:
            print(f"Warning: Could not find {target_bipolar} or its referential components. Filling with zeros.")
            bipolar_data.append(np.zeros(raw.n_times))
            ch_names.append(target_bipolar)
            
    bipolar_array = np.array(bipolar_data) # (18, N_samples)
    
    # Z-score normalize per channel
    means = np.mean(bipolar_array, axis=1, keepdims=True)
    stds = np.std(bipolar_array, axis=1, keepdims=True) + 1e-8
    bipolar_array = (bipolar_array - means) / stds
    
    # Windowing
    print(f"Slicing into {window_sec}s windows with {overlap_sec}s overlap...")
    fs = int(raw.info['sfreq'])
    window_samples = window_sec * fs
    step_samples = (window_sec - overlap_sec) * fs
    
    windows = []
    for start in range(0, bipolar_array.shape[1] - window_samples + 1, step_samples):
        windows.append(bipolar_array[:, start:start+window_samples])
        
    windows_array = np.array(windows) # (N_windows, 18, 1024)
    print(f"Generated {windows_array.shape[0]} windows of shape {windows_array.shape[1:]}")
    
    return windows_array, ch_names, fs
