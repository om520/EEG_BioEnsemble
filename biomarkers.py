import numpy as np
from scipy import stats, signal

def get_temporal_baseline(all_windows, seizure_start_idx, fs_windows=0.5):
    """
    Get baseline from 5-10 minutes BEFORE the detected seizure.
    This is INDEPENDENT of the model's predictions — eliminates circular validation.
    """
    baseline_end   = int(seizure_start_idx - (5 * 60 * fs_windows))   # 5 min before
    baseline_start = int(seizure_start_idx - (10 * 60 * fs_windows))  # 10 min before
    baseline_start = max(0, baseline_start)
    baseline_end   = max(0, baseline_end)

    if baseline_end - baseline_start < 30:  # Need at least 30 windows
        baseline_start = 0
        baseline_end   = min(len(all_windows), int(10 * 60 * fs_windows))

    # If still not enough windows (e.g. seizure at the very start)
    if baseline_end - baseline_start <= 0:
        return all_windows[0:min(len(all_windows), 30)]

    return all_windows[baseline_start:baseline_end]


def extract_biomarkers_per_channel(windows, fs=256):
    """
    Extract the 7 retained biomarkers PER CHANNEL, PER WINDOW.
    windows: shape (N_windows, 18_channels, 1024_samples)
    Returns: dict of biomarker_name -> array shape (N_windows, 18_channels)
    """
    if len(windows) == 0:
        return {}
        
    N, C, T = windows.shape
    bio = {}

    # --- Time-domain ---
    bio['line_length'] = np.mean(np.abs(np.diff(windows, axis=2)), axis=2)   # (N, 18)
    bio['rms']         = np.sqrt(np.mean(windows ** 2, axis=2))              # (N, 18)
    bio['kurtosis']    = stats.kurtosis(windows, axis=2)                     # (N, 18)

    # Hjorth Complexity
    dx  = np.diff(windows, axis=2)
    ddx = np.diff(dx, axis=2)
    var_x   = np.var(windows, axis=2) + 1e-12
    var_dx  = np.var(dx, axis=2) + 1e-12
    var_ddx = np.var(ddx, axis=2) + 1e-12
    mobility_x  = np.sqrt(var_dx / var_x)
    mobility_dx = np.sqrt(var_ddx / var_dx)
    bio['hjorth_complexity'] = mobility_dx / (mobility_x + 1e-12)            # (N, 18)

    # --- Frequency-domain (Welch PSD per channel per window) ---
    flat = windows.reshape(N * C, T)
    freqs, psd = signal.welch(flat, fs=fs, nperseg=256, axis=1)
    total_power = np.sum(psd, axis=1, keepdims=True) + 1e-12

    theta_delta = np.sum(psd[:, (freqs >= 0.5) & (freqs < 8)], axis=1) / total_power.squeeze()
    high_gamma  = np.sum(psd[:, (freqs >= 25) & (freqs < 45)], axis=1) / total_power.squeeze()
    bio['rel_theta_delta_power'] = theta_delta.reshape(N, C)                 # (N, 18)
    bio['rel_high_gamma_power']  = high_gamma.reshape(N, C)                  # (N, 18)

    psd_norm = psd / total_power
    spec_ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-12), axis=1) / np.log2(psd.shape[1])
    bio['spectral_entropy'] = spec_ent.reshape(N, C)                         # (N, 18)

    # Fast proxy for sample entropy using permutation entropy
    def _perm_entropy(x, order=3, delay=1):
        from itertools import permutations
        n = len(x)
        perms = np.array(list(permutations(range(order))))
        counts = np.zeros(len(perms))
        for i in range(n - (order - 1) * delay):
            motif = x[i:i + order * delay:delay]
            idx = np.argsort(motif)
            for j, p in enumerate(perms):
                if np.array_equal(idx, p):
                    counts[j] += 1
                    break
        counts = counts[counts > 0]
        probs = counts / counts.sum()
        return -np.sum(probs * np.log2(probs))

    samp_ent = np.zeros((N, C))
    for i in range(N):
        for j in range(C):
            samp_ent[i, j] = _perm_entropy(windows[i, j, ::4])
    bio['sample_entropy'] = samp_ent                                          # (N, 18)

    return bio

def compute_contrast_per_channel(seizure_bio, baseline_bio, n_seizure_windows, overlap_fraction=0.5):
    contrasts = {}
    n_eff = max(1, int(n_seizure_windows * (1 - overlap_fraction)))
    z_threshold = 2.0 * np.sqrt(n_seizure_windows / max(n_eff, 1))

    for name in seizure_bio:
        seiz_vals = seizure_bio[name]    
        base_vals = baseline_bio[name]   

        base_mean = np.mean(base_vals, axis=0)   
        base_std  = np.std(base_vals, axis=0)     
        seiz_mean = np.mean(seiz_vals, axis=0)    
        seiz_std  = np.std(seiz_vals, axis=0)     

        z_scores = np.where(base_std > 1e-10, (seiz_mean - base_mean) / base_std, 0.0)
        pooled = np.sqrt((base_std**2 + seiz_std**2) / 2) + 1e-12
        cohens_d = (seiz_mean - base_mean) / pooled  

        significant = np.abs(z_scores) > z_threshold  

        reliable = n_seizure_windows >= 5
        if n_seizure_windows < 30 and n_seizure_windows >= 5:
            # Bootstrap
            boot_means = np.array([
                np.mean(seiz_vals[np.random.choice(n_seizure_windows, n_seizure_windows, replace=True)], axis=0)
                for _ in range(100) # 100 for speed
            ])
            ci_low  = np.percentile(boot_means, 2.5, axis=0)
            ci_high = np.percentile(boot_means, 97.5, axis=0)
        else:
            ci_low = ci_high = None

        contrasts[name] = {
            'z_scores': z_scores,
            'cohens_d': cohens_d,
            'significant_channels': significant,
            'n_significant': int(np.sum(significant)),
            'reliable': reliable,
            'ci_low': ci_low,
            'ci_high': ci_high
        }

    return contrasts
