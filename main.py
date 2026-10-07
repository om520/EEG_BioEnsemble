import os
import torch
import numpy as np
from data_processor import download_edf, preprocess_edf
from model_cnn import CNN1DModel
from model_transformer import EEGTransformer
from model_unet import UNet1DModel
from biomarkers import get_temporal_baseline, extract_biomarkers_per_channel, compute_contrast_per_channel
from ensemble_connector import check_artifact_rejection, match_ictal_signatures, compute_confidence_index

def group_contiguous(indices, tol=2):
    if not indices: return []
    episodes = []
    start = indices[0]
    last = indices[0]
    for i in indices[1:]:
        if i - last <= tol:
            last = i
        else:
            episodes.append((start, last))
            start = i
            last = i
    episodes.append((start, last))
    return episodes

def print_banner(text):
    print("\n" + "="*70)
    print(f" {text}")
    print("="*70)

def main():
    print_banner("CHB-MIT EEG SEIZURE DETECTION PIPELINE (v2)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Models on {device} (using dummy initialized weights)...")
    cnn = CNN1DModel(in_channels=18).to(device)
    transformer = EEGTransformer(in_channels=18).to(device)
    unet = UNet1DModel(in_channels=18).to(device)
    
    cnn.eval(); transformer.eval(); unet.eval()

    # 2. Process a sample patient/file
    patient_id = "chb01"
    file_name = "chb01_03.edf" # Contains a seizure!
    
    print_banner(f"Processing Patient: {patient_id} | File: {file_name}")
    
    edf_path = download_edf(patient_id, file_name, download_dir="data")
    if not edf_path: return
    
    windows, ch_names, fs = preprocess_edf(edf_path)
    if windows is None: return
    
    print(f"\nChannel Montage ({len(ch_names)} channels):")
    for i, ch in enumerate(ch_names):
        print(f"[{i:02d}] {ch}", end="  ")
        if (i+1) % 4 == 0: print()
    print("\n")
    
    # 3. Model Inference
    print_banner("Running Ensemble Inference (Simulated)...")
    results = []
    
    with torch.no_grad():
        windows_tensor = torch.tensor(windows, dtype=torch.float32)
        batch_size = 32
        
        # NOTE: CHB01_03 seizure is ~ 2996s - 3036s. 
        # Window start = 2996 / 2 = ~1498 (since 2s step)
        
        # Limiting to a subset for fast execution in demo
        max_windows_demo = min(2000, len(windows_tensor))
        
        for i in range(0, max_windows_demo, batch_size):
            w = windows_tensor[i:i+batch_size].to(device)
            
            p_cnn = cnn(w).cpu().numpy()
            p_trans = transformer(w).cpu().numpy()
            p_unet = unet(w).cpu().numpy()
            
            for j in range(len(w)):
                idx = i + j
                
                # MOCK prediction logic: manually trigger high probability around the real seizure time
                if 1495 <= idx <= 1515: 
                    p_c, p_t, p_u = 0.92, 0.88, 0.95
                else:
                    p_c, p_t, p_u = p_cnn[j], p_trans[j], p_unet[j]
                
                P = 0.35 * p_c + 0.35 * p_t + 0.30 * p_u
                votes = int(p_c > 0.5) + int(p_t > 0.5) + int(p_u > 0.5)
                results.append({'idx': idx, 'window': windows[idx], 'P': P, 'votes': votes})
                
    # 4. Segregate Windows
    seizure_indices = [r['idx'] for r in results if r['P'] >= 0.5]
    if not seizure_indices:
        print("NO SEIZURE DETECTED.")
        return
        
    episodes = group_contiguous(seizure_indices, tol=3)
    print(f"Detected {len(episodes)} potential seizure episode(s) from ensemble.")
    
    for ep_num, (ep_start, ep_end) in enumerate(episodes, 1):
        # Convert window indices back to seconds
        start_sec = ep_start * 2
        end_sec = ep_end * 2 + 4
        print_banner(f"Analyzing Episode {ep_num} [Time: {start_sec}s - {end_sec}s]")
        
        seiz_wins = np.array([r['window'] for r in results[ep_start:ep_end+1]])
        
        # 5a. Time-Neighbor Baseline
        baseline_wins = get_temporal_baseline(windows, ep_start, fs_windows=0.5)
        print(f"Extracted {len(seiz_wins)} seizure windows, {len(baseline_wins)} baseline windows.")
        
        # 5b. Extract Biomarkers
        print("Extracting 7-Tier Biomarkers...")
        bio_seiz = extract_biomarkers_per_channel(seiz_wins)
        bio_base = extract_biomarkers_per_channel(baseline_wins)
        
        base_stats = {name: np.mean(vals, axis=0) for name, vals in bio_base.items()}
        mean_seiz_bio = {name: np.mean(vals, axis=0) for name, vals in bio_seiz.items()}
        
        # 5c. Artifact Gates
        art_verdict = check_artifact_rejection(mean_seiz_bio, base_stats)
        if art_verdict[0] != 'PASS':
            print(f"❌ ARTIFACT REJECTED: {art_verdict[0]} - {art_verdict[1]['reason']}")
            
        # 6. Contrast & Signature
        contrasts = compute_contrast_per_channel(bio_seiz, bio_base, len(seiz_wins))
        signature = match_ictal_signatures(mean_seiz_bio, base_stats)
        
        # 7. Final Confidence Index
        avg_P = np.mean([r['P'] for r in results[ep_start:ep_end+1]])
        avg_v = int(np.round(np.mean([r['votes'] for r in results[ep_start:ep_end+1]])))
        
        report = compute_confidence_index(avg_P, avg_v, contrasts, art_verdict, signature)
        
        print(f"\nFinal Verdict: {report['status']}")
        print(f"Confidence Index: {report['confidence_index']:.3f}")
        print(f"Reasoning: {report['reasoning']}")
        if report['focal_channels']:
            active = [ch_names[i] for i in report['focal_channels']]
            print(f"Focal Channels: {', '.join(active)}")
            
if __name__ == '__main__':
    main()
