import numpy as np

FRONTAL_CHANNELS = [0, 4, 8, 12]  # FP1-F7, FP1-F3, FP2-F4, FP2-F8
OTHER_CHANNELS   = [i for i in range(18) if i not in FRONTAL_CHANNELS]

def check_artifact_rejection(bio_window, baseline_stats):
    ll_ratio  = bio_window['line_length'] / (baseline_stats['line_length'] + 1e-6)
    rms_ratio = bio_window['rms'] / (baseline_stats['rms'] + 1e-6)
    spec_ent  = bio_window['spectral_entropy']
    kurt      = bio_window['kurtosis']
    gamma_rel = bio_window['rel_high_gamma_power']
    td_rel    = bio_window['rel_theta_delta_power']

    muscle_mask = (gamma_rel > 0.45) & (spec_ent > 0.80) & (ll_ratio > 3.0)
    if np.sum(muscle_mask) >= 4:
        return 'REJECT_MUSCLE', {
            'reason': 'High gamma + high spectral entropy = broadband EMG, not seizure',
            'channels_affected': int(np.sum(muscle_mask))
        }

    frontal_delta_high = np.mean(td_rel[FRONTAL_CHANNELS]) > 0.70
    frontal_kurt_high  = np.mean(kurt[FRONTAL_CHANNELS]) > 4.0
    other_ch_flat      = np.mean(rms_ratio[OTHER_CHANNELS]) < 1.3
    if frontal_delta_high and frontal_kurt_high and other_ch_flat:
        return 'REJECT_EYEBLINK', {
            'reason': 'High delta+kurtosis confined to frontopolar channels, zero propagation',
            'frontal_kurt': float(np.mean(kurt[FRONTAL_CHANNELS]))
        }

    return 'PASS', None

def match_ictal_signatures(bio_window, baseline_stats):
    ll_ratio  = bio_window['line_length'] / (baseline_stats['line_length'] + 1e-6)
    spec_ent  = bio_window['spectral_entropy']
    kurt      = bio_window['kurtosis']
    td_rel    = bio_window['rel_theta_delta_power']
    hjorth_c  = bio_window['hjorth_complexity']

    sig_a = (ll_ratio > 2.5) & (kurt > 3.0) & (spec_ent < 0.65)
    sig_b = (spec_ent < 0.55) & (td_rel > 0.50) & (ll_ratio > 1.8) & (hjorth_c < 1.35)

    combined = sig_a | sig_b
    channels_with_support = int(np.sum(combined))
    
    return {
        'spike_wave_channels': int(np.sum(sig_a)),
        'rhythmic_sync_channels': int(np.sum(sig_b)),
        'total_supporting_channels': channels_with_support,
        'support_ratio': channels_with_support / 18.0
    }

def compute_confidence_index(ensemble_prob, model_agreement, contrasts, artifact_verdict, signature_match, n_biomarkers_total=7):
    if artifact_verdict[0] != 'PASS':
        return {
            'confidence_index': 0.0,
            'status': 'REJECTED',
            'reasoning': f"Artifact detected: {artifact_verdict[0]} — {artifact_verdict[1]['reason']}",
            'artifact_type': artifact_verdict[0]
        }

    A_norm = model_agreement / 3.0

    significant_count = sum(1 for c in contrasts.values() if c['n_significant'] >= 4 and c['reliable'])
    B = significant_count / n_biomarkers_total

    sig_z_scores = [
        float(np.mean(np.abs(c['z_scores'][c['significant_channels']])))
        for c in contrasts.values() if c['n_significant'] >= 4 and c['reliable']
    ]
    mean_z = np.mean(sig_z_scores) if sig_z_scores else 0.0

    CI = ensemble_prob * A_norm * B

    focal_channels = []
    for c in contrasts.values():
        focal_channels.extend(np.where(c['significant_channels'])[0].tolist())
    
    from collections import Counter
    ch_counts = Counter(focal_channels)
    most_active = [i for i, cnt in ch_counts.items() if cnt >= 3]

    if CI >= 0.45 and A_norm >= 0.67 and B >= 0.43:
        status = 'CONFIRMED'
        reasoning = f"Strong detection: {model_agreement}/3 models agree (P={ensemble_prob:.2f}), {significant_count}/{n_biomarkers_total} biomarkers deviate significantly."
    elif CI >= 0.15 and (A_norm >= 0.33 or B >= 0.29):
        status = 'PROBABLE'
        reasoning = f"Moderate evidence: {model_agreement}/3 models, {significant_count} biomarkers support."
    else:
        status = 'REJECTED'
        reasoning = f"Insufficient evidence: CI={CI:.3f}, low model/biomarker agreement."

    return {
        'confidence_index': round(CI, 4),
        'ensemble_prob': round(ensemble_prob, 4),
        'model_agreement': f"{model_agreement}/3",
        'biomarker_support': f"{significant_count}/{n_biomarkers_total}",
        'mean_z_score': round(mean_z, 2),
        'status': status,
        'reasoning': reasoning,
        'ictal_signature': signature_match,
        'focal_channels': most_active,
        'reliable': all(c['reliable'] for c in contrasts.values()) if contrasts else False
    }
