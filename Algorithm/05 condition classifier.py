"""
05_condition_classifier.py
============================
Algorithm 5 of 10 — Condition Classifier  (BLIND TEST)

Classifies a raw PPG signal into one of 4 conditions:
    Normal / Bradycardia / Tachycardia / AFib

A single threshold (SD1/SD2 alone) can only separate AFib from
"not AFib" — it has no information about heart RATE, so it cannot
tell Normal from Brady from Tachy. To classify all 4 conditions,
TWO features are needed together:

    Feature 1 (Rhythm) : SD1/SD2 ratio        - chaos vs regularity
    Feature 2 (Speed)  : Mean IBI (ms)         - translates to BPM

Decision tree logic:

                     Is SD1/SD2 >= 0.65 ?
                            |
              YES ----------+---------- NO
               |                         |
            AFib                 What is Mean IBI?
                                         |
                 -----------------------+-----------------------
                 |                      |                      |
          IBI < 600ms           600ms <= IBI <= 1100ms     IBI > 1100ms
          (BPM > 100)              (BPM 55-100)              (BPM < 55)
                 |                      |                      |
           Tachycardia              Normal                Bradycardia

── WHY THIS IS A BLIND TEST ──────────────────────────────────────────
The 4 dataset files are shuffled into random order and relabelled
Signal A/B/C/D. classify_condition() only ever receives the raw
signal - no filename, no label, no hint about which condition it is.
It predicts first; the true answer is revealed only afterward, to
score accuracy. This mirrors the real watch, which has no prior
knowledge of the wearer's condition and must decide from the
PPG signal alone.

Usage:
    python 05_condition_classifier.py
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import random

fs        = 100
DATA_DIR  = 'data/'
OUT_DIR   = 'algorithms/outputs/'
os.makedirs(OUT_DIR, exist_ok=True)

# ── THRESHOLDS — derived from Algorithms 2 and 4 validation ─────────────
AFIB_SD_THRESHOLD = 0.65     # SD1/SD2 above this -> AFib
TACHY_IBI_MS      = 600      # Mean IBI below this  -> Tachycardia (BPM>100)
BRADY_IBI_MS      = 1100     # Mean IBI above this  -> Bradycardia (BPM<55)
# between TACHY_IBI_MS and BRADY_IBI_MS (and SD ratio low) -> Normal


def bandpass_filter(signal, fs, low=0.5, high=8.0, order=3):
    nyq = fs / 2
    b, a = butter(order, [low/nyq, min(high, nyq*0.95)/nyq], btype='bandpass')
    return filtfilt(b, a, signal)


def detect_peaks(signal, fs):
    min_dist = int(0.4 * fs)
    min_prom = 0.30 * (signal.max() - signal.min())
    peak_locs, _ = find_peaks(signal, distance=min_dist, prominence=min_prom)
    return peak_locs


def compute_ibi(peak_locs, fs):
    ibi = np.diff(peak_locs) / fs
    ibi = ibi[(ibi > 0.25) & (ibi < 2.0)]
    return ibi


def compute_features(ibi):
    """
    Computes the two features needed for classification:
        sd_ratio  : SD1/SD2  (rhythm regularity)
        mean_ibi  : mean IBI in ms  (heart rate / speed)
    """
    if len(ibi) < 3:
        return 0.0, 0.0

    x = ibi[:-1]
    y = ibi[1:]
    sd1 = (np.std(y - x) / np.sqrt(2)) * 1000
    sd2 = (np.std(y + x) / np.sqrt(2)) * 1000
    sd_ratio = sd1 / sd2 if sd2 > 0 else 0.0

    mean_ibi_ms = np.mean(ibi) * 1000

    return sd_ratio, mean_ibi_ms


def classify_condition(signal, fs):
    """
    BLIND prediction function - the exact logic that will run on the watch.
    Input : raw signal only - no label, no filename, no hint.
    Output: condition (str), confidence (0-100%), sd_ratio, mean_ibi_ms

    Decision tree:
        1. Check rhythm chaos first (SD1/SD2) -> AFib branch
        2. If rhythm is regular, check speed (Mean IBI) -> Brady/Normal/Tachy
    """
    filtered  = bandpass_filter(signal, fs)
    peak_locs = detect_peaks(filtered, fs)
    ibi       = compute_ibi(peak_locs, fs)
    sd_ratio, mean_ibi_ms = compute_features(ibi)

    # ── Branch 1: rhythm check ──────────────────────────────────────────
    if sd_ratio >= AFIB_SD_THRESHOLD:
        condition = 'AFib'
        margin = sd_ratio - AFIB_SD_THRESHOLD
        confidence = np.clip(50 + (margin / 0.3) * 50, 0, 100)

    # ── Branch 2: speed check (only reached if rhythm is regular) ──────
    elif mean_ibi_ms < TACHY_IBI_MS:
        condition = 'Tachycardia'
        margin = (TACHY_IBI_MS - mean_ibi_ms) / TACHY_IBI_MS
        confidence = np.clip(50 + margin * 150, 0, 100)

    elif mean_ibi_ms > BRADY_IBI_MS:
        condition = 'Bradycardia'
        margin = (mean_ibi_ms - BRADY_IBI_MS) / BRADY_IBI_MS
        confidence = np.clip(50 + margin * 150, 0, 100)

    else:
        condition = 'Normal'
        # confidence based on how centered it is in the normal band
        band_center = (TACHY_IBI_MS + BRADY_IBI_MS) / 2
        band_half   = (BRADY_IBI_MS - TACHY_IBI_MS) / 2
        dist_from_center = abs(mean_ibi_ms - band_center) / band_half
        confidence = np.clip(100 - dist_from_center * 50, 50, 100)

    return condition, confidence, sd_ratio, mean_ibi_ms


# =============================================================================
# MAIN — BLIND VALIDATION ON ALL 4 CONDITIONS
# =============================================================================

if __name__ == '__main__':

    # ── Step 1: true answers, hidden until after prediction ───────────────
    true_answers = {
        'ppg_normal.csv': 'Normal',
        'ppg_brady.csv':  'Bradycardia',
        'ppg_tachy.csv':  'Tachycardia',
        'ppg_afib.csv':   'AFib',
    }
    filenames = list(true_answers.keys())

    # ── Step 2: shuffle into random order, relabel as mystery signals ─────
    random.seed(None)
    shuffled = filenames.copy()
    random.shuffle(shuffled)

    mystery_labels = ['Signal A', 'Signal B', 'Signal C', 'Signal D']
    mystery_map    = dict(zip(mystery_labels, shuffled))

    print('='*85)
    print('  CONDITION CLASSIFIER — BLIND TEST (4-way: Normal/Brady/Tachy/AFib)')
    print('='*85)
    print('  Algorithm uses TWO features: SD1/SD2 (rhythm) + Mean IBI (speed).')
    print('  No filenames or labels passed in. True answers revealed after prediction.\n')

    print(f'{"Mystery ID":12s} {"SD1/SD2":>9s} {"MeanIBI(ms)":>12s} {"Predicted":>14s} '
          f'{"Confidence":>11s} {"True Label":>14s} {"Result":>8s}')

    results  = []
    all_pass = True

    for mystery_id in mystery_labels:
        fname = mystery_map[mystery_id]

        # ── Algorithm sees ONLY this — no label, no filename hint ──────
        df     = pd.read_csv(DATA_DIR + fname)
        signal = df['ppg'].values
        condition, confidence, sd_ratio, mean_ibi_ms = classify_condition(signal, fs)
        # ─────────────────────────────────────────────────────────────

        true_name = true_answers[fname]
        correct   = (condition == true_name)
        if not correct:
            all_pass = False
        status = 'PASS' if correct else 'FAIL'

        print(f'{mystery_id:12s} {sd_ratio:9.3f} {mean_ibi_ms:12.1f} {condition:>14s} '
              f'{confidence:10.1f}% {true_name:>14s} {status:>8s}')

        results.append((mystery_id, true_name, sd_ratio, mean_ibi_ms, condition, confidence))

    # ── Plot: 2D feature space scatter (SD1/SD2 vs Mean IBI) ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    cond_colors = {'Normal':'#3399E6', 'Bradycardia':'#26B36B',
                   'Tachycardia':'#F25940', 'AFib':'#F2A30F'}

    # Plot 1: feature space with decision boundaries
    ax = axes[0]
    for r in results:
        mid, true_name, sd, ibi_ms, pred, conf = r
        color = cond_colors[true_name]
        ax.scatter(ibi_ms, sd, s=160, color=color, edgecolor='black', linewidth=1.2, zorder=3)
        ax.annotate(f'{mid}\n(true: {true_name})', (ibi_ms, sd),
                   textcoords='offset points', xytext=(8, 8), fontsize=8)

    ax.axhline(AFIB_SD_THRESHOLD, color='red', linestyle='--', linewidth=1, label='AFib threshold')
    ax.axvline(TACHY_IBI_MS, color='gray', linestyle=':', linewidth=1, label='Tachy/Normal boundary')
    ax.axvline(BRADY_IBI_MS, color='gray', linestyle='-.', linewidth=1, label='Normal/Brady boundary')
    ax.set_xlabel('Mean IBI (ms)  — Speed')
    ax.set_ylabel('SD1/SD2 Ratio  — Rhythm')
    ax.set_title('Decision Space: Rhythm vs Speed', fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    # Plot 2: confusion matrix — industry standard evaluation format
    classes       = ['Normal', 'Bradycardia', 'Tachycardia', 'AFib']
    true_labels   = [r[1] for r in results]   # ground truth
    pred_labels   = [r[4] for r in results]   # algorithm predictions

    cm = confusion_matrix(true_labels, pred_labels, labels=classes)

    ax2 = axes[1]
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        annot_kws={'size': 14, 'weight': 'bold'},
        ax=ax2
    )
    ax2.set_title('Blind Validation — Confusion Matrix', fontweight='bold')
    ax2.set_xlabel('Predicted (Algorithm Guess)')
    ax2.set_ylabel('True (Ground Truth)')

    plt.tight_layout()
    plt.savefig(OUT_DIR + '05_condition_classifier_results.png', dpi=120, bbox_inches='tight')
    print(f'\nPlot saved to {OUT_DIR}05_condition_classifier_results.png')

    print('\n' + '='*85)
    print('  CLASSIFICATION REPORT')
    print('='*85)
    print(classification_report(true_labels, pred_labels, labels=classes,
                                target_names=classes, zero_division=0))

    n_correct = sum(1 for r in results if r[4] == r[1])
    print('='*85)
    print(f'  OVERALL: {"ALL PASSED" if all_pass else "SOME FAILED"}  '
          f'({n_correct}/{len(results)} correct)')
    print('='*85)