"""
07_activity_classifier.py
============================
Algorithm 7 of 10 — Activity Classifier  (BLIND TEST)

Classifies accelerometer data into one of 5 activity states:
    Sleep / Rest / Light / Moderate / Intense

This feeds directly into Algorithm 8 (context-aware threshold) and
is also what Algorithm 6 used for its "motion_paused" gate.

Feature used: standard deviation of acceleration magnitude over a
rolling window. At rest, magnitude sits almost perfectly flat at ~1g
(gravity only). During activity, limb swing and impact cause the
magnitude to oscillate - the MORE intense the activity, the LARGER
that oscillation. So std(magnitude) is a clean single-number proxy
for "how much is this person moving right now".

Classification bands (calibrated from measured std(magnitude) per
activity in the synthetic dataset: Sleep~0.005, Rest~0.017,
Light~0.110, Moderate~0.313, Intense~0.526 - thresholds are set at
the midpoint between each pair of neighboring class means):
    std < 0.011        -> Sleep      (near motionless, deep stillness)
    0.011 <= std < 0.064 -> Rest       (sitting, minor shifting)
    0.064 <= std < 0.212 -> Light      (walking pace)
    0.212 <= std < 0.420 -> Moderate   (jogging pace)
    std >= 0.420        -> Intense    (running pace)

── WHY THIS IS A BLIND TEST ──────────────────────────────────────────
The 5 accelerometer files are shuffled and relabelled Signal A-E.
classify_activity() receives only the raw ax/ay/az stream - no
filename or label. True answers are revealed only after prediction,
to score accuracy.

Usage:
    python 07_activity_classifier.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import random

fs        = 100
DATA_DIR  = 'data/'
OUT_DIR   = 'algorithms/outputs/'
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_S = 10   # seconds per classification window

# ── Classification thresholds on std(magnitude) ─────────────────────────
# Set at the midpoint between each pair of neighboring class means,
# measured from the synthetic dataset (see module docstring).
THRESHOLDS = [
    (0.011, 'Sleep'),
    (0.064, 'Rest'),
    (0.212, 'Light'),
    (0.420, 'Moderate'),
    (np.inf, 'Intense'),
]


def classify_activity(ax, ay, az):
    """
    BLIND prediction function - the exact logic that will run on the watch.
    Input : raw ax, ay, az arrays only - no label, no filename, no hint.
    Output: activity label (str), std_magnitude (float)
    """
    magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    std_mag   = np.std(magnitude)

    for threshold, label in THRESHOLDS:
        if std_mag < threshold:
            return label, std_mag
    return 'Intense', std_mag   # fallback, should not reach here


# =============================================================================
# MAIN — BLIND VALIDATION ON ALL 5 ACTIVITY STATES
# =============================================================================

if __name__ == '__main__':

    true_answers = {
        'accel_sleep.csv':    'Sleep',
        'accel_rest.csv':     'Rest',
        'accel_light.csv':    'Light',
        'accel_moderate.csv': 'Moderate',
        'accel_intense.csv':  'Intense',
    }
    filenames = list(true_answers.keys())

    random.seed(None)
    shuffled = filenames.copy()
    random.shuffle(shuffled)

    mystery_labels = ['Signal A', 'Signal B', 'Signal C', 'Signal D', 'Signal E']
    mystery_map    = dict(zip(mystery_labels, shuffled))

    print('='*80)
    print('  ACTIVITY CLASSIFIER — BLIND TEST (5-way: Sleep/Rest/Light/Moderate/Intense)')
    print('='*80)
    print('  Each file is classified window-by-window (10s windows), then the')
    print('  MAJORITY vote across windows gives the final prediction per file.\n')

    print(f'{"Mystery ID":12s} {"StdMag":>9s} {"Predicted":>11s} {"True Label":>11s} {"Result":>8s}')

    results = []
    all_pass = True
    window_level_data = []   # for the scatter plot

    for mystery_id in mystery_labels:
        fname = mystery_map[mystery_id]

        # ── Algorithm sees ONLY raw accel data — no label ───────────────
        df = pd.read_csv(DATA_DIR + fname)
        ax_full, ay_full, az_full = df['ax'].values, df['ay'].values, df['az'].values

        window_samples = WINDOW_S * fs
        n_windows = len(ax_full) // window_samples

        window_preds = []
        window_stds  = []
        for w in range(n_windows):
            s, e = w*window_samples, (w+1)*window_samples
            pred, std_mag = classify_activity(ax_full[s:e], ay_full[s:e], az_full[s:e])
            window_preds.append(pred)
            window_stds.append(std_mag)
            window_level_data.append((mystery_id, std_mag, pred))

        # Majority vote across windows -> final file-level prediction
        final_pred = max(set(window_preds), key=window_preds.count)
        avg_std    = np.mean(window_stds)
        # ─────────────────────────────────────────────────────────────

        true_name = true_answers[fname]
        correct   = (final_pred == true_name)
        if not correct:
            all_pass = False
        status = 'PASS' if correct else 'FAIL'

        print(f'{mystery_id:12s} {avg_std:9.4f} {final_pred:>11s} {true_name:>11s} {status:>8s}')
        results.append((mystery_id, true_name, final_pred, avg_std))

    # ── Confusion matrix ────────────────────────────────────────────────
    classes     = ['Sleep', 'Rest', 'Light', 'Moderate', 'Intense']
    true_labels = [r[1] for r in results]
    pred_labels = [r[2] for r in results]
    cm = confusion_matrix(true_labels, pred_labels, labels=classes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Plot 1: std(magnitude) per window, colored by true file
    ax1 = axes[0]
    colors = {'Sleep':'#8E44AD','Rest':'#3399E6','Light':'#26B36B',
              'Moderate':'#F2A30F','Intense':'#D62728'}
    true_name_by_mystery = {r[0]: r[1] for r in results}
    for mid in mystery_labels:
        pts = [d for d in window_level_data if d[0] == mid]
        stds_this = [p[1] for p in pts]
        tname = true_name_by_mystery[mid]
        ax1.scatter([mid]*len(stds_this), stds_this, color=colors[tname],
                   alpha=0.6, s=25, label=tname)
    for thresh, lbl in THRESHOLDS[:-1]:
        ax1.axhline(thresh, color='gray', linestyle='--', linewidth=0.7)
    handles, labels_ = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    ax1.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper left')
    ax1.set_ylabel('Std(magnitude) per 10s window')
    ax1.set_title('Per-window Activity Index (colored by true label)', fontweight='bold')
    ax1.grid(alpha=0.3)

    # Plot 2: confusion matrix
    ax2 = axes[1]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, cbar=False,
                annot_kws={'size':13,'weight':'bold'}, ax=ax2)
    ax2.set_title('Blind Validation — Confusion Matrix', fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(OUT_DIR + '07_activity_classifier_results.png', dpi=120, bbox_inches='tight')
    print(f'\nPlot saved to {OUT_DIR}07_activity_classifier_results.png')

    print('\n' + '='*80)
    print('  CLASSIFICATION REPORT')
    print('='*80)
    print(classification_report(true_labels, pred_labels, labels=classes,
                                target_names=classes, zero_division=0))

    n_correct = sum(1 for r in results if r[1] == r[2])
    print('='*80)
    print(f'  OVERALL: {"ALL PASSED" if all_pass else "SOME FAILED"}  '
          f'({n_correct}/{len(results)} correct)')
    print('='*80)