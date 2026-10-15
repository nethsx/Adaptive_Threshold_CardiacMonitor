"""
02_bpm_calculation.py
======================
Algorithm 2 of 10 — BPM Calculation

Computes heart rate (BPM) from detected peak locations.
Builds directly on Algorithm 1 (peak detection).

Two BPM outputs are computed:
    1. Overall Mean BPM  — average over the whole signal window
    2. Instantaneous BPM — beat-by-beat, useful for real-time display
                            and for spotting sudden rate changes

Pipeline:
    Raw PPG
        -> peak detection (Algorithm 1)
        -> IBI series (time between consecutive peaks)
        -> Mean BPM        = 60 / mean(IBI)
        -> Instantaneous BPM[i] = 60 / IBI[i]   (one value per beat)

Usage:
    python 02_bpm_calculation.py
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import os

fs       = 100
DATA_DIR = 'data/'
OUT_DIR  = 'algorithms/outputs/'
os.makedirs(OUT_DIR, exist_ok=True)


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
    """
    Converts peak sample locations to Inter-Beat-Interval (IBI) series in seconds.
    Removes physiologically impossible intervals:
        IBI < 0.25s -> >240 BPM, noise/double-detection
        IBI > 2.0s  -> <30 BPM,  missed beat
    """
    ibi = np.diff(peak_locs) / fs
    ibi = ibi[(ibi > 0.25) & (ibi < 2.0)]
    return ibi


def compute_mean_bpm(ibi):
    """
    Mean BPM over the whole signal window.
    This is the single-number summary used for display and condition detection.
    """
    if len(ibi) == 0:
        return 0.0
    return 60.0 / np.mean(ibi)


def compute_instantaneous_bpm(ibi):
    """
    Instantaneous BPM — one value per beat.
    Useful for:
        - real-time display on watch (updates every heartbeat)
        - detecting sudden rate changes (e.g. during HRR, fall response)
    """
    return 60.0 / ibi


# =============================================================================
# MAIN — VALIDATE ON ALL 4 PPG CONDITIONS
# =============================================================================

if __name__ == '__main__':

    conditions = [
        ('Normal',      'ppg_normal.csv',   70),
        ('Bradycardia', 'ppg_brady.csv',    45),
        ('Tachycardia', 'ppg_tachy.csv',   110),
        ('AFib',        'ppg_afib.csv',     75),
    ]

    print('='*65)
    print('  BPM CALCULATION VALIDATION')
    print('='*65)
    print(f'{"Condition":15s} {"Expected":>10s} {"Mean BPM":>10s} {"Min":>8s} {"Max":>8s} {"Std":>8s}')

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    results = []
    for i, (name, fname, expected_bpm) in enumerate(conditions):
        df     = pd.read_csv(DATA_DIR + fname)
        signal = df['ppg'].values

        filtered  = bandpass_filter(signal, fs)
        peak_locs = detect_peaks(filtered, fs)
        ibi       = compute_ibi(peak_locs, fs)

        mean_bpm  = compute_mean_bpm(ibi)
        inst_bpm  = compute_instantaneous_bpm(ibi)

        print(f'{name:15s} {expected_bpm:10d} {mean_bpm:10.1f} '
              f'{inst_bpm.min():8.1f} {inst_bpm.max():8.1f} {inst_bpm.std():8.2f}')

        results.append((name, expected_bpm, mean_bpm))

        # Plot instantaneous BPM over time — shows stability vs chaos
        ax = axes[i]
        ax.plot(inst_bpm, color='darkorange', linewidth=0.9)
        ax.axhline(mean_bpm, color='black', linestyle='--', linewidth=1,
                   label=f'Mean = {mean_bpm:.1f} BPM')
        ax.set_title(f'{name}  (expected ~{expected_bpm} BPM)', fontweight='bold')
        ax.set_xlabel('Beat index')
        ax.set_ylabel('Instantaneous BPM')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Instantaneous BPM — Beat by Beat', y=1.02, fontweight='bold')
    plt.savefig(OUT_DIR + '02_bpm_calculation_results.png', dpi=120, bbox_inches='tight')
    print(f'\nPlot saved to {OUT_DIR}02_bpm_calculation_results.png')

    print('\n' + '='*65)
    print('  SUMMARY — Mean BPM vs Expected')
    print('='*65)
    all_pass = True
    for name, expected, mean_bpm in results:
        diff = abs(mean_bpm - expected)
        status = 'PASS' if diff < 3 else 'FAIL'
        if status == 'FAIL':
            all_pass = False
        print(f'  {name:15s}  expected={expected:5d}  got={mean_bpm:6.1f}  '
              f'diff={diff:5.2f}  [{status}]')
    print('='*65)
    print(f'  Overall: {"ALL PASSED" if all_pass else "SOME FAILED"}')
    print('='*65)