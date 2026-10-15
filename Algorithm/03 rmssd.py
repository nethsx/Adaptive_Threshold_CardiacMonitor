"""
03_rmssd.py
============
Algorithm 3 of 10 — RMSSD (Root Mean Square of Successive Differences)

Measures short-term, beat-to-beat heart rate variability.
RMSSD is one of the most important HRV markers — reflects how much
each IBI differs from the one right before it.

Formula:
    RMSSD = sqrt( mean( (IBI[n+1] - IBI[n])^2 ) )   in ms

Interpretation:
    Low RMSSD  -> beats are very similar to each other (regular rhythm)
    High RMSSD -> large beat-to-beat jumps (irregular rhythm, e.g. AFib)

Pipeline:
    Raw PPG
        -> peak detection (Algorithm 1)
        -> IBI series (Algorithm 2)
        -> successive differences: diff(IBI)
        -> RMSSD = sqrt(mean(diff^2))

Usage:
    python 03_rmssd.py
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
    ibi = np.diff(peak_locs) / fs
    ibi = ibi[(ibi > 0.25) & (ibi < 2.0)]
    return ibi


def compute_rmssd(ibi):
    """
    RMSSD = sqrt(mean(successive_diff^2))  in milliseconds

    Captures short-term variability — specifically sensitive to
    sudden beat-to-beat changes, which is exactly what AFib produces.

    Unlike SDNN (overall spread), RMSSD only looks at CONSECUTIVE
    pairs, making it more specific to rhythm irregularity rather
    than just overall rate drift.
    """
    if len(ibi) < 2:
        return 0.0
    successive_diff = np.diff(ibi)              # seconds
    rmssd = np.sqrt(np.mean(successive_diff ** 2)) * 1000   # convert to ms
    return rmssd


def compute_sdnn(ibi):
    """
    SDNN = std(IBI) in ms — included for comparison alongside RMSSD.
    Reflects overall variability, not just beat-to-beat.
    """
    if len(ibi) == 0:
        return 0.0
    return np.std(ibi) * 1000


# =============================================================================
# MAIN — VALIDATE ON ALL 4 PPG CONDITIONS
# =============================================================================

if __name__ == '__main__':

    conditions = [
        ('Normal',      'ppg_normal.csv'),
        ('Bradycardia', 'ppg_brady.csv'),
        ('Tachycardia', 'ppg_tachy.csv'),
        ('AFib',        'ppg_afib.csv'),
    ]

    print('='*60)
    print('  RMSSD VALIDATION')
    print('='*60)
    print(f'{"Condition":15s} {"RMSSD (ms)":>12s} {"SDNN (ms)":>12s}')

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    results = []
    successive_diffs_all = []

    for name, fname in conditions:
        df     = pd.read_csv(DATA_DIR + fname)
        signal = df['ppg'].values

        filtered  = bandpass_filter(signal, fs)
        peak_locs = detect_peaks(filtered, fs)
        ibi       = compute_ibi(peak_locs, fs)

        rmssd = compute_rmssd(ibi)
        sdnn  = compute_sdnn(ibi)

        print(f'{name:15s} {rmssd:12.2f} {sdnn:12.2f}')
        results.append((name, rmssd, sdnn))

        successive_diffs_all.append((name, np.abs(np.diff(ibi)) * 1000))

    # ── Plot 1: RMSSD bar chart ───────────────────────────────────────────
    names  = [r[0] for r in results]
    rmssds = [r[1] for r in results]
    colors = ['#3399E6', '#26B36B', '#F25940', '#F2A30F']

    bars = ax[0].bar(names, rmssds, color=colors)
    ax[0].set_ylabel('RMSSD (ms)')
    ax[0].set_title('RMSSD Across Conditions', fontweight='bold')
    ax[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, rmssds):
        ax[0].text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}',
                   ha='center', fontsize=9)

    # ── Plot 2: Successive IBI differences over time ─────────────────────
    for (name, diffs), color in zip(successive_diffs_all, colors):
        ax[1].plot(diffs, label=name, color=color, linewidth=0.8, alpha=0.8)
    ax[1].axhline(50, color='gray', linestyle='--', linewidth=1, label='50ms ref')
    ax[1].set_xlabel('Beat pair index')
    ax[1].set_ylabel('|Successive IBI diff| (ms)')
    ax[1].set_title('Beat-to-Beat Differences', fontweight='bold')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR + '03_rmssd_results.png', dpi=120, bbox_inches='tight')
    print(f'\nPlot saved to {OUT_DIR}03_rmssd_results.png')

    print('\n' + '='*60)
    print('  SUMMARY')
    print('='*60)
    afib_rmssd = [r[1] for r in results if r[0] == 'AFib'][0]
    others_max = max(r[1] for r in results if r[0] != 'AFib')
    print(f'  AFib RMSSD        : {afib_rmssd:.2f} ms')
    print(f'  Highest non-AFib  : {others_max:.2f} ms')
    separation = afib_rmssd / others_max if others_max > 0 else float('inf')
    print(f'  Separation factor : {separation:.1f}x')
    print(f'  Status: {"PASS - AFib clearly separated" if separation > 3 else "FAIL - weak separation"}')
    print('='*60)