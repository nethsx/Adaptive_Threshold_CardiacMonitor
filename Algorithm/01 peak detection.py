"""
01_peak_detection.py
=====================
Algorithm 1 of 10 — Peak Detection

Finds heartbeat peak locations in a raw PPG signal.
This is the foundation algorithm — every other algorithm
(BPM, RMSSD, SD1/SD2, AFib detection, etc.) depends on
having accurate peak locations first.

Pipeline:
    Raw PPG signal
        -> Bandpass filter (0.5-8 Hz)   removes baseline wander + noise
        -> Peak detection                finds systolic peaks
        -> Output: peak sample indices

Designed to be portable to C++ (Arduino) for the T-Watch S3 later.
The logic here is intentionally simple — no heavy libraries beyond
basic filtering and peak finding, so it translates directly.

Usage:
    python 01_peak_detection.py
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import os

# ── CONFIG ──────────────────────────────────────────────────────────────
fs       = 100                      # Hz — MAX30102 sampling rate
DATA_DIR = 'data/'
OUT_DIR  = 'algorithms/outputs/'
os.makedirs(OUT_DIR, exist_ok=True)


def bandpass_filter(signal, fs, low=0.5, high=8.0, order=3):
    """
    Bandpass filter 0.5-8 Hz.
    Removes baseline wander (breathing, <0.5Hz) and high frequency noise.
    Covers the cardiac frequency band (30-480 BPM range).
    """
    nyq = fs / 2
    b, a = butter(order, [low/nyq, min(high, nyq*0.95)/nyq], btype='bandpass')
    return filtfilt(b, a, signal)


def detect_peaks(signal, fs):
    """
    Detects heartbeat peaks in a filtered PPG signal.

    MinPeakDistance = 0.4s -> limits detection to max ~150 BPM
                              (prevents double-counting one beat)
    MinPeakProminence = 30% of signal range -> ignores small noise bumps,
                              only counts genuine pulse peaks

    Returns:
        peak_locs : array of sample indices where peaks occur
    """
    min_dist = int(0.4 * fs)
    min_prom = 0.30 * (signal.max() - signal.min())

    peak_locs, properties = find_peaks(
        signal,
        distance=min_dist,
        prominence=min_prom
    )
    return peak_locs


def run_peak_detection(filepath, fs, condition_name):
    """
    Full pipeline: load -> filter -> detect peaks -> report
    """
    df     = pd.read_csv(filepath)
    signal = df['ppg'].values

    filtered  = bandpass_filter(signal, fs)
    peak_locs = detect_peaks(filtered, fs)

    # Compute resulting BPM estimate for sanity check
    if len(peak_locs) > 1:
        ibi = np.diff(peak_locs) / fs
        ibi = ibi[(ibi > 0.25) & (ibi < 2.0)]
        bpm = 60 / np.mean(ibi) if len(ibi) > 0 else 0
    else:
        bpm = 0

    print(f'{condition_name:12s}  peaks_found={len(peak_locs):4d}  '
          f'estimated_BPM={bpm:6.1f}')

    return filtered, peak_locs, bpm


def plot_detection(signal, filtered, peak_locs, fs, condition_name, ax):
    """
    Plots first 10 seconds showing raw vs filtered signal with detected peaks.
    """
    t = np.arange(len(signal)) / fs
    window = int(10 * fs)   # show first 10 seconds

    ax.plot(t[:window], filtered[:window], color='steelblue', linewidth=1.0, label='Filtered')
    peaks_in_window = peak_locs[peak_locs < window]
    ax.plot(t[peaks_in_window], filtered[peaks_in_window], 'rv',
            markersize=6, label='Detected Peaks')
    ax.set_title(condition_name, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.3)


# =============================================================================
# MAIN — VALIDATE ON ALL 4 PPG CONDITIONS
# =============================================================================

if __name__ == '__main__':

    conditions = [
        ('Normal (expect ~70 BPM)',      'ppg_normal.csv'),
        ('Bradycardia (expect ~45 BPM)', 'ppg_brady.csv'),
        ('Tachycardia (expect ~110 BPM)','ppg_tachy.csv'),
        ('AFib (expect ~75 BPM, irregular)', 'ppg_afib.csv'),
    ]

    print('='*60)
    print('  PEAK DETECTION VALIDATION')
    print('='*60)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    results = []
    for i, (name, fname) in enumerate(conditions):
        filepath = DATA_DIR + fname
        df       = pd.read_csv(filepath)
        signal   = df['ppg'].values

        filtered, peak_locs, bpm = run_peak_detection(filepath, fs, name)
        results.append((name, len(peak_locs), bpm))

        plot_detection(signal, filtered, peak_locs, fs, name, axes[i])

    plt.tight_layout()
    plt.suptitle('Peak Detection — First 10s of Each Condition', y=1.02, fontweight='bold')
    plt.savefig(OUT_DIR + '01_peak_detection_results.png', dpi=120, bbox_inches='tight')
    print(f'\nPlot saved to {OUT_DIR}01_peak_detection_results.png')

    print('\n' + '='*60)
    print('  SUMMARY')
    print('='*60)
    print(f'{"Condition":35s} {"Peaks":>8s} {"BPM":>8s}')
    for name, n_peaks, bpm in results:
        print(f'{name:35s} {n_peaks:8d} {bpm:8.1f}')
    print('='*60)