"""
04_sd1_sd2_poincare.py
========================
Algorithm 4 of 10 — SD1 / SD2 / SD1-SD2 Ratio (Poincare Analysis)

Measures rhythm regularity by plotting each IBI against the next one.
This is the single most powerful feature for AFib detection — it
captures CHAOS, not just speed or average variability.

Concept:
    Plot IBI(n) on x-axis, IBI(n+1) on y-axis.
    A perfectly regular heart -> every point falls on the diagonal line.
    A chaotic heart (AFib)    -> points scatter in a wide circle.

Formula:
    x = IBI(n),  y = IBI(n+1)
    SD1 = std(y - x) / sqrt(2)     -> short axis (beat-to-beat spread)
    SD2 = std(y + x) / sqrt(2)     -> long axis  (overall spread)
    SD1/SD2 ratio:
        ~0.2-0.4  -> narrow ellipse -> regular rhythm
        ~0.8-1.2  -> circular       -> chaotic rhythm (AFib)

Pipeline:
    Raw PPG
        -> peak detection (Algorithm 1)
        -> IBI series (Algorithm 2)
        -> SD1, SD2, ratio

Usage:
    python 04_sd1_sd2_poincare.py
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


def compute_poincare(ibi):
    """
    Computes SD1, SD2, and SD1/SD2 ratio from the IBI series.

    x = IBI(n)    - current interval
    y = IBI(n+1)  - next interval

    SD1 = width of the scatter perpendicular to the identity line
          -> sensitive to SHORT-term beat-to-beat variability
    SD2 = length of the scatter along the identity line
          -> sensitive to LONG-term overall variability

    Returns:
        SD1, SD2, ratio, x, y   (x,y returned for plotting)
    """
    if len(ibi) < 3:
        return 0.0, 0.0, 0.0, np.array([]), np.array([])

    x = ibi[:-1]
    y = ibi[1:]

    sd1 = (np.std(y - x) / np.sqrt(2)) * 1000   # ms
    sd2 = (np.std(y + x) / np.sqrt(2)) * 1000   # ms
    ratio = sd1 / sd2 if sd2 > 0 else 0.0

    return sd1, sd2, ratio, x, y


# =============================================================================
# MAIN — VALIDATE ON ALL 4 PPG CONDITIONS
# =============================================================================

if __name__ == '__main__':

    conditions = [
        ('Normal',      'ppg_normal.csv',  '#3399E6'),
        ('Bradycardia', 'ppg_brady.csv',   '#26B36B'),
        ('Tachycardia', 'ppg_tachy.csv',   '#F25940'),
        ('AFib',        'ppg_afib.csv',    '#F2A30F'),
    ]

    print('='*65)
    print('  SD1 / SD2 / POINCARE RATIO VALIDATION')
    print('='*65)
    print(f'{"Condition":15s} {"SD1 (ms)":>10s} {"SD2 (ms)":>10s} {"SD1/SD2":>10s}')

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.flatten()

    results = []
    for i, (name, fname, color) in enumerate(conditions):
        df     = pd.read_csv(DATA_DIR + fname)
        signal = df['ppg'].values

        filtered  = bandpass_filter(signal, fs)
        peak_locs = detect_peaks(filtered, fs)
        ibi       = compute_ibi(peak_locs, fs)

        sd1, sd2, ratio, x, y = compute_poincare(ibi)
        print(f'{name:15s} {sd1:10.2f} {sd2:10.2f} {ratio:10.3f}')
        results.append((name, sd1, sd2, ratio))

        # ── Poincare scatter plot ──────────────────────────────────────
        ax = axes[i]
        x_ms = x * 1000
        y_ms = y * 1000
        ax.scatter(x_ms, y_ms, s=15, color=color, alpha=0.5)

        lo = min(x_ms.min(), y_ms.min()) - 20
        hi = max(x_ms.max(), y_ms.max()) + 20
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='Identity line')

        ax.set_xlabel('IBI(n) (ms)')
        ax.set_ylabel('IBI(n+1) (ms)')
        ax.set_title(f'{name}  |  SD1={sd1:.1f}  SD2={sd2:.1f}  ratio={ratio:.3f}',
                    fontweight='bold', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    plt.savefig(OUT_DIR + '04_poincare_results.png', dpi=120, bbox_inches='tight')
    print(f'\nPlot saved to {OUT_DIR}04_poincare_results.png')

    print('\n' + '='*65)
    print('  SUMMARY')
    print('='*65)
    afib_ratio   = [r[3] for r in results if r[0] == 'AFib'][0]
    others_ratio = [r[3] for r in results if r[0] != 'AFib']
    max_other    = max(others_ratio)

    print(f'  AFib SD1/SD2         : {afib_ratio:.3f}')
    print(f'  Highest non-AFib     : {max_other:.3f}')
    print(f'  Separation gap       : {afib_ratio - max_other:.3f}')

    threshold = (afib_ratio + max_other) / 2
    print(f'  Suggested threshold  : {threshold:.3f}  (midpoint)')
    status = 'PASS' if afib_ratio > 0.7 and max_other < 0.5 else 'FAIL'
    print(f'  Status: {status}')
    print('='*65)