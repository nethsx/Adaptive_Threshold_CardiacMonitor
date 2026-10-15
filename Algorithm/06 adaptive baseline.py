"""
06_adaptive_baseline.py
=========================
Algorithm 6 of 10 — Adaptive Baseline  (v2: Dynamic Alpha + Anomaly
Rejection + Context Gating)

Learns the WEARER'S OWN personal normal BPM range over time, instead
of using a fixed population threshold (e.g. "60-100 BPM = normal for
everyone"). v2 fixes three real-world weaknesses of a plain EMA:

1. DYNAMIC ALPHA (fast warm-up)
   Plain EMA with a small fixed alpha takes ~10 minutes to converge -
   unacceptable for a real user. Here alpha starts high (0.30) for the
   first 60s so the baseline snaps quickly to the user's actual rate,
   then decays exponentially down to a stable tracking rate (0.01) by
   180s. Calibration time drops from ~10 min to ~3 min.

2. ANOMALY REJECTION (artifact / arrhythmia gating)
   If a single 15s BPM reading deviates more than 20% from the current
   baseline, it is treated as an outlier (motion artifact, a transient
   PVC burst, momentary panic spike, etc.) and EXCLUDED from updating
   the baseline. The baseline stays flat through the anomaly instead
   of being dragged off course by one bad window.

3. CONTEXT GATING (activity-aware pausing)
   Accelerometer magnitude is checked alongside each BPM reading. If
   magnitude indicates the wearer is physically active (not resting),
   baseline learning is PAUSED entirely for that window - a walking
   or exercise-induced BPM rise is real physiology, not something the
   resting baseline should absorb.

Formula (dynamic alpha):
    alpha(t) = alpha_min + (alpha_max - alpha_min) * exp(-t / tau)
    alpha_max = 0.30, alpha_min = 0.01, tau = 60s

Pipeline:
    PPG signal + accelerometer magnitude (synchronized)
        -> per 15s window: compute BPM (Algorithms 1-2) + mean |accel|
        -> context gate check (is wearer resting?)
        -> anomaly check (is this BPM reading plausible given baseline?)
        -> dynamic-alpha EMA update (if both checks pass)

Usage:
    python 06_adaptive_baseline.py
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
import os

fs        = 100
DATA_DIR  = 'data/'
OUT_DIR   = 'algorithms/outputs/'
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_S        = 15      # seconds per BPM measurement window
ALPHA_MAX       = 0.30    # fast warm-up smoothing factor
ALPHA_MIN       = 0.01    # stable long-term smoothing factor
TAU             = 60.0    # decay time constant (s) for alpha
ANOMALY_PCT     = 0.20    # reject readings deviating >20% from baseline
MOTION_THRESHOLD = 1.08   # accel magnitude above this = "active", pause learning


def bandpass_filter(signal, fs, low=0.5, high=8.0, order=3):
    nyq = fs / 2
    b, a = butter(order, [low/nyq, min(high, nyq*0.95)/nyq], btype='bandpass')
    return filtfilt(b, a, signal)


def detect_peaks(signal, fs):
    min_dist = int(0.4 * fs)
    min_prom = 0.30 * (signal.max() - signal.min())
    peak_locs, _ = find_peaks(signal, distance=min_dist, prominence=min_prom)
    return peak_locs


def compute_window_bpm(signal_window, fs):
    filtered  = bandpass_filter(signal_window, fs)
    peak_locs = detect_peaks(filtered, fs)
    if len(peak_locs) < 2:
        return None
    ibi = np.diff(peak_locs) / fs
    ibi = ibi[(ibi > 0.25) & (ibi < 2.0)]
    if len(ibi) == 0:
        return None
    return 60.0 / np.mean(ibi)


class AdaptiveBaselineV2:
    """
    Adaptive personal BPM baseline with:
        - dynamic alpha warm-up
        - anomaly rejection (gating on % deviation)
        - context gating (pauses on detected motion)
    """
    def __init__(self, alpha_max=ALPHA_MAX, alpha_min=ALPHA_MIN, tau=TAU,
                 anomaly_pct=ANOMALY_PCT, motion_threshold=MOTION_THRESHOLD):
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.tau        = tau
        self.anomaly_pct = anomaly_pct
        self.motion_threshold = motion_threshold

        self.mean = None
        self.std  = 5.0
        self.elapsed_learning_time = 0.0   # only accumulates during accepted updates

    def current_alpha(self):
        """Dynamic alpha: high at start, decays exponentially toward alpha_min."""
        return self.alpha_min + (self.alpha_max - self.alpha_min) * np.exp(
            -self.elapsed_learning_time / self.tau)

    def update(self, bpm, accel_magnitude, window_duration=WINDOW_S):
        """
        Processes one new BPM reading.
        Returns: (mean, std, status) where status is one of:
            'updated'        - reading accepted, baseline updated
            'anomaly_rejected' - reading deviated too much, baseline untouched
            'motion_paused'    - wearer active, baseline untouched
        """
        # ── Context gate: is the wearer resting? ───────────────────────
        if accel_magnitude > self.motion_threshold:
            return self.mean, self.std, 'motion_paused'

        # ── First ever reading — initialize directly ───────────────────
        if self.mean is None:
            self.mean = bpm
            self.std  = 5.0
            self.elapsed_learning_time += window_duration
            return self.mean, self.std, 'updated'

        # ── Anomaly gate: is this plausible given current baseline? ────
        pct_deviation = abs(bpm - self.mean) / self.mean
        if pct_deviation > self.anomaly_pct:
            return self.mean, self.std, 'anomaly_rejected'

        # ── Accepted — apply dynamic-alpha EMA update ───────────────────
        alpha = self.current_alpha()
        deviation = abs(bpm - self.mean)
        self.mean = alpha * bpm + (1 - alpha) * self.mean
        self.std  = alpha * deviation + (1 - alpha) * self.std
        self.elapsed_learning_time += window_duration

        return self.mean, self.std, 'updated'

    def z_score(self, bpm):
        if self.mean is None or self.std == 0:
            return 0.0
        return (bpm - self.mean) / self.std

    def personal_range(self, n_sigma=2):
        if self.mean is None:
            return None, None
        return self.mean - n_sigma*self.std, self.mean + n_sigma*self.std


# =============================================================================
# BUILD TEST SCENARIO — normal rest + injected anomaly + injected motion burst
#   This stresses all three fixes in one 15-minute run:
#     0:00-4:00   resting, stable ~70 BPM           (normal learning)
#     4:00-4:30   anomaly burst ~95 BPM              (artifact/PVC-like spike)
#     4:30-8:00   back to resting ~70 BPM            (should recover cleanly)
#     8:00-9:30   walking, accel elevated, ~95 BPM   (should PAUSE, not corrupt)
#     9:30-15:00  resting again ~70 BPM              (baseline still accurate)
# =============================================================================

def build_test_scenario():
    np.random.seed(123)
    segments = [
        # (duration_s, bpm, accel_mag, label)
        (240, 70, 0.98, 'rest'),
        (30,  95, 0.99, 'anomaly'),     # sudden BPM spike, wearer still resting
        (210, 70, 0.98, 'rest'),
        (90,  95, 1.25, 'motion'),      # walking - BPM rise is real exertion
        (330, 70, 0.98, 'rest'),
    ]

    all_ppg   = []
    all_accel = []
    labels    = []

    for dur, bpm, accel_mag, label in segments:
        n_windows = dur // WINDOW_S
        for _ in range(n_windows):
            # Generate one window of PPG at target BPM using simple AR(1) IBI
            base_ibi = 60.0 / bpm
            n_beats  = int(WINDOW_S / base_ibi) + 3
            ibi      = np.zeros(n_beats)
            ibi[0]   = base_ibi
            for i in range(1, n_beats):
                ibi[i] = base_ibi + 0.85*(ibi[i-1]-base_ibi) + 0.006*np.random.randn()
            ibi = np.clip(ibi, 0.3, 1.5)

            peak_times = np.cumsum(np.concatenate([[0.2], ibi]))
            peak_samp  = (peak_times * fs).astype(int)
            N = WINDOW_S * fs
            peak_samp  = peak_samp[peak_samp < N]

            sig = np.zeros(N)
            for p in peak_samp:
                idx = np.arange(max(0,p-8), min(N,p+20))
                rel = idx - p
                sig[idx] += np.exp(-0.5*(rel/3.0)**2) + 0.4*np.exp(-0.5*((rel-8)/5.0)**2)
            sig += 0.02 * np.random.randn(N)

            all_ppg.append(sig)
            # accel must be sampled at the same rate (fs) as PPG, not WINDOW_S,
            # otherwise the arrays misalign and window slicing reads garbage/NaN
            all_accel.append(np.full(N, accel_mag) + 0.01*np.random.randn(N))
            labels.append(label)

    ppg_full   = np.concatenate(all_ppg)
    accel_full = np.concatenate(all_accel)
    return ppg_full, accel_full, labels


# =============================================================================
# MAIN — VALIDATE v2 ALGORITHM
# =============================================================================

if __name__ == '__main__':

    print('='*75)
    print('  ADAPTIVE BASELINE v2 VALIDATION')
    print('  (Dynamic Alpha + Anomaly Rejection + Context Gating)')
    print('='*75)

    ppg_signal, accel_signal, window_labels = build_test_scenario()
    window_samples = WINDOW_S * fs
    n_windows = len(ppg_signal) // window_samples
    total_min = len(ppg_signal) / fs / 60
    print(f'  Test scenario: {total_min:.1f} min — rest / anomaly / rest / motion / rest\n')

    baseline = AdaptiveBaselineV2()

    timestamps, raw_bpms, means, stds, statuses, alphas = [], [], [], [], [], []

    print(f'{"t(s)":>6s} {"label":>9s} {"BPM":>7s} {"accel":>7s} {"status":>17s} {"baseline":>9s} {"alpha":>7s}')

    for w in range(n_windows):
        start = w * window_samples
        end   = start + window_samples
        window_ppg   = ppg_signal[start:end]
        window_accel = np.mean(accel_signal[start:end])
        label        = window_labels[w]

        bpm = compute_window_bpm(window_ppg, fs)
        if bpm is None:
            continue

        alpha_before = baseline.current_alpha()
        mean, std, status = baseline.update(bpm, window_accel)

        t = start / fs
        timestamps.append(t); raw_bpms.append(bpm); means.append(mean)
        stds.append(std); statuses.append(status); alphas.append(alpha_before)

        if w % 4 == 0 or status != 'updated':
            print(f'{t:6.0f} {label:>9s} {bpm:7.1f} {window_accel:7.2f} '
                  f'{status:>17s} {mean:9.2f} {alpha_before:7.3f}')

    # ── Convergence check: time to reach std < 2.0 ─────────────────────────
    converge_idx = next((i for i, s in enumerate(stds) if s < 2.0), None)
    converge_time = timestamps[converge_idx] if converge_idx else None

    print(f'\n  Final baseline        : {baseline.mean:.2f} BPM')
    print(f'  Final std             : {baseline.std:.2f} BPM')
    if converge_time:
        print(f'  Time to converge (std<2.0): {converge_time:.0f}s ({converge_time/60:.1f} min)')
    n_rejected = sum(1 for s in statuses if s == 'anomaly_rejected')
    n_paused   = sum(1 for s in statuses if s == 'motion_paused')
    print(f'  Anomaly windows rejected : {n_rejected}')
    print(f'  Motion windows paused    : {n_paused}')

    # ── Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    status_colors = {'updated':'steelblue', 'anomaly_rejected':'red', 'motion_paused':'orange'}
    point_colors  = [status_colors[s] for s in statuses]

    ax = axes[0]
    ax.scatter(timestamps, raw_bpms, c=point_colors, s=25, zorder=3)
    ax.plot(timestamps, means, color='darkgreen', linewidth=2, label='Adaptive baseline')
    ax.fill_between(timestamps,
                    np.array(means)-2*np.array(stds),
                    np.array(means)+2*np.array(stds),
                    color='darkgreen', alpha=0.12, label='Personal range (+/-2 std)')
    for lbl, color in [('Updated','steelblue'),('Anomaly rejected','red'),('Motion paused','orange')]:
        ax.scatter([],[], c=color, label=lbl)
    ax.set_ylabel('BPM')
    ax.set_title('Adaptive Baseline v2 — Dynamic Alpha + Anomaly Rejection + Context Gating', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(timestamps, alphas, color='purple', linewidth=1.5)
    ax2.set_ylabel('Alpha (learning rate)')
    ax2.set_title('Dynamic Alpha Decay (fast warm-up -> stable tracking)', fontweight='bold')
    ax2.grid(alpha=0.3)

    ax3 = axes[2]
    ax3.plot(timestamps, stds, color='crimson', linewidth=1.5)
    ax3.axhline(2.0, color='gray', linestyle='--', linewidth=1, label='Convergence threshold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Baseline Std Dev')
    ax3.set_title('Baseline Uncertainty Over Time', fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR + '06_adaptive_baseline_results.png', dpi=120, bbox_inches='tight')
    print(f'\nPlot saved to {OUT_DIR}06_adaptive_baseline_results.png')

    print('\n' + '='*75)
    checks = []
    checks.append(('Fast warm-up (<3min)', converge_time is not None and converge_time < 180))
    checks.append(('Anomaly rejected at least once', n_rejected > 0))
    checks.append(('Motion paused at least once', n_paused > 0))
    checks.append(('Final baseline near true 70 BPM', abs(baseline.mean - 70) < 2))

    for check_name, passed in checks:
        print(f'  [{"PASS" if passed else "FAIL"}]  {check_name}')
    overall = all(c[1] for c in checks)
    print('='*75)
    print(f'  OVERALL: {"ALL PASSED" if overall else "SOME FAILED"}')
    print('='*75)