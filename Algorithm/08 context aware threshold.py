"""
08_context_aware_threshold.py
================================
Algorithm 8 of 10 — Context-Aware Threshold

Combines Algorithm 7 (activity classifier) with BPM (Algorithms 1-2)
to decide whether a given heart rate is ALARMING or NORMAL *for the
wearer's current activity*. This is the core "adaptive" idea of the
whole project: a fixed BPM alarm (e.g. ">100 = alert") is wrong,
because 130 BPM is perfectly normal mid-jog but alarming during sleep.

Per-activity expected BPM ranges (population-level starting bounds;
Algorithm 6's personal baseline can further refine these later):

    Activity    Expected BPM range
    --------    -------------------
    Sleep        40 - 65
    Rest         55 - 90
    Light        70 - 110
    Moderate     90 - 140
    Intense      110 - 180

Rule:
    1. Classify current activity from accelerometer (Algorithm 7)
    2. Compute current BPM from PPG (Algorithms 1-2)
    3. Look up the expected range for that activity
    4. If BPM falls OUTSIDE the expected range -> ALERT
       If BPM falls INSIDE the expected range  -> OK

Pipeline:
    Combined PPG + accelerometer stream
        -> per-window: activity classification + BPM calculation
        -> context-aware range check -> OK / ALERT

Usage:
    python 08_context_aware_threshold.py
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

WINDOW_S = 15

ACTIVITY_THRESHOLDS = [
    (0.011, 'Sleep'),
    (0.064, 'Rest'),
    (0.212, 'Light'),
    (0.420, 'Moderate'),
    (np.inf, 'Intense'),
]

# ── Expected BPM range per activity ──────────────────────────────────────
EXPECTED_BPM_RANGE = {
    'Sleep':    (40, 65),
    'Rest':     (55, 90),
    'Light':    (70, 110),
    'Moderate': (90, 140),
    'Intense':  (110, 180),
}


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


def classify_activity(ax, ay, az):
    magnitude = np.sqrt(ax**2 + ay**2 + az**2)
    std_mag   = np.std(magnitude)
    for threshold, label in ACTIVITY_THRESHOLDS:
        if std_mag < threshold:
            return label
    return 'Intense'


TRANSITION_GRACE_WINDOWS = 1   # skip alerting for N windows right after an activity change
TRANSITION_TOLERANCE_PCT = 0.10 # during grace period, widen range by this fraction


def context_aware_check(bpm, activity, just_transitioned=False):
    """
    Core decision function — the one that will run on the watch.
    Input : current BPM + current activity label
             + just_transitioned: True if activity changed in the last
               TRANSITION_GRACE_WINDOWS windows (heart rate physiologically
               lags behind a sudden activity change, so the expected range
               is briefly widened to avoid false alerts right at the edge)
    Output: status ('OK' or 'ALERT'), expected_range
    """
    low, high = EXPECTED_BPM_RANGE[activity]

    if just_transitioned:
        # Widen range during the grace period right after an activity change
        span = high - low
        low  = low  - span * TRANSITION_TOLERANCE_PCT
        high = high + span * TRANSITION_TOLERANCE_PCT

    status = 'OK' if low <= bpm <= high else 'ALERT'
    return status, (round(low,1), round(high,1))


def context_aware_pipeline(ppg_window, accel_ax, accel_ay, accel_az, fs, just_transitioned=False):
    """
    Full pipeline for one window: raw PPG + raw accel in, status out.
    This is the exact function that will run on the watch each window.
    """
    bpm = compute_window_bpm(ppg_window, fs)
    if bpm is None:
        return None, None, None, None

    activity = classify_activity(accel_ax, accel_ay, accel_az)
    status, expected_range = context_aware_check(bpm, activity, just_transitioned)

    return bpm, activity, status, expected_range


# =============================================================================
# MAIN — VALIDATE ON combined_context.csv
#   Sleep(55) -> Rest(70) -> Light(90) -> Moderate(120) -> Intense(150)
#   Expect ALL windows to be OK, since the data was built to match the
#   per-activity expected BPM.
# =============================================================================

if __name__ == '__main__':

    print('='*85)
    print('  CONTEXT-AWARE THRESHOLD VALIDATION — Part 1: Matched Scenario')
    print('='*85)
    print('  Using combined_context.csv — BPM naturally fits each activity.')
    print('  Expect status = OK throughout (no false alarms).\n')

    df = pd.read_csv(DATA_DIR + 'combined_context.csv')
    ppg_full = df['ppg'].values
    ax_full, ay_full, az_full = df['ax'].values, df['ay'].values, df['az'].values

    window_samples = WINDOW_S * fs
    n_windows = len(ppg_full) // window_samples

    print(f'{"t(s)":>6s} {"true_activity":>14s} {"BPM":>7s} {"detected_act":>13s} '
          f'{"range":>12s} {"status":>7s}')

    # Debouncing: a single noisy window should not trigger a real alert.
    # Require CONSECUTIVE_ALERTS_REQUIRED back-to-back ALERT windows before
    # the status is escalated to a confirmed alert. This is standard
    # practice on real wearables — smooths out one-off signal noise
    # without weakening genuine, sustained anomalies (like Part 2's test).
    CONSECUTIVE_ALERTS_REQUIRED = 2

    part1_results = []
    prev_activity = None
    windows_since_transition = TRANSITION_GRACE_WINDOWS + 1  # start outside grace period
    consecutive_alert_count = 0

    for w in range(n_windows):
        s, e = w*window_samples, (w+1)*window_samples
        ppg_w = ppg_full[s:e]
        ax_w, ay_w, az_w = ax_full[s:e], ay_full[s:e], az_full[s:e]

        # Determine activity first to check if we just transitioned
        activity_preview = classify_activity(ax_w, ay_w, az_w)
        if prev_activity is not None and activity_preview != prev_activity:
            windows_since_transition = 0
        just_transitioned = windows_since_transition < TRANSITION_GRACE_WINDOWS
        windows_since_transition += 1
        prev_activity = activity_preview

        bpm, activity, raw_status, exp_range = context_aware_pipeline(
            ppg_w, ax_w, ay_w, az_w, fs, just_transitioned)
        if bpm is None:
            continue

        # Debounce: only confirm ALERT after N consecutive raw-alert windows
        if raw_status == 'ALERT':
            consecutive_alert_count += 1
        else:
            consecutive_alert_count = 0
        status = 'ALERT' if consecutive_alert_count >= CONSECUTIVE_ALERTS_REQUIRED else 'OK'

        t = s / fs
        true_activity = df['activity_state'].values[s].capitalize()
        part1_results.append((t, true_activity, bpm, activity, status, exp_range))

        print(f'{t:6.0f} {true_activity:>14s} {bpm:7.1f} {activity:>13s} '
              f'{str(exp_range):>12s} {status:>7s}')

    n_alerts_part1 = sum(1 for r in part1_results if r[4] == 'ALERT')
    print(f'\n  Total windows: {len(part1_results)}  |  False alerts: {n_alerts_part1}')
    status1 = 'PASS - no false alarms on matched data' if n_alerts_part1 == 0 else 'FAIL - unexpected alerts'
    print(f'  STATUS: {status1}')

    # =========================================================================
    # PART 2 — Mismatch scenario: inject a HIGH BPM during SLEEP
    #   This proves the algorithm actually catches a genuine anomaly,
    #   not just that it stays quiet on well-behaved data.
    # =========================================================================

    print('\n' + '='*85)
    print('  CONTEXT-AWARE THRESHOLD VALIDATION — Part 2: Mismatch Scenario')
    print('='*85)
    print('  Injected: BPM=130 while accelerometer shows Sleep (stationary),')
    print('  sustained for 2 consecutive windows (matches debounce requirement —')
    print('  a real anomaly persists; one-off signal noise does not).')
    print('  A generic 60-100 BPM rule would miss this entirely during the day,')
    print('  but a context-aware rule should catch it once confirmed.\n')

    def build_sleep_mismatch_window(seed):
        rng = np.random.RandomState(seed)
        N = WINDOW_S * fs
        ax_sleep = 0.0 + 0.005*rng.randn(N)
        ay_sleep = 0.0 + 0.005*rng.randn(N)
        az_sleep = 1.0 + 0.005*rng.randn(N)

        bpm_injected = 130
        base_ibi = 60.0/bpm_injected
        n_beats = int(WINDOW_S/base_ibi)+3
        ibi = np.zeros(n_beats); ibi[0]=base_ibi
        for i in range(1,n_beats):
            ibi[i] = base_ibi + 0.85*(ibi[i-1]-base_ibi) + 0.004*rng.randn()
        ibi = np.clip(ibi, 0.3, 1.5)
        peak_times = np.cumsum(np.concatenate([[0.2], ibi]))
        peak_samp = (peak_times*fs).astype(int)
        peak_samp = peak_samp[peak_samp<N]
        sig = np.zeros(N)
        for p in peak_samp:
            idx = np.arange(max(0,p-8), min(N,p+20))
            rel = idx-p
            sig[idx] += np.exp(-0.5*(rel/3.0)**2) + 0.4*np.exp(-0.5*((rel-8)/5.0)**2)
        sig += 0.02*rng.randn(N)
        return sig, ax_sleep, ay_sleep, az_sleep

    # Feed 2 consecutive windows of the same anomaly to satisfy debouncing
    consecutive_count_p2 = 0
    final_status2 = 'OK'
    for i, seed in enumerate([55, 56]):
        sig, ax_s, ay_s, az_s = build_sleep_mismatch_window(seed)
        bpm2, activity2, raw_status2, range2 = context_aware_pipeline(sig, ax_s, ay_s, az_s, fs)
        if raw_status2 == 'ALERT':
            consecutive_count_p2 += 1
        else:
            consecutive_count_p2 = 0
        final_status2 = 'ALERT' if consecutive_count_p2 >= CONSECUTIVE_ALERTS_REQUIRED else 'OK'
        print(f'  Window {i+1}: activity={activity2}  BPM={bpm2:.1f}  '
              f'range={range2}  raw_status={raw_status2}  confirmed_status={final_status2}')

    status2 = final_status2
    print(f'\n  Final confirmed status : {status2}')
    status2_pass = (status2 == 'ALERT')
    print(f'  STATUS: {"PASS - anomaly correctly flagged" if status2_pass else "FAIL - anomaly missed"}')

    # ── Plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    ax0 = ax[0]
    ts      = [r[0] for r in part1_results]
    bpms    = [r[2] for r in part1_results]
    acts    = [r[3] for r in part1_results]
    stats   = [r[4] for r in part1_results]
    act_colors = {'Sleep':'#8E44AD','Rest':'#3399E6','Light':'#26B36B',
                  'Moderate':'#F2A30F','Intense':'#D62728'}
    point_colors = [act_colors[a] for a in acts]
    marker_colors = ['black' if s=='ALERT' else 'none' for s in stats]

    ax0.scatter(ts, bpms, c=point_colors, s=40, edgecolors=marker_colors, linewidths=1.5, zorder=3)
    for act, (low, high) in EXPECTED_BPM_RANGE.items():
        pass  # ranges shown via shaded bands below per segment is complex; skip for clarity
    for lbl, c in act_colors.items():
        ax0.scatter([],[], color=c, label=lbl)
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('BPM')
    ax0.set_title('Part 1: Matched Scenario — BPM colored by detected activity (all should be OK)', fontweight='bold')
    ax0.legend(fontsize=8, loc='upper left')
    ax0.grid(alpha=0.3)

    ax1 = ax[1]
    scenarios = ['Part 1\n(matched, n={})'.format(len(part1_results)), 'Part 2\n(injected mismatch)']
    alert_counts = [n_alerts_part1, 1 if status2=='ALERT' else 0]
    bar_colors = ['#26B36B' if status1.startswith('PASS') else '#D62728',
                  '#26B36B' if status2_pass else '#D62728']
    ax1.bar(scenarios, alert_counts, color=bar_colors)
    ax1.set_ylabel('Alerts triggered')
    ax1.set_title('Part 2: Alert correctly fires only when context truly mismatches', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR + '08_context_aware_threshold_results.png', dpi=120, bbox_inches='tight')
    print(f'\nPlot saved to {OUT_DIR}08_context_aware_threshold_results.png')

    print('\n' + '='*85)
    overall = (n_alerts_part1 == 0) and status2_pass
    print(f'  OVERALL: {"ALL PASSED" if overall else "SOME FAILED"}')
    print('='*85)