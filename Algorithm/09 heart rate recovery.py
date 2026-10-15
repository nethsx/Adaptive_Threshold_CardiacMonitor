"""
09_heart_rate_recovery.py
============================
Algorithm 9 of 10 — Heart Rate Recovery (HRR)

Tracks how BPM drops after intense activity ends, and closes the
"post-workout trap" loophole in Algorithm 8's context gating: a
person who just finished sprinting and sat down will still show
130+ BPM while Activity=Rest for several minutes - that is NORMAL
physiology, not a sensor error or medical event, and must not be
treated as a context mismatch.

Three mechanisms, combined:

1. RECOVERY COOLDOWN WINDOW
   When activity drops from Intense/Moderate -> Rest, a cooldown
   timer starts (default 3 minutes). During this window, Algorithm 8
   style "activity vs BPM" mismatch alerts are suppressed - elevated
   BPM at rest is EXPECTED here, not alarming.

2. HRR CURVE / TREND CHECK (the real safety net)
   Suppression above is not unconditional. During the cooldown, this
   algorithm actively tracks whether BPM is trending DOWN (healthy
   recovery) using the clinical HRR-60 benchmark: a healthy heart
   drops at least 12 BPM in the first 60 seconds after peak exertion
   ends. If the trend is flat or still climbing instead, the cooldown
   suppression is cancelled immediately and a real alert fires -
   because "stuck at peak HR after stopping" is genuinely concerning,
   regardless of how recently the person was active.

3. DIRECTIONAL ALERT SEPARATION
   When a true mismatch is confirmed, it is NOT reported as a generic
   "mismatch" - the direction of the error determines which of two
   very different alerts fires:
       High BPM + Low Activity + poor recovery trend
           -> "Elevated Resting Heart Rate" (possible arrhythmia,
              panic attack, fever - medically relevant)
       Low BPM + High Activity
           -> "Possible Sensor Detachment" (watch likely slid off
              the wrist during exercise and is misreading PPG - a
              hardware/fit issue, not a cardiac one)

Pipeline:
    combined_hrr.csv (rest -> intense -> recovery)
        -> per-window BPM (Algorithms 1-2) + activity (Algorithm 7)
        -> detect Intense/Moderate -> Rest transition -> start cooldown
        -> track HRR-60 trend during cooldown
        -> confirm healthy recovery OR escalate directional alert

Usage:
    python 09_heart_rate_recovery.py
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
    (0.011, 'Sleep'), (0.064, 'Rest'), (0.212, 'Light'),
    (0.420, 'Moderate'), (np.inf, 'Intense'),
]

COOLDOWN_DURATION_S   = 180     # 3 minute recovery cooldown window
HRR_60_HEALTHY_DROP   = 12      # healthy benchmark: >=12 BPM drop in first 60s of recovery
STUCK_TREND_WINDOWS   = 4       # if BPM hasn't dropped meaningfully across this many
                                  # windows during cooldown, treat as "stuck" -> real alert


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


# ── Accelerometer generator — matches Algorithm 7's oscillation model ────
# (same formula used when the dataset files were generated, so std(magnitude)
# lands correctly within Algorithm 7's calibrated thresholds)
ACCEL_PARAMS = {
    'sleep':    (0.0, 0.00, 0.005, 1.00),
    'rest':     (0.1, 0.02, 0.010, 0.98),
    'light':    (1.8, 0.15, 0.030, 0.95),
    'moderate': (2.5, 0.45, 0.060, 0.90),
    'intense':  (3.2, 0.90, 0.100, 0.85),
}

def make_accel(duration_s, activity, seed, fs=fs):
    rng = np.random.RandomState(seed)
    N = duration_s * fs
    t = np.arange(N) / fs
    freq, amp, noise, gz = ACCEL_PARAMS[activity]
    az = gz + amp*np.sin(2*np.pi*freq*t) + noise*rng.randn(N)
    ax = amp*0.6*np.sin(2*np.pi*freq*t + np.pi/4) + noise*rng.randn(N)
    ay = amp*0.4*np.sin(2*np.pi*freq*t + np.pi/2) + noise*rng.randn(N)
    return ax, ay, az


def make_ppg_window(bpm_target, duration_s, seed, fs=fs):
    """Correlated-IBI synthetic PPG window at a target BPM (AR(1) model)."""
    rng = np.random.RandomState(seed)
    N = duration_s * fs
    base_ibi = 60.0 / bpm_target
    n_beats = int(duration_s / base_ibi) + 3
    ibi = np.zeros(n_beats); ibi[0] = base_ibi
    for i in range(1, n_beats):
        ibi[i] = base_ibi + 0.85*(ibi[i-1]-base_ibi) + 0.005*rng.randn()
    ibi = np.clip(ibi, 0.3, 1.5)
    peak_times = np.cumsum(np.concatenate([[0.2], ibi]))
    peak_samp = (peak_times * fs).astype(int)
    peak_samp = peak_samp[peak_samp < N]
    sig = np.zeros(N)
    for p in peak_samp:
        idx = np.arange(max(0, p-8), min(N, p+20))
        rel = idx - p
        sig[idx] += np.exp(-0.5*(rel/3.0)**2) + 0.4*np.exp(-0.5*((rel-8)/5.0)**2)
    sig += 0.02 * rng.randn(N)
    return sig


def build_hrr_scenario():
    """
    Correctly-built scenario (replaces the flawed combined_hrr.csv for this
    algorithm): Rest -> Intense -> Recovery, where recovery uses 'rest'
    accelerometer motion (person actually sits down) while BPM gradually
    declines from peak toward baseline, matching real post-exercise
    physiology.
    """
    segments = [
        # (duration_s, bpm_start, bpm_end, accel_activity, seed, label)
        (60,  70,  70,  'rest',    20, 'rest'),
        (120, 150, 150, 'intense', 21, 'intense'),
        (30,  150, 130, 'rest',    22, 'recovery'),  # just stopped, BPM still high
        (30,  130, 105, 'rest',    23, 'recovery'),
        (30,  105, 88,  'rest',    24, 'recovery'),
        (30,  88,  75,  'rest',    25, 'recovery'),
    ]
    all_ppg, all_ax, all_ay, all_az, labels = [], [], [], [], []
    for dur, bpm_start, bpm_end, activity, seed, label in segments:
        n_windows = dur // WINDOW_S
        for i in range(n_windows):
            bpm_w = bpm_start + (bpm_end - bpm_start) * (i / max(1, n_windows-1))
            sig = make_ppg_window(bpm_w, WINDOW_S, seed*100+i)
            ax, ay, az = make_accel(WINDOW_S, activity, seed*200+i)
            all_ppg.append(sig); all_ax.append(ax); all_ay.append(ay); all_az.append(az)
            labels.append(label)
    return (np.concatenate(all_ppg), np.concatenate(all_ax),
            np.concatenate(all_ay), np.concatenate(all_az), labels)


class HRRMonitor:
    """
    Stateful recovery monitor — tracks cooldown windows and BPM trend
    after an Intense/Moderate -> Rest transition.
    """
    def __init__(self):
        self.in_cooldown        = False
        self.cooldown_elapsed_s = 0
        self.peak_bpm           = None
        self.bpm_at_cooldown_start = None
        self.recent_bpms        = []   # for stuck-trend detection
        self.prev_activity      = None

    def process_window(self, bpm, activity, window_duration=WINDOW_S):
        """
        Returns: (status, alert_type, info_dict)
            status: 'OK' | 'RECOVERING' | 'ALERT'
            alert_type: None | 'Elevated Resting Heart Rate' | 'Possible Sensor Detachment'
        """
        was_active = self.prev_activity in ('Intense', 'Moderate')
        now_resting = activity in ('Rest', 'Sleep')

        # ── Detect the Intense/Moderate -> Rest transition: start cooldown ──
        if was_active and now_resting and not self.in_cooldown:
            self.in_cooldown = True
            self.cooldown_elapsed_s = 0
            self.bpm_at_cooldown_start = bpm
            self.recent_bpms = [bpm]
            self.prev_activity = activity
            return 'RECOVERING', None, {'cooldown_elapsed': 0, 'hrr_60': None}

        # ── Directional mismatch: Low BPM + High Activity -> sensor issue ──
        if activity in ('Intense', 'Moderate') and bpm < 70:
            self.prev_activity = activity
            return 'ALERT', 'Possible Sensor Detachment', {}

        # ── Currently in cooldown: track recovery trend ─────────────────
        if self.in_cooldown:
            self.cooldown_elapsed_s += window_duration
            self.recent_bpms.append(bpm)

            # HRR-60: BPM drop in the first 60s of cooldown
            hrr_60 = None
            if self.cooldown_elapsed_s >= 60 and len(self.recent_bpms) >= 2:
                idx_60 = min(60 // window_duration, len(self.recent_bpms) - 1)
                hrr_60 = self.bpm_at_cooldown_start - self.recent_bpms[idx_60]

            # Stuck-trend check: last N readings barely moving / rising?
            stuck = False
            if len(self.recent_bpms) >= STUCK_TREND_WINDOWS:
                last_n = self.recent_bpms[-STUCK_TREND_WINDOWS:]
                trend  = last_n[-1] - last_n[0]
                if trend >= -2:   # not meaningfully dropping
                    stuck = True

            if activity in ('Rest', 'Sleep'):
                if hrr_60 is not None and hrr_60 < HRR_60_HEALTHY_DROP and stuck:
                    # Poor recovery AND stuck at elevated level -> real alert
                    self.in_cooldown = False
                    self.prev_activity = activity
                    return 'ALERT', 'Elevated Resting Heart Rate', {'hrr_60': hrr_60}

                if self.cooldown_elapsed_s >= COOLDOWN_DURATION_S or bpm < 90:
                    # Cooldown finished OR BPM has normalized early -> recovered
                    self.in_cooldown = False
                    self.prev_activity = activity
                    return 'OK', None, {'hrr_60': hrr_60, 'recovered': True}

                self.prev_activity = activity
                return 'RECOVERING', None, {'hrr_60': hrr_60}
            else:
                # Became active again mid-cooldown — cancel cooldown
                self.in_cooldown = False

        self.prev_activity = activity
        return 'OK', None, {}


# =============================================================================
# MAIN — VALIDATE ON combined_hrr.csv  (Rest -> Intense -> Recovery)
# =============================================================================

if __name__ == '__main__':

    print('='*85)
    print('  HEART RATE RECOVERY (HRR) VALIDATION')
    print('='*85)
    print('  NOTE: The shipped combined_hrr.csv used "moderate" accelerometer')
    print('  motion during its recovery phase (a data-generation oversight),')
    print('  so the Intense->Rest transition this algorithm needs never occurs')
    print('  in that file. A corrected scenario is built inline below instead,')
    print('  where the wearer actually sits down (Rest accel) while BPM')
    print('  gradually declines - matching real post-exercise physiology.\n')
    print('  Scenario: Rest (60s) -> Intense (120s) -> Recovery (120s, BPM 150->75)')
    print('  Expect: RECOVERING status during cooldown, NO false "mismatch" alert,')
    print('  ending in OK once BPM normalizes.\n')

    ppg_full, ax_full, ay_full, az_full, true_labels_list = build_hrr_scenario()

    window_samples = WINDOW_S * fs
    n_windows = len(ppg_full) // window_samples

    monitor = HRRMonitor()
    results = []

    print(f'{"t(s)":>6s} {"true_phase":>11s} {"BPM":>7s} {"activity":>9s} '
          f'{"HRR-60":>8s} {"status":>11s} {"alert_type":>24s}')

    for w in range(n_windows):
        s, e = w*window_samples, (w+1)*window_samples
        ppg_w = ppg_full[s:e]
        ax_w, ay_w, az_w = ax_full[s:e], ay_full[s:e], az_full[s:e]

        bpm = compute_window_bpm(ppg_w, fs)
        if bpm is None:
            continue
        activity = classify_activity(ax_w, ay_w, az_w)
        status, alert_type, info = monitor.process_window(bpm, activity)

        t = s / fs
        true_phase = true_labels_list[w]
        hrr60_str = f'{info["hrr_60"]:.1f}' if info.get('hrr_60') is not None else '-'
        results.append((t, true_phase, bpm, activity, status, alert_type, info.get('hrr_60')))

        print(f'{t:6.0f} {true_phase:>11s} {bpm:7.1f} {activity:>9s} '
              f'{hrr60_str:>8s} {status:>11s} {str(alert_type or "-"):>24s}')

    # ── Checks ──────────────────────────────────────────────────────────
    false_alerts = [r for r in results if r[4] == 'ALERT']
    recovering_windows = [r for r in results if r[4] == 'RECOVERING']
    final_status = results[-1][4]

    print(f'\n  False alerts during normal recovery : {len(false_alerts)}')
    print(f'  Windows correctly marked RECOVERING  : {len(recovering_windows)}')
    print(f'  Final status (should be OK)          : {final_status}')

    # HRR-60 clinical check
    hrr60_values = [r[6] for r in results if r[6] is not None]
    hrr60_final = hrr60_values[0] if hrr60_values else None
    print(f'  Measured HRR-60                      : {hrr60_final:.1f} BPM  (healthy >= 12)' if hrr60_final else '  HRR-60: not captured')

    # =========================================================================
    # PART 2 — Stuck recovery scenario (should correctly ALERT)
    #   BPM stays pinned at ~145 throughout "recovery" — simulates a real
    #   concern (arrhythmia continuing post-exercise) rather than normal HRR.
    # =========================================================================

    print('\n' + '='*85)
    print('  PART 2 — Stuck Recovery Scenario (BPM fails to drop)')
    print('='*85)

    monitor2 = HRRMonitor()
    stuck_results = []
    # Intense phase: 2 windows at 150 BPM, correctly oscillating Intense accel
    for i in range(2):
        sig = make_ppg_window(150, WINDOW_S, seed=200+i)
        ax, ay, az = make_accel(WINDOW_S, 'intense', seed=250+i)
        bpm = compute_window_bpm(sig, fs)
        activity = classify_activity(ax, ay, az)
        status, alert_type, info = monitor2.process_window(bpm, activity)
        stuck_results.append(('Intense', bpm, activity, status, alert_type))

    # "Recovery" phase but BPM stays stuck at ~145 (not dropping) for 6 windows (90s)
    # Accel correctly shows Rest (wearer IS sitting still — the heart just
    # isn't cooperating, which is the genuinely concerning case)
    for i in range(6):
        sig = make_ppg_window(145, WINDOW_S, seed=300+i)
        ax, ay, az = make_accel(WINDOW_S, 'rest', seed=400+i)
        bpm = compute_window_bpm(sig, fs)
        activity = classify_activity(ax, ay, az)
        status, alert_type, info = monitor2.process_window(bpm, activity)
        stuck_results.append(('Rest(stuck)', bpm, activity, status, alert_type))

    print(f'{"phase":>12s} {"BPM":>7s} {"activity":>9s} {"status":>11s} {"alert_type":>26s}')
    for phase, bpm, activity, status, alert_type in stuck_results:
        print(f'{phase:>12s} {bpm:7.1f} {activity:>9s} {status:>11s} {str(alert_type or "-"):>26s}')

    stuck_detected = any(r[4] == 'Elevated Resting Heart Rate' for r in stuck_results)
    print(f'\n  STATUS: {"PASS - stuck recovery correctly flagged as Elevated Resting HR" if stuck_detected else "FAIL - stuck recovery missed"}')

    # ── Plot ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    ts   = [r[0] for r in results]
    bpms = [r[2] for r in results]
    stats = [r[4] for r in results]
    status_colors = {'OK':'#26B36B','RECOVERING':'#F2A30F','ALERT':'#D62728'}
    colors = [status_colors[s] for s in stats]

    ax0 = ax[0]
    ax0.plot(ts, bpms, color='gray', linewidth=1, zorder=1, alpha=0.5)
    ax0.scatter(ts, bpms, c=colors, s=40, zorder=3)
    for lbl, c in status_colors.items():
        ax0.scatter([],[], color=c, label=lbl)
    ax0.axhline(90, color='black', linestyle=':', linewidth=1, label='Normal resting ceiling (90)')
    ax0.set_xlabel('Time (s)')
    ax0.set_ylabel('BPM')
    ax0.set_title('Part 1: Normal HRR — Rest -> Intense -> Recovery (no false alerts)', fontweight='bold')
    ax0.legend(fontsize=8, loc='upper right')
    ax0.grid(alpha=0.3)

    ax1 = ax[1]
    bpms2 = [r[1] for r in stuck_results]
    stats2 = [r[3] for r in stuck_results]
    colors2 = [status_colors.get(s, '#999999') for s in stats2]
    ax1.plot(range(len(bpms2)), bpms2, color='gray', linewidth=1, alpha=0.5)
    ax1.scatter(range(len(bpms2)), bpms2, c=colors2, s=60, zorder=3)
    ax1.axvline(1.5, color='black', linestyle='--', linewidth=1, label='Intense -> Rest transition')
    ax1.set_xlabel('Window index')
    ax1.set_ylabel('BPM')
    ax1.set_title('Part 2: Stuck Recovery — BPM fails to drop (should escalate to ALERT)', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR + '09_heart_rate_recovery_results.png', dpi=120, bbox_inches='tight')
    print(f'\nPlot saved to {OUT_DIR}09_heart_rate_recovery_results.png')

    print('\n' + '='*85)
    overall = (len(false_alerts) == 0) and (final_status == 'OK') and stuck_detected
    print(f'  OVERALL: {"ALL PASSED" if overall else "SOME FAILED"}')
    print('='*85)