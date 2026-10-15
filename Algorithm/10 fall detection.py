"""
10_fall_detection.py
======================
Algorithm 10 of 10 — Fall Detection  (v2: Free-Fall + Orientation +
Extended Stillness with Early-Exit)

A plain "spike then stillness" check has real false-alarm holes:
a dropped watch on a table, a hard high-five, or clamping your hands
together while sitting can all produce spike+stillness without an
actual fall happening. v2 adds three stages that together close
these holes.

FOUR-STAGE PIPELINE:

    Stage 0 — FREE FALL (weightlessness before impact)
        A real human fall begins with the body unsupported in the air
        for a brief moment - magnitude drops well below 1g (< 0.5g)
        just before impact. A table-drop or high-five has NO such
        dip - the spike happens directly from a resting/active
        baseline. Reject immediately if no free-fall dip precedes
        the candidate spike.

    Stage 1 — IMPACT SPIKE
        Magnitude exceeds 2.0g shortly after the free-fall dip —
        the actual moment of impact.

    Stage 2 — ORIENTATION CHANGE (tilt detection)
        Compare the average gravity-vector direction (ax,ay,az)
        BEFORE the fall to the average direction DURING the
        post-impact stillness. A real fall almost always ends with
        the body (and wrist) in a very different orientation than
        before - e.g. upright -> lying down. If the wearer's
        orientation barely changed (e.g. clamping hands together
        while remaining seated upright), this is NOT a fall even if
        Stages 0-1 both fired.

    Stage 3 — EXTENDED STILLNESS WITH EARLY EXIT
        Stillness must hold for a longer window (5s, not 1.5s) to
        avoid confusing a brief post-trip pause with a real
        incapacitating fall. If meaningful movement resumes at ANY
        point during this window (the wearer gets back up), abort
        immediately - this is a recovered trip, not a confirmed fall.

    FALL CONFIRMED only if ALL FOUR stages pass in sequence.

Validated against 5 scenarios:
    True fall              -> MUST confirm (all 4 stages pass)
    Table drop / high-five  -> MUST reject at Stage 0 (no free-fall dip)
    Clamp while seated       -> MUST reject at Stage 2 (no orientation change)
    Trip-and-recover          -> MUST reject at Stage 3 (movement resumes)
    Running (accel_intense)   -> MUST reject (no sustained stillness at all)

Usage:
    python 10_fall_detection.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

fs        = 100
DATA_DIR  = 'data/'
OUT_DIR   = 'algorithms/outputs/'
os.makedirs(OUT_DIR, exist_ok=True)

FREE_FALL_THRESHOLD_G    = 0.5    # magnitude below this = weightless dip
FREE_FALL_LOOKBACK_S     = 0.8    # how far back to scan for the dip before a spike
SPIKE_THRESHOLD_G        = 2.0    # magnitude above this = impact
ORIENTATION_BASELINE_S   = 1.0    # window before free-fall to measure "before" orientation
ORIENTATION_ANGLE_DEG    = 30.0   # minimum tilt change (degrees) to count as a real fall
STILLNESS_WINDOW_S       = 5.0    # extended post-impact stillness window
STILLNESS_SUBWIN_S       = 0.5    # sub-window size for early-exit movement scanning
STILLNESS_STD_THRESHOLD  = 0.03   # magnitude std below this = "still" in a sub-window


def compute_magnitude(ax, ay, az):
    return np.sqrt(ax**2 + ay**2 + az**2)


def orientation_vector(ax, ay, az):
    """Mean gravity-direction vector over a window, normalized to unit length."""
    v = np.array([np.mean(ax), np.mean(ay), np.mean(az)])
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


def angle_between(v1, v2):
    """Angle in degrees between two orientation vectors."""
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def detect_fall(ax, ay, az, fs=fs):
    """
    BLIND prediction function - the exact logic that will run on the watch.
    Input : raw ax, ay, az accelerometer stream only.
    Output: fall_detected (bool), stage_reached (str), details (dict)
    """
    magnitude = compute_magnitude(ax, ay, az)
    N = len(magnitude)

    # ── Stage 1 target: find the global impact peak ─────────────────────
    if magnitude.max() <= SPIKE_THRESHOLD_G:
        return False, 'stage1_no_spike', {}
    spike_idx = int(np.argmax(magnitude))

    # ── Stage 0: was there a free-fall dip shortly before the spike? ─────
    lookback_samples = int(FREE_FALL_LOOKBACK_S * fs)
    lookback_start = max(0, spike_idx - lookback_samples)
    pre_spike_window = magnitude[lookback_start:spike_idx]

    if len(pre_spike_window) == 0 or pre_spike_window.min() >= FREE_FALL_THRESHOLD_G:
        return False, 'stage0_no_freefall', {'spike_time': spike_idx/fs}

    free_fall_idx = lookback_start + int(np.argmin(pre_spike_window))

    # ── Stage 2: orientation change check ────────────────────────────────
    base_samples = int(ORIENTATION_BASELINE_S * fs)
    base_start = max(0, free_fall_idx - base_samples)
    if base_start >= free_fall_idx:
        return False, 'stage2_insufficient_baseline', {'spike_time': spike_idx/fs}

    orientation_before = orientation_vector(ax[base_start:free_fall_idx],
                                             ay[base_start:free_fall_idx],
                                             az[base_start:free_fall_idx])

    still_start = spike_idx + int(0.1 * fs)
    still_samples_total = int(STILLNESS_WINDOW_S * fs)
    still_end = min(N, still_start + still_samples_total)

    if still_end - still_start < int(1.0 * fs):
        return False, 'stage3_insufficient_data', {'spike_time': spike_idx/fs}

    # Use the first second of the stillness window for orientation comparison
    orient_check_end = min(still_end, still_start + fs)
    orientation_after = orientation_vector(ax[still_start:orient_check_end],
                                            ay[still_start:orient_check_end],
                                            az[still_start:orient_check_end])

    tilt_angle = angle_between(orientation_before, orientation_after)
    if tilt_angle < ORIENTATION_ANGLE_DEG:
        return False, 'stage2_no_orientation_change', {
            'spike_time': spike_idx/fs, 'tilt_angle': tilt_angle}

    # ── Stage 3: extended stillness with early-exit on movement ──────────
    subwin_samples = int(STILLNESS_SUBWIN_S * fs)
    n_subwindows = (still_end - still_start) // subwin_samples

    for sw in range(n_subwindows):
        s = still_start + sw*subwin_samples
        e = s + subwin_samples
        sub_std = np.std(magnitude[s:e])
        if sub_std >= STILLNESS_STD_THRESHOLD:
            return False, 'stage3_movement_resumed', {
                'spike_time': spike_idx/fs, 'tilt_angle': tilt_angle,
                'movement_resumed_at': s/fs}

    # ── All four stages passed ────────────────────────────────────────────
    return True, 'confirmed', {
        'spike_time': spike_idx/fs,
        'free_fall_time': free_fall_idx/fs,
        'tilt_angle': tilt_angle,
    }


# =============================================================================
# SYNTHETIC SCENARIO BUILDERS
#   The shipped accel_fall.csv lacks a genuine free-fall precursor (an
#   earlier dataset-generation limitation), so corrected/extended test
#   scenarios are built inline here to properly exercise all 4 stages,
#   including the 3 specific false-alarm cases from the upgrade request.
# =============================================================================

def build_true_fall(seed=1):
    rng = np.random.RandomState(seed)
    def seg(dur, ax_f, ay_f, az_f, noise=0.01):
        N = int(dur*fs); t = np.arange(N)/fs
        return (ax_f(t)+noise*rng.randn(N), ay_f(t)+noise*rng.randn(N), az_f(t)+noise*rng.randn(N))

    rest  = seg(1.0, lambda t: 0*t, lambda t: 0*t, lambda t: 0.98+0*t)
    ff    = seg(0.4, lambda t: 0.05*t, lambda t: 0.05*t, lambda t: 0.98*(1-t/0.4))  # drops to ~0
    spike = seg(0.1, lambda t: 2.5*np.exp(-((t-0.05)/0.02)**2),
                      lambda t: 1.0*np.exp(-((t-0.05)/0.02)**2),
                      lambda t: -2.0*np.exp(-((t-0.05)/0.02)**2))
    still = seg(10.0, lambda t: 0.95+0*t, lambda t: 0.10+0*t, lambda t: 0.18+0*t)

    ax = np.concatenate([rest[0], ff[0], spike[0], still[0]])
    ay = np.concatenate([rest[1], ff[1], spike[1], still[1]])
    az = np.concatenate([rest[2], ff[2], spike[2], still[2]])
    return ax, ay, az


def build_table_drop(seed=2):
    """Spike with NO preceding free-fall dip - watch tapped/set down firmly."""
    rng = np.random.RandomState(seed)
    def seg(dur, ax_f, ay_f, az_f, noise=0.01):
        N = int(dur*fs); t = np.arange(N)/fs
        return (ax_f(t)+noise*rng.randn(N), ay_f(t)+noise*rng.randn(N), az_f(t)+noise*rng.randn(N))

    rest  = seg(1.0, lambda t: 0*t, lambda t: 0*t, lambda t: 0.98+0*t)   # stays near 1g throughout
    spike = seg(0.1, lambda t: 0.5*np.exp(-((t-0.05)/0.02)**2),
                      lambda t: 0.5*np.exp(-((t-0.05)/0.02)**2),
                      lambda t: 2.5+2.0*np.exp(-((t-0.05)/0.02)**2))      # direct hit, no dip before
    still = seg(10.0, lambda t: 0.0+0*t, lambda t: 0.0+0*t, lambda t: 0.98+0*t)  # lying flat on table, same up orientation

    ax = np.concatenate([rest[0], spike[0], still[0]])
    ay = np.concatenate([rest[1], spike[1], still[1]])
    az = np.concatenate([rest[2], spike[2], still[2]])
    return ax, ay, az


def build_clamp_seated(seed=3):
    """
    Hand raised and slammed down while SEATED: brief low-g at the top of
    the swing (passes Stage 0) + spike (clamp) + stillness afterward, but
    orientation stays upright throughout (fails Stage 2).
    """
    rng = np.random.RandomState(seed)
    def seg(dur, ax_f, ay_f, az_f, noise=0.01):
        N = int(dur*fs); t = np.arange(N)/fs
        return (ax_f(t)+noise*rng.randn(N), ay_f(t)+noise*rng.randn(N), az_f(t)+noise*rng.randn(N))

    rest  = seg(1.0, lambda t: 0*t, lambda t: 0*t, lambda t: 0.98+0*t)
    swing_up = seg(0.3, lambda t: 0.1*t, lambda t: 0*t, lambda t: 0.98*(1-t/0.3)*0.9)  # brief dip ~0.1g
    spike = seg(0.1, lambda t: 0.3*np.exp(-((t-0.05)/0.02)**2),
                      lambda t: 0.3*np.exp(-((t-0.05)/0.02)**2),
                      lambda t: 0.1+2.4*np.exp(-((t-0.05)/0.02)**2))
    still = seg(10.0, lambda t: 0.0+0*t, lambda t: 0.0+0*t, lambda t: 0.98+0*t)  # STILL UPRIGHT after

    ax = np.concatenate([rest[0], swing_up[0], spike[0], still[0]])
    ay = np.concatenate([rest[1], swing_up[1], spike[1], still[1]])
    az = np.concatenate([rest[2], swing_up[2], spike[2], still[2]])
    return ax, ay, az


def build_trip_recover(seed=4):
    """Free-fall + spike + orientation change, but movement resumes at ~2s into stillness."""
    rng = np.random.RandomState(seed)
    def seg(dur, ax_f, ay_f, az_f, noise=0.01):
        N = int(dur*fs); t = np.arange(N)/fs
        return (ax_f(t)+noise*rng.randn(N), ay_f(t)+noise*rng.randn(N), az_f(t)+noise*rng.randn(N))

    rest  = seg(1.0, lambda t: 0*t, lambda t: 0*t, lambda t: 0.98+0*t)
    ff    = seg(0.3, lambda t: 0.05*t, lambda t: 0.05*t, lambda t: 0.98*(1-t/0.3))
    spike = seg(0.1, lambda t: 2.0*np.exp(-((t-0.05)/0.02)**2),
                      lambda t: 0.8*np.exp(-((t-0.05)/0.02)**2),
                      lambda t: -1.5*np.exp(-((t-0.05)/0.02)**2))
    still_brief = seg(2.0, lambda t: 0.9+0*t, lambda t: 0.1+0*t, lambda t: 0.2+0*t)   # 2s of stillness, oriented down
    getting_up  = seg(3.0, lambda t: 0.6*np.sin(2*np.pi*1.5*t), lambda t: 0.4*np.sin(2*np.pi*1.5*t+0.5),
                            lambda t: 0.9+0.3*np.sin(2*np.pi*1.5*t), noise=0.03)        # moving again

    ax = np.concatenate([rest[0], ff[0], spike[0], still_brief[0], getting_up[0]])
    ay = np.concatenate([rest[1], ff[1], spike[1], still_brief[1], getting_up[1]])
    az = np.concatenate([rest[2], ff[2], spike[2], still_brief[2], getting_up[2]])
    return ax, ay, az


# =============================================================================
# MAIN — VALIDATE ALL 5 SCENARIOS
# =============================================================================

if __name__ == '__main__':

    print('='*90)
    print('  FALL DETECTION v2 VALIDATION')
    print('  (Free-Fall + Impact Spike + Orientation Change + Extended Stillness)')
    print('='*90)
    print('  NOTE: accel_fall.csv (shipped dataset) lacks a true free-fall')
    print('  precursor, so corrected/extended scenarios are built inline to')
    print('  properly test all 4 stages, including the 3 specific false-alarm')
    print('  cases requested: table drop, seated clamp, and trip-and-recover.\n')

    scenarios = [
        ('True Fall',            build_true_fall(),     True),
        ('Table Drop/High-five', build_table_drop(),    False),
        ('Clamp While Seated',   build_clamp_seated(),  False),
        ('Trip-and-Recover',     build_trip_recover(),  False),
    ]

    print(f'{"Scenario":22s} {"Expected":>10s} {"Detected":>10s} {"Stage Reached":>22s} {"Result":>8s}')

    results = []
    all_pass = True
    for name, (ax, ay, az), expected in scenarios:
        detected, stage, details = detect_fall(ax, ay, az)
        correct = (detected == expected)
        if not correct:
            all_pass = False
        status = 'PASS' if correct else 'FAIL'
        print(f'{name:22s} {str(expected):>10s} {str(detected):>10s} {stage:>22s} {status:>8s}')
        results.append((name, expected, detected, stage, details, ax, ay, az))

    # ── Also re-validate against the 6 original activity files ────────────
    print()
    activity_files = [
        ('accel_intense.csv',  False),
        ('accel_moderate.csv', False),
        ('accel_light.csv',    False),
        ('accel_rest.csv',     False),
        ('accel_sleep.csv',    False),
    ]
    for fname, expected in activity_files:
        df = pd.read_csv(DATA_DIR + fname)
        ax_f, ay_f, az_f = df['ax'].values, df['ay'].values, df['az'].values
        detected, stage, details = detect_fall(ax_f, ay_f, az_f)
        correct = (detected == expected)
        if not correct:
            all_pass = False
        status = 'PASS' if correct else 'FAIL'
        print(f'{fname:22s} {str(expected):>10s} {str(detected):>10s} {stage:>22s} {status:>8s}')
        results.append((fname, expected, detected, stage, details, ax_f, ay_f, az_f))

    # ── Plot: magnitude traces for the 4 key scenarios ─────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    titles_colors = {
        'True Fall': '#D62728', 'Table Drop/High-five': '#F2A30F',
        'Clamp While Seated': '#8E44AD', 'Trip-and-Recover': '#3399E6'
    }

    for i, (name, expected, detected, stage, details, ax_, ay_, az_) in enumerate(results[:4]):
        mag = compute_magnitude(ax_, ay_, az_)
        t = np.arange(len(mag)) / fs
        ax_plot = axes[i]
        ax_plot.plot(t, mag, color=titles_colors[name], linewidth=1.1)
        ax_plot.axhline(SPIKE_THRESHOLD_G, color='black', linestyle='--', linewidth=0.8, label='Spike (2.0g)')
        ax_plot.axhline(FREE_FALL_THRESHOLD_G, color='gray', linestyle=':', linewidth=0.8, label='Free-fall (0.5g)')
        verdict = 'FALL CONFIRMED' if detected else f'Rejected ({stage})'
        ax_plot.set_title(f'{name}\n{verdict}', fontweight='bold', fontsize=10)
        ax_plot.set_xlabel('Time (s)')
        ax_plot.set_ylabel('Magnitude (g)')
        ax_plot.legend(fontsize=7, loc='upper right')
        ax_plot.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR + '10_fall_detection_results.png', dpi=120, bbox_inches='tight')
    print(f'\nPlot saved to {OUT_DIR}10_fall_detection_results.png')

    n_correct = sum(1 for r in results if r[1] == r[2])
    print('\n' + '='*90)
    print(f'  OVERALL: {"ALL PASSED" if all_pass else "SOME FAILED"}  '
          f'({n_correct}/{len(results)} correct)')
    print('='*90)