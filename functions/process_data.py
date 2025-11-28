import numpy as np

# REAL DATA
def extract_threshold_waveforms(signal, threshold, fs):
    """
    Extracts spike-aligned waveforms and timing information.
    Requires the voltage trace to cross 0 before the next crossing time is detected.

    Parameters
    ----------
    signal : 1D array
        Voltage trace.
    threshold : float
        Threshold value (positive).
    fs : float
        Sampling frequency in Hz (e.g., 12500).

    Returns
    -------
    crossing_times : (num_crossings,) array
        Crossing times in seconds.
    """
    samples = int(round(0.001 * fs))       # 1 ms before and after
    window = np.arange(-samples, samples + 1)
    num_samples = len(window)

    # Detect negative threshold crossings (downward crossings from above -threshold to below -threshold)
    neg_threshold_crossings = np.where(np.diff(np.concatenate([[0], signal < -threshold])) == 1)[0]
    
    # Detect zero crossings (signal crosses from negative to positive or positive to negative)
    # Find where consecutive samples have opposite signs (product is negative)
    zero_crossings = np.where(signal[:-1] * signal[1:] < 0)[0]
    
    # Filter threshold crossings: only keep those where a zero crossing occurred since the last threshold crossing
    valid_crossings = []
    last_threshold_idx = -1
    
    for thresh_idx in neg_threshold_crossings:
        # Check if there's a zero crossing between the last threshold crossing and this one
        if last_threshold_idx == -1:
            # First crossing is always valid
            valid_crossings.append(thresh_idx)
            last_threshold_idx = thresh_idx
        else:
            # Check if there's a zero crossing after the last threshold crossing and before this one
            zero_after_last = zero_crossings[(zero_crossings > last_threshold_idx) & (zero_crossings < thresh_idx)]
            if len(zero_after_last) > 0:
                # Found a zero crossing, so this threshold crossing is valid
                valid_crossings.append(thresh_idx)
                last_threshold_idx = thresh_idx
    
    crossings = np.array(valid_crossings)
    num_crossings = len(crossings)

    for i, t in enumerate(crossings):
        sample_idx = t + window
        valid = (sample_idx >= 0) & (sample_idx < len(signal))

    crossing_times = crossings / fs

    return crossing_times


def calculate_threshold(curr_channel_data):
    median_val = np.median(curr_channel_data)
    absolute_deviations = np.abs(curr_channel_data - median_val)
    mad = np.median(absolute_deviations)
    stdev = mad / 0.6745
    threshold = 4 * stdev

    return threshold



#TOY DATA
def inhomogeneous_poisson_sinusoidal(
    duration: float,
    max_rate: float,
    min_rate: float,
    frequency: float,
    phase: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate an inhomogeneous Poisson process with sinusoidal rate via thinning.

    Args:
        duration: Total simulation time (seconds).
        max_rate: Maximum rate (Hz). Defines the rejection envelope.
        min_rate: Minimum rate (Hz). Must satisfy 0 <= min_rate <= max_rate.
        frequency: Sinusoid frequency (Hz).
        phase: Optional phase offset (radians).
        rng: Optional numpy Generator for reproducibility.

    Returns:
        np.ndarray of event times (seconds) sorted in ascending order.
    """
    if max_rate <= 0:
        raise ValueError("max_rate must be positive.")
    if min_rate < 0 or min_rate > max_rate:
        raise ValueError("min_rate must be in [0, max_rate].")
    if frequency <= 0:
        raise ValueError("frequency must be positive.")
    if rng is None:
        rng = np.random.default_rng()

    def lambda_t(t: float) -> float:
        # Sinusoid scaled to [min_rate, max_rate]
        return min_rate + (max_rate - min_rate) * 0.5 * (1 + math.sin(2 * math.pi * frequency * t + phase))

    lam_max = max_rate
    t = 0.0
    events = []
    while t < duration:
        t += rng.exponential(1.0 / lam_max)
        if t >= duration:
            break
        if rng.random() < lambda_t(t) / lam_max:
            events.append(t)
    return np.array(events, dtype=float)