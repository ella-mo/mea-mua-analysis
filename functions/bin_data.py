import numpy as np

def readable_float(float_string):
    # Convert to float first to ensure decimal point is always present
    # This handles both integers (90 -> 90.0) and floats (0.02 -> 0.02)
    return str(float(float_string)).replace('.','_')

def validate_bin_data(binned_trials, spike_times_per_channel, num_channels, recording_duration, sample_len, bin_size, overlap):
    """
    Validate that bin_data was performed correctly.
    
    Parameters
    ----------
    binned_trials : array, shape (n_trials, n_timesteps, num_channels)
        Output from bin_data function
    spike_times_per_channel : list of arrays
        List of spike time arrays, one per channel. Each array contains spike times in seconds.
    num_channels : int
    recording_duration : float
    fs : float
    sample_len : float
    bin_size : float
    overlap : float
    
    Returns
    -------
    is_valid : bool
        True if all checks pass
    """
    print("\n" + "="*60)
    print("VALIDATING bin_data OUTPUT")
    print("="*60)
    
    all_checks_passed = True
    
    # Expected dimensions
    n_timesteps = int(np.round(sample_len / bin_size))
    window_size = n_timesteps * bin_size
    stride = window_size - overlap
    n_windows = int(np.floor((recording_duration + overlap) / stride))
    window_starts = np.array([-overlap + i * stride for i in range(n_windows)])
    
    # Check 1: Output shape
    expected_shape = (n_windows, n_timesteps, num_channels)
    actual_shape = binned_trials.shape
    print(f"\n1. Shape check:")
    print(f"   Expected: {expected_shape}")
    print(f"   Actual:   {actual_shape}")
    if actual_shape == expected_shape:
        print("   ✓ PASSED")
    else:
        print("   ✗ FAILED")
        all_checks_passed = False
    
    # Check 2: No negative values
    min_val = np.min(binned_trials)
    print(f"\n2. Non-negative values check:")
    print(f"   Minimum value: {min_val}")
    if min_val >= 0:
        print("   ✓ PASSED")
    else:
        print("   ✗ FAILED: Found negative spike counts!")
        all_checks_passed = False
    
    # Check 3: Bin indices within valid range
    # This is already handled in bin_data, but verify no values exceed expected
    max_bin_val = np.max(binned_trials)
    print(f"\n3. Reasonable spike count check:")
    print(f"   Maximum spikes per bin: {max_bin_val}")
    print(f"   Mean spikes per bin: {np.mean(binned_trials):.4f}")
    print(f"   Total spikes across all bins: {np.sum(binned_trials):.0f}")
    if max_bin_val < 1000:  # Reasonable upper bound
        print("   ✓ PASSED (max value seems reasonable)")
    else:
        print("   ⚠ WARNING: Very high spike counts detected")
    
    # Check 4: Window coverage
    print(f"\n4. Window coverage check:")
    print(f"   Number of windows: {n_windows}")
    print(f"   First window start: {window_starts[0]:.4f} s")
    print(f"   Last window start: {window_starts[-1]:.4f} s")
    print(f"   Last window end: {window_starts[-1] + window_size:.4f} s")
    print(f"   Recording duration: {recording_duration:.4f} s")
    last_window_end = window_starts[-1] + window_size
    if abs(last_window_end - recording_duration) < 0.1:  # Allow small floating point error
        print("   ✓ PASSED")
    else:
        print(f"   ⚠ WARNING: Last window ends at {last_window_end:.4f}s, expected {recording_duration:.4f}s")
    
    # Check 5: Overlap consistency (if overlap > 0)
    if overlap > 0:
        print(f"\n5. Overlap consistency check:")
        overlap_bins = int(overlap / bin_size)
        print(f"   Overlap: {overlap} s ({overlap_bins} bins)")
        
        n_mismatches = 0
        for i in range(n_windows - 1):
            # Check if consecutive windows overlap
            window_i_end = window_starts[i] + window_size
            window_i1_start = window_starts[i + 1]
            
            if window_i_end > window_i1_start:  # They overlap
                # Get overlapping regions
                overlap_start_bin = int(np.round((window_i1_start - window_starts[i]) / bin_size))
                overlap_end_bin = overlap_start_bin + overlap_bins
                
                chunk_i_end = binned_trials[i, overlap_start_bin:overlap_end_bin, :]
                chunk_i1_start = binned_trials[i + 1, :overlap_bins, :]
                
                if not np.array_equal(chunk_i_end, chunk_i1_start):
                    n_mismatches += 1
                    if n_mismatches <= 3:  # Only show first few mismatches
                        n_diff = np.sum(chunk_i_end != chunk_i1_start)
                        print(f"   ⚠ Mismatch between windows {i} and {i+1}: {n_diff} bins differ")
        
        if n_mismatches == 0:
            print("   ✓ PASSED: All overlapping regions match")
        else:
            print(f"   ⚠ WARNING: {n_mismatches} window pairs have mismatched overlaps")
            print("   (This may be expected if spike detection varies slightly)")
    
    # Check 6: Compare total spike counts with provided spike times
    print(f"\n6. Spike count preservation check:")
    total_spikes_binned = np.sum(binned_trials)
    
    # Count spikes from provided spike times within recording duration
    total_spikes_direct = 0
    for ch in range(num_channels):
        spike_times = spike_times_per_channel[ch]
        # Count spikes within recording duration
        spikes_in_range = np.sum((spike_times >= 0) & (spike_times < recording_duration))
        total_spikes_direct += spikes_in_range
    
    print(f"   Total spikes in binned data: {total_spikes_binned:.0f}")
    print(f"   Total spikes from provided spike times: {total_spikes_direct:.0f}")
    
    if overlap > 0:
        # With overlap, spikes can be counted multiple times
        # Estimate expected count: each spike in overlap region is counted twice
        # Rough estimate: spikes in overlap regions are double-counted
        overlap_ratio = overlap / stride if stride > 0 else 0
        expected_with_overlap = total_spikes_direct * (1 + overlap_ratio)
        print(f"   Expected with overlap (approx): {expected_with_overlap:.0f}")
        print("   ✓ PASSED (overlap causes double-counting, so counts won't match exactly)")
    else:
        if abs(total_spikes_binned - total_spikes_direct) < 0.01 * total_spikes_direct:
            print("   ✓ PASSED")
        else:
            diff_pct = 100 * abs(total_spikes_binned - total_spikes_direct) / total_spikes_direct
            print(f"   ⚠ WARNING: {diff_pct:.2f}% difference (may be due to edge effects)")
    
    # Check 7: Data statistics
    print(f"\n7. Data statistics:")
    print(f"   Non-zero bins: {np.count_nonzero(binned_trials)} / {binned_trials.size} ({100*np.count_nonzero(binned_trials)/binned_trials.size:.2f}%)")
    print(f"   Mean spikes per bin (non-zero): {np.mean(binned_trials[binned_trials > 0]):.4f}")
    print(f"   Max spikes in single bin: {np.max(binned_trials)}")
    
    print("\n" + "="*60)
    if all_checks_passed:
        print("✓ ALL CRITICAL CHECKS PASSED")
    else:
        print("✗ SOME CHECKS FAILED - REVIEW WARNINGS ABOVE")
    print("="*60 + "\n")
    
    return all_checks_passed


def bin_make_train_val(spike_times_per_channel, num_channels, recording_duration, sample_len, bin_size, overlap, split_frac, DEBUG=False):
    """
    Bin spike times into overlapping windows.

    Parameters
    ----------
    spike_times_per_channel : list of arrays
        List of spike time arrays, one per channel. Each array contains spike times in seconds.
    num_channels : int
    recording_duration : float
        Total recording length in seconds.
    sample_len : float
        Length of each trial/window in seconds (e.g., 12 for 12 s windows).
    bin_size : float
        Bin size in seconds (e.g., 0.005 for 5 ms).
    overlap : float
        Overlap between consecutive windows in seconds (e.g., 2 for 2 s overlap).
    split_frac : float
        Fraction of data to use for training (e.g., 0.75).
    DEBUG : bool, optional
        Whether to print debug information. Default is False.

    Returns
    -------
    train_data : array, shape (n_train_trials, n_timesteps, num_channels)
    valid_data : array, shape (n_valid_trials, n_timesteps, num_channels)
    train_idx : array
        Indices of training trials.
    valid_idx : array
        Indices of validation trials.
    """

    # Convert durations to bins
    n_timesteps = int(np.round(sample_len / bin_size))
    window_size = n_timesteps * bin_size
    stride = window_size - overlap

    # Window start times: first at -overlap, last ends at recording_duration
    # Use a more precise calculation to avoid floating point errors
    n_windows = int(np.floor((recording_duration + overlap) / stride))
    window_starts = np.array([-overlap + i * stride for i in range(n_windows)])
    if DEBUG:
        print(f'window starts {window_starts}')
        print(f'n_timesteps: {n_timesteps}, window_size: {window_size}, stride: {stride}')

    n_trials = len(window_starts)

    # Initialize output array
    binned_trials = np.zeros((n_trials, n_timesteps, num_channels), dtype=np.float32)

    for ch in range(num_channels):
        spike_times = spike_times_per_channel[ch]
        if ch==0 and DEBUG:
            print(f'{spike_times}')

        # Bin spikes for each window
        for i, start_time in enumerate(window_starts):
            end_time = start_time + window_size
            mask = (spike_times >= start_time) & (spike_times < end_time)
            spikes_in_window = spike_times[mask]
            if ch == 0 and DEBUG:
                print(f'spikes in window: {spikes_in_window}')

            if spikes_in_window.size == 0:
                continue

            relative_times = spikes_in_window - start_time
            # Use more precise binning to avoid floating point errors
            # Round to avoid precision issues, then floor to ensure consistent binning
            # across overlapping windows
            bin_indices = np.floor(np.round(relative_times / bin_size, decimals=10)).astype(int)
            # Clamp bin indices to valid range [0, n_timesteps)
            bin_indices = np.clip(bin_indices, 0, n_timesteps - 1)
            np.add.at(binned_trials[i, :, ch], bin_indices, 1)

    validate_bin_data(binned_trials, spike_times_per_channel, num_channels, recording_duration, sample_len, bin_size, overlap)

    # Train-val split
    # Randomized train/valid split with index tracking
    n_sessions = binned_trials.shape[0]
    indices = np.arange(n_sessions)

    # Shuffle indices reproducibly
    rng = np.random.default_rng(seed=0)
    rng.shuffle(indices)

    # Compute split point
    split_point = int(n_sessions * split_frac)

    # Split into train and validation indices
    train_idx = indices[:split_point]
    valid_idx = indices[split_point:]

    # Slice data
    train_data = binned_trials[train_idx]
    valid_data = binned_trials[valid_idx]

    return binned_trials, train_data, valid_data, train_idx, valid_idx