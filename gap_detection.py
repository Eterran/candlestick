"""
Gap detection and candlestick interpolation module.
Detects missing candlesticks by analyzing X-axis spacing patterns with robust statistics.
"""
import numpy as np
from typing import List, Dict, Tuple


def calculate_mad(values: np.ndarray, median: float = None) -> float:
    """
    Calculate Median Absolute Deviation (MAD) - a robust measure of variability.
    
    Args:
        values: Array of values
        median: Pre-computed median (optional)
        
    Returns:
        MAD value
    """
    if median is None:
        median = np.median(values)
    return np.median(np.abs(values - median))


def filter_outliers_iqr(values: List[float], k: float = 1.5) -> Tuple[List[float], float, float]:
    """
    Filter outliers using IQR (Interquartile Range) method.
    
    Args:
        values: List of values
        k: IQR multiplier for outlier detection (default: 1.5)
        
    Returns:
        Tuple of (filtered_values, lower_bound, upper_bound)
    """
    if len(values) < 4:
        return values, min(values) if values else 0, max(values) if values else 0
        
    arr = np.array(values)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    filtered = [v for v in values if lower_bound <= v <= upper_bound]
    
    return filtered if filtered else values, lower_bound, upper_bound


def calculate_robust_spacing_stats(candlesticks: List[Dict]) -> Tuple[float, float, float]:
    """
    Calculate robust spacing statistics with outlier filtering.
    
    Args:
        candlesticks: Sorted list of candlestick dictionaries
        
    Returns:
        Tuple of (median_spacing, mad_spacing, robust_threshold)
    """
    if len(candlesticks) < 2:
        return 0.0, 0.0, 0.0
    
    # Calculate all spacings
    spacings = []
    for i in range(len(candlesticks) - 1):
        x1_end = candlesticks[i]['bbox'][2]
        x2_start = candlesticks[i + 1]['bbox'][0]
        spacing = x2_start - x1_end
        spacings.append(spacing)
    
    # Filter outliers
    filtered_spacings, _, _ = filter_outliers_iqr(spacings, k=1.5)
    
    if not filtered_spacings:
        filtered_spacings = spacings
    
    # Calculate robust statistics
    median_spacing = np.median(filtered_spacings)
    mad_spacing = calculate_mad(np.array(filtered_spacings), median_spacing)
    
    # Adaptive threshold: median + k * MAD (k=3 for conservative detection)
    # MAD needs to be scaled by ~1.4826 to match std deviation for normal distribution
    robust_threshold = median_spacing + 3.0 * mad_spacing * 1.4826
    
    return median_spacing, mad_spacing, robust_threshold


def calculate_gap_confidence(gap_size: float, median_spacing: float, 
                             mad_spacing: float, median_width: float) -> float:
    """
    Calculate confidence score for a detected gap.
    
    Args:
        gap_size: Size of the gap in pixels
        median_spacing: Median spacing between candlesticks
        mad_spacing: MAD of spacing
        median_width: Median candlestick width
        
    Returns:
        Confidence score between 0.0 and 1.0
    """
    
    if mad_spacing > 0:
        z_score = (gap_size - median_spacing) / (mad_spacing * 1.4826)
    else:
        z_score = gap_size / median_spacing if median_spacing > 0 else 0
    
    # Confidence based on z-score (higher z-score = more confident it's a gap)
    # z > 3: very confident, z > 2: confident, z < 1.5: low confidence
    if z_score > 3:
        confidence = 0.95
    elif z_score > 2:
        confidence = 0.75 + (z_score - 2) * 0.2  # 0.75 to 0.95
    elif z_score > 1.5:
        confidence = 0.5 + (z_score - 1.5) * 0.5  # 0.5 to 0.75
    else:
        confidence = min(0.5, z_score / 1.5 * 0.5)  # 0 to 0.5
    
    # Also check if gap is at least 1.2x the median width (another sanity check)
    min_gap_threshold = median_width * 1.2
    if gap_size < min_gap_threshold:
        confidence *= 0.5
    
    return confidence


def detect_gaps(candlesticks: List[Dict], 
                gap_threshold_multiplier: float = 1.8,
                min_confidence: float = 0.5,
                verbose: bool = True) -> List[Tuple[int, int, int, float]]:
    """
    Detect gaps where candlesticks are likely missing using robust statistics.
    
    Args:
        candlesticks: List of candlestick dictionaries with 'bbox' key
        gap_threshold_multiplier: Multiplier for median spacing (legacy, now uses MAD-based threshold)
        min_confidence: Minimum confidence to report a gap (default: 0.5)
        verbose: Whether to print detection details (default: True)
        
    Returns:
        List of tuples: (index_before, index_after, estimated_missing_count, confidence)
    """
    if len(candlesticks) < 3:
        if verbose:
            print("Not enough candlesticks to detect gaps (need at least 3)")
        return []
    
    sorted_candles = sorted(candlesticks, key=lambda c: c['bbox'][0])
    
    # Calculate robust statistics
    median_spacing, mad_spacing, robust_threshold = calculate_robust_spacing_stats(sorted_candles)
    median_width = np.median([c['bbox'][2] - c['bbox'][0] for c in sorted_candles])
    
    if verbose:
        print(f"\n--- Gap Detection (Robust) ---")
        print(f"Median candlestick spacing: {median_spacing:.2f}px")
        print(f"MAD of spacing: {mad_spacing:.2f}px")
        print(f"Median candlestick width: {median_width:.2f}px")
        print(f"Adaptive gap threshold: {robust_threshold:.2f}px")
    
    gaps = []
    
    for i in range(len(sorted_candles) - 1):
        current_candle = sorted_candles[i]
        next_candle = sorted_candles[i + 1]
        
        x1_end = current_candle['bbox'][2]
        x2_start = next_candle['bbox'][0]
        gap_size = x2_start - x1_end
        
        # Check if gap exceeds adaptive threshold
        if gap_size > robust_threshold:
            # Calculate confidence
            confidence = calculate_gap_confidence(gap_size, median_spacing, mad_spacing, median_width)
            
            # Only report gaps above minimum confidence
            if confidence >= min_confidence:
                # Estimate missing count
                expected_space_per_candle = median_width + median_spacing
                estimated_missing = max(1, int(round(gap_size / expected_space_per_candle)))
                
                gaps.append((i, i + 1, estimated_missing, confidence))
                if verbose:
                    print(f"  Gap detected between candle {i} and {i+1}:")
                    print(f"    Gap size: {gap_size:.2f}px (threshold: {robust_threshold:.2f}px)")
                    print(f"    Estimated missing candlesticks: {estimated_missing}")
                    print(f"    Confidence: {confidence:.2f}")
            else:
                if verbose:
                    print(f"  Low-confidence gap rejected between candle {i} and {i+1} (confidence: {confidence:.2f})")
    
    if not gaps and verbose:
        print("  No significant gaps detected")
    
    return gaps


def create_interpolated_candlestick(candle_before: Dict, candle_after: Dict, 
                                   position_fraction: float, index: int) -> Dict:
    """
    Create an interpolated candlestick between two existing candlesticks.
    
    Args:
        candle_before: Candlestick dictionary before the gap
        candle_after: Candlestick dictionary after the gap
        position_fraction: Position between candles (0.0 to 1.0)
        index: Index for the interpolated candlestick
        
    Returns:
        Interpolated candlestick dictionary
    """
    bbox_before = candle_before['bbox']
    bbox_after = candle_after['bbox']
    
    # Interpolate x position
    x1 = bbox_before[0] + (bbox_after[0] - bbox_before[0]) * position_fraction
    x2 = bbox_before[2] + (bbox_after[2] - bbox_before[2]) * position_fraction
    
    # Use average y positions (or interpolate)
    y1 = (bbox_before[1] + bbox_after[1]) / 2
    y2 = (bbox_before[3] + bbox_after[3]) / 2
    
    interpolated = {
        'bbox': [x1, y1, x2, y2],
        'confidence': 0.0,  # Mark as interpolated
        'class_id': candle_before.get('class_id', 0),
        'interpolated': True,
        'interpolated_index': index
    }
    
    return interpolated


def fill_gaps(candlesticks: List[Dict], gaps: List[Tuple[int, int, int, float]], verbose: bool = True) -> List[Dict]:
    """
    Fill detected gaps with interpolated candlesticks.
    
    Args:
        candlesticks: Original list of candlestick dictionaries
        gaps: List of detected gaps (from detect_gaps) with confidence scores
        verbose: Whether to print filling details
        
    Returns:
        New list with interpolated candlesticks added
    """
    if not gaps:
        return candlesticks
    
    sorted_candles = sorted(candlesticks, key=lambda c: c['bbox'][0])
    result = []
    
    if verbose:
        print(f"\n--- Filling Gaps ---")
    
    gap_index = 0
    for i, candle in enumerate(sorted_candles):
        result.append(candle)
        
        if gap_index < len(gaps) and gaps[gap_index][0] == i:
            idx_before, idx_after, missing_count, confidence = gaps[gap_index]
            candle_after = sorted_candles[idx_after]
            
            if verbose:
                print(f"  Filling gap between candle {idx_before} and {idx_after} with {missing_count} interpolated candlestick(s) (confidence: {confidence:.2f})")
            
            # Create interpolated candlesticks
            for j in range(missing_count):
                fraction = (j + 1) / (missing_count + 1)
                interpolated = create_interpolated_candlestick(
                    candle, candle_after, fraction, j
                )
                result.append(interpolated)
            
            gap_index += 1
    
    if verbose:
        print(f"  Total candlesticks after filling: {len(result)} (added {len(result) - len(candlesticks)})")
    
    return result


def detect_and_fill_gaps(candlesticks: List[Dict], 
                         gap_threshold_multiplier: float = 1.8,
                         enable_filling: bool = True,
                         min_confidence: float = 0.5,
                         verbose: bool = True) -> List[Dict]:
    """
    Main function to detect and optionally fill gaps in candlestick data.
    
    Args:
        candlesticks: List of candlestick dictionaries with 'bbox' key
        gap_threshold_multiplier: Legacy parameter (now uses adaptive MAD-based threshold)
        enable_filling: Whether to fill detected gaps (default: True)
        min_confidence: Minimum confidence for gap detection (default: 0.5)
        verbose: Whether to print details
    """
    if len(candlesticks) < 3:
        if verbose:
            print("Not enough candlesticks for gap detection")
        return candlesticks
    
    gaps = detect_gaps(candlesticks, gap_threshold_multiplier, min_confidence, verbose=verbose)
    
    if not gaps:
        return candlesticks
    
    if enable_filling:
        return fill_gaps(candlesticks, gaps, verbose=verbose)
    else:
        if verbose:
            print("\nGap filling disabled, returning original candlesticks")
        return candlesticks
