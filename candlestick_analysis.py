"""
Candlestick body detection and Open/Close price level extraction.
Analyzes pixel density to find the body region (Open/Close) within a candlestick bounding box.
"""
import numpy as np
from PIL import Image
import cv2


def detect_candlestick_body(image, candle_bbox, debug=False):
    """
    Detect the candlestick body region by analyzing vertical pixel density.
    
    The body is the thick part of the candlestick (between Open and Close).
    The wicks are the thin lines extending above/below the body.
    
    Args:
        image: PIL Image object
        candle_bbox: [x1, y1, x2, y2] bounding box of the candlestick
        debug: If True, print debug information
        
    Returns:
        dict: {
            'body_top': y-coordinate of body top (max of Open/Close),
            'body_bottom': y-coordinate of body bottom (min of Open/Close),
            'wick_top': y-coordinate of upper wick end (High),
            'wick_bottom': y-coordinate of lower wick end (Low),
            'body_center_x': x-coordinate of body center,
            'confidence': confidence score (0-1)
        }
    """
    x1, y1, x2, y2 = [int(coord) for coord in candle_bbox]
    
    # Crop the candlestick region
    candle_crop = image.crop((x1, y1, x2, y2))
    candle_array = np.array(candle_crop.convert('L'))  # Convert to grayscale
    
    height, width = candle_array.shape
    
    if height < 5 or width < 3:
        # Candlestick too small to analyze
        return None
    
    # Calculate horizontal profile (sum of pixels along each row)
    # Higher values = more filled pixels (body region)
    # Lower values = fewer pixels (wick region)
    horizontal_profile = np.sum(candle_array < 200, axis=1)  # Count dark pixels per row
    
    if debug:
        print(f"\n--- Candlestick Body Detection ---")
        print(f"Candlestick bbox: [{x1}, {y1}, {x2}, {y2}]")
        print(f"Crop size: {width}x{height}")
        print(f"Horizontal profile shape: {horizontal_profile.shape}")
    
    # Smooth the profile to reduce noise
    from scipy.ndimage import gaussian_filter1d
    try:
        smoothed_profile = gaussian_filter1d(horizontal_profile.astype(float), sigma=1.5)
    except:
        # Fallback if scipy not available
        kernel_size = 3
        kernel = np.ones(kernel_size) / kernel_size
        smoothed_profile = np.convolve(horizontal_profile, kernel, mode='same')
    
    # Find the body region (consecutive rows with high pixel density)
    # Body is typically 40-80% of the width
    median_width_threshold = width * 0.35
    body_threshold = max(median_width_threshold, np.percentile(smoothed_profile, 60))
    
    body_mask = smoothed_profile > body_threshold
    
    if not np.any(body_mask):
        # No clear body detected, might be a doji or very small body
        # Use central 50% of the candlestick as estimated body
        body_top_rel = int(height * 0.4)
        body_bottom_rel = int(height * 0.6)
        confidence = 0.3
    else:
        # Find continuous body region
        body_indices = np.where(body_mask)[0]
        
        # Find the largest continuous segment
        segments = []
        start_idx = body_indices[0]
        for i in range(1, len(body_indices)):
            if body_indices[i] - body_indices[i-1] > 2:  # Gap detected
                segments.append((start_idx, body_indices[i-1]))
                start_idx = body_indices[i]
        segments.append((start_idx, body_indices[-1]))
        
        # Choose the largest segment
        largest_segment = max(segments, key=lambda s: s[1] - s[0])
        body_top_rel = largest_segment[0]
        body_bottom_rel = largest_segment[1]
        
        # Calculate confidence based on profile clarity
        # Calculate confidence based on profile clarity
        body_slice = smoothed_profile[body_top_rel:body_bottom_rel+1]
        body_profile_mean = np.mean(body_slice) if len(body_slice) > 0 else 0
        
        wick_parts = []
        if body_top_rel > 0:
            wick_parts.append(smoothed_profile[:body_top_rel])
        if body_bottom_rel < height-1:
            wick_parts.append(smoothed_profile[body_bottom_rel+1:])
            
        if wick_parts:
            wick_pixels = np.concatenate(wick_parts)
            wick_profile_mean = np.mean(wick_pixels) if len(wick_pixels) > 0 else 0
        else:
            wick_profile_mean = 0
        
        confidence = min(1.0, (body_profile_mean - wick_profile_mean) / (width * 0.5))
        confidence = max(0.0, confidence)
    
    # Convert relative coordinates to absolute image coordinates
    body_top_abs = y1 + body_top_rel
    body_bottom_abs = y1 + body_bottom_rel
    wick_top_abs = y1  # Top of bounding box
    wick_bottom_abs = y2  # Bottom of bounding box
    body_center_x = (x1 + x2) / 2
    
    if debug:
        print(f"Body detected at relative y: [{body_top_rel}, {body_bottom_rel}]")
        print(f"Body in absolute coords: [{body_top_abs:.1f}, {body_bottom_abs:.1f}]")
        print(f"Body height: {body_bottom_rel - body_top_rel} pixels")
        print(f"Confidence: {confidence:.2f}")
    
    return {
        'body_top': body_top_abs,
        'body_bottom': body_bottom_abs,
        'wick_top': wick_top_abs,
        'wick_bottom': wick_bottom_abs,
        'body_center_x': body_center_x,
        'confidence': confidence,
        'body_height_pixels': body_bottom_rel - body_top_rel
    }


def extract_ohlc_from_candlestick(image, candle_bbox, price_mapper, debug=False):
    """
    Extract OHLC (Open, High, Low, Close) values from a single candlestick.
    
    Args:
        image: PIL Image object
        candle_bbox: [x1, y1, x2, y2] bounding box of the candlestick
        price_mapper: Function to convert pixel y-coordinate to price
        debug: If True, print debug information
        
    Returns:
        dict: {
            'high': High price,
            'low': Low price,
            'open_close_1': First body edge price (either Open or Close),
            'open_close_2': Second body edge price (either Open or Close),
            'body_detected': True if body was successfully detected,
            'confidence': confidence score
        }
    """
    if price_mapper is None:
        print("Error: price_mapper is None")
        return None
    
    # Detect body region
    body_info = detect_candlestick_body(image, candle_bbox, debug=debug)
    
    if body_info is None:
        return None
    
    # Extract prices using the price mapper
    high_price = price_mapper(body_info['wick_top'])
    low_price = price_mapper(body_info['wick_bottom'])
    
    # Body edges represent Open and Close (we don't know which is which yet)
    open_close_1 = price_mapper(body_info['body_top'])
    open_close_2 = price_mapper(body_info['body_bottom'])
    
    if debug:
        print(f"\n--- OHLC Extraction ---")
        print(f"High: ${high_price:.2f} at pixel y={body_info['wick_top']:.1f}")
        print(f"Low: ${low_price:.2f} at pixel y={body_info['wick_bottom']:.1f}")
        print(f"Body edge 1: ${open_close_1:.2f} at pixel y={body_info['body_top']:.1f}")
        print(f"Body edge 2: ${open_close_2:.2f} at pixel y={body_info['body_bottom']:.1f}")
    
    return {
        'high': high_price,
        'low': low_price,
        'open_close_1': open_close_1,  # Top of body
        'open_close_2': open_close_2,  # Bottom of body
        'body_detected': True,
        'confidence': body_info['confidence'],
        'body_height_pixels': body_info['body_height_pixels'],
        'body_top_px': body_info['body_top'],
        'body_bottom_px': body_info['body_bottom']
    }


def determine_open_close_from_color(image, candle_bbox, open_close_1, open_close_2):
    """
    Determine which body edge is Open and which is Close based on candle color.
    
    Args:
        image: PIL Image object
        candle_bbox: [x1, y1, x2, y2] bounding box of the candlestick
        open_close_1: Price at body top
        open_close_2: Price at body bottom
        
    Returns:
        tuple: (open_price, close_price) or (None, None) if color detection fails
    """
    from ocr import get_candle_color
    
    color = get_candle_color(image, candle_bbox)
    
    # open_close_1 is body_top (smaller y, higher price)
    # open_close_2 is body_bottom (larger y, lower price)
    
    if color == 'up':
        # Green/bullish candle: Close > Open
        # Body top (open_close_1) = Close (higher price)
        # Body bottom (open_close_2) = Open (lower price)
        return open_close_2, open_close_1
    elif color == 'down':
        # Red/bearish candle: Close < Open
        # Body top (open_close_1) = Open (higher price)
        # Body bottom (open_close_2) = Close (lower price)
        return open_close_1, open_close_2
    else:
        # Unknown color - can't determine Open/Close order
        return None, None


if __name__ == '__main__':
    print("Candlestick body detection module")
    print("Use extract_ohlc_from_candlestick() to extract OHLC values")


def visualize_body_detection(image, candle_bbox, body_info, save_path=None):
    """
    Create a visualization of the body detection result.
    
    Args:
        image: PIL Image object
        candle_bbox: [x1, y1, x2, y2] candlestick bounding box
        body_info: Output from detect_candlestick_body()
        save_path: Optional path to save the visualization
    """
    from PIL import ImageDraw, ImageFont
    
    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    x1, y1, x2, y2 = [int(c) for c in candle_bbox]
    
    # Draw full candlestick box (blue)
    draw.rectangle([x1, y1, x2, y2], outline='blue', width=2)
    draw.text((x1, y1-15), "Full Candlestick", fill='blue', font=font)
    
    # Draw body region (green)
    body_top = body_info['body_top']
    body_bottom = body_info['body_bottom']
    draw.rectangle([x1, body_top, x2, body_bottom], outline='green', width=3)
    draw.text((x2+5, body_top), f"Body (conf: {body_info['confidence']:.2f})", fill='green', font=font)
    
    # Draw wick markers (red)
    center_x = int(body_info['body_center_x'])
    draw.line([(center_x, y1), (center_x, body_top)], fill='red', width=2)  # Upper wick
    draw.line([(center_x, body_bottom), (center_x, y2)], fill='red', width=2)  # Lower wick
    
    if save_path:
        vis_img.save(save_path)
        print(f"Body detection visualization saved to: {save_path}")
    
    return vis_img
