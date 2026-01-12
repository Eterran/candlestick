"""
Chart axis mapping and structured data extraction module.
Handles OCR-based price/time mapping and candlestick data extraction.
"""
import os
from PIL import Image
import pandas as pd
import numpy as np
from ocr import preprocess_for_ocr, reader, get_candle_color
from generate.config import CLASS_MAPPING
from gap_detection import detect_and_fill_gaps
from candlestick_analysis import extract_ohlc_from_candlestick, determine_open_close_from_color

CONFIDENCE_THRESHOLD = 0.85


def get_pixel_to_price_map_easyocr(full_image_path, y_axis_bbox, confidence_threshold=CONFIDENCE_THRESHOLD, verbose=True):
    """
    Create a pixel-to-price mapping from y-axis OCR analysis.
    
    Returns:
        tuple: (price_mapping_function, ocr_results_list) or (None, []) if failed
    """
    try:
        img = Image.open(full_image_path)
    except FileNotFoundError:
        if verbose:
            print(f"Error: Image not found at {full_image_path}")
        return None, []

    x1, y1, x2, y2 = [int(coord) for coord in y_axis_bbox]
    y_axis_image = img.crop((x1, y1, x2, y2))
    y_axis_image = preprocess_for_ocr(y_axis_image)

    results = reader.readtext(np.array(y_axis_image), allowlist='0123456789.,')
    
    # --- MERGING HEURISTIC FOR SPLIT NUMBERS (e.g. "46," "500.00") ---
    # Sort by Y center (to group lines) then X (left to right)
    # This helps if EasyOCR splits a single line into two chunks
    def get_center(bbox):
        return (bbox[0][1] + bbox[2][1]) / 2  # Y center
    
    results.sort(key=lambda x: (get_center(x[0]), x[0][0][0]))
    
    merged_results = []
    if len(results) > 0:
        curr_bbox, curr_text, curr_prob = results[0]
        
        for i in range(1, len(results)):
            next_bbox, next_text, next_prob = results[i]
            
            # Check vertical alignment (same line)
            curr_y_center = get_center(curr_bbox)
            next_y_center = get_center(next_bbox)
            
            # Check horizontal proximity
            # curr_bbox[1] is top-right, next_bbox[0] is top-left
            # We use x coordinates of these points
            curr_x_right = curr_bbox[1][0]
            next_x_left = next_bbox[0][0]
            
            # Thresholds
            y_diff = abs(curr_y_center - next_y_center)
            x_gap = next_x_left - curr_x_right
            
            # If on same line (within 10px Y) and close horizontally (within 20px X)
            # AND the current text ends with ',' or just looks like the start of a number
            if y_diff < 10 and x_gap < 20:
                # Merge
                if verbose:
                    print(f"  OCR Merge: '{curr_text}' + '{next_text}'")
                
                curr_text += next_text
                curr_prob = (curr_prob + next_prob) / 2 # Average confidence
                # Extend bbox: use min x/y and max x/y
                # Simplistic bbox merge: [TL, TR, BR, BL]
                # We just take TL of curr and BR of next ? roughly
                # Correct way for 4 points [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                new_x2 = next_bbox[1][0] # Right X from next
                curr_bbox[1][0] = new_x2 
                curr_bbox[2][0] = new_x2
                
            else:
                merged_results.append((curr_bbox, curr_text, curr_prob))
                curr_bbox, curr_text, curr_prob = next_bbox, next_text, next_prob
        
        merged_results.append((curr_bbox, curr_text, curr_prob))
        results = merged_results

    if verbose:
        print("--- EasyOCR Results (Post-Filtering & Merging) ---")
    
    pixel_price_pairs = []
    processed_ocr_results = []
    
    scale_factor = 2.0
    
    for (bbox, text, prob) in results:
        scaled_bbox = [[coord[0] / scale_factor, coord[1] / scale_factor] for coord in bbox]
        
        pixel_y_center_local = (scaled_bbox[0][1] + scaled_bbox[2][1]) / 2
        pixel_y_center_global = y1 + pixel_y_center_local
        
        abs_bbox = [[coord[0] + x1, coord[1] + y1] for coord in scaled_bbox]
        
        ocr_item = {
            'text': text,
            'confidence': prob,
            'bbox_relative': scaled_bbox,
            'bbox_absolute': abs_bbox,
            'center_y_absolute': pixel_y_center_global,
            'center_y_relative': pixel_y_center_local
        }
        
        if prob < confidence_threshold:
            ocr_item['used_for_mapping'] = False
            processed_ocr_results.append(ocr_item)
            continue

        try:
            # Remove commas (thousands separators) before converting to float
            cleaned_text = text.strip().replace(',', '')
            price_val = float(cleaned_text)
            ocr_item['price'] = price_val
            ocr_item['used_for_mapping'] = True
            processed_ocr_results.append(ocr_item)
            
            pixel_price_pairs.append((pixel_y_center_global, price_val))
            if verbose:
                print(f"  - Kept: '{text}' (Prob: {prob:.2f}) -> Price: {price_val}, Pixel-Y: {pixel_y_center_global:.1f}")
        except ValueError:
            if verbose:
                print(f"  - Discarded: '{text}' (Prob: {prob:.2f}) -> Not a valid number")
            ocr_item['used_for_mapping'] = False
            processed_ocr_results.append(ocr_item)

    if len(pixel_price_pairs) < 2:
        if verbose:
            print("Error: Need at least two valid price points to create a map.")
        return None, processed_ocr_results
        
    if len(pixel_price_pairs) < 2:
        if verbose:
            print("Error: Need at least two valid price points to create a map.")
        return None, processed_ocr_results
        
    
    # -------------------------------------------------------------------------
    # SCALE CORRECTION HEURISTIC
    # Detect if some numbers are missed decimals (e.g. 5970 instead of 59.70)
    # by using values with explicit decimal points as the "trusted" baseline.
    # -------------------------------------------------------------------------
    valid_items = [item for item in processed_ocr_results if item.get('used_for_mapping')]
    decimal_values = [item['price'] for item in valid_items if '.' in item['text']]
    
    if len(decimal_values) > 0:
        median_base = np.median(decimal_values)
        if verbose:
             print(f"  Scale Correction: Found base median price {median_base:.2f} from {len(decimal_values)} items with dots.")
        
        new_pixel_price_pairs = []
        for item in valid_items:
             price = item['price']
             ratio = price / median_base if median_base != 0 else 0
             
             # Check for missing '.' for different scales
             if 50 < ratio < 200:
                  if verbose:
                      print(f"    - Correcting {price} -> {price/100:.2f} (100x factor detected)")
                  price /= 100.0
             
             elif 5 < ratio < 20: 
                  if verbose:
                      print(f"    - Correcting {price} -> {price/10:.2f} (10x factor detected)")
                  price /= 10.0
                  
             new_pixel_price_pairs.append((item['center_y_absolute'], price))
        
        if len(new_pixel_price_pairs) >= 2:
            pixel_price_pairs = new_pixel_price_pairs
    # -------------------------------------------------------------------------

    pixels, prices = zip(*pixel_price_pairs)
    
    # Use RANSACRegressor for robust fitting
    try:
        from sklearn.linear_model import RANSACRegressor
        
        X = np.array(pixels).reshape(-1, 1)
        y = np.array(prices)
        
        # Configure RANSAC
        # residual_threshold: 10 pixels is a generous margin for OCR alignment errors, 
        # but small enough to exclude massive price jumps (like 111 vs 11100 on a 0-200 axis)
        ransac = RANSACRegressor(min_samples=2, residual_threshold=None, random_state=42)
        ransac.fit(X, y)
        
        # Log outliers and INLIERS for debug
        inlier_mask = ransac.inlier_mask_
        outlier_mask = np.logical_not(inlier_mask)
        
        if verbose:
            print(f"  RANSAC Results:")
            print(f"    - Inliers: {np.sum(inlier_mask)}")
            print(f"    - Outliers: {np.sum(outlier_mask)}")
            
            # Print ALL values to really see what's going on
            print("    --- All Candidate Points ---")
            for px, pr in zip(pixels, prices):
                print(f"      Pixel: {px:.1f}, Price: {pr} (Inlier: {ransac.predict([[px]])[0] - pr < 1.0 if ransac.estimator_ else 'Unknown'})")
                
            n_outliers = np.sum(outlier_mask)
            if n_outliers > 0:
                print(f"  RANSAC detected {n_outliers} outliers in price mapping:")
                outlier_pixels = X[outlier_mask].flatten()
                outlier_prices = y[outlier_mask]
                for p_px, p_val in zip(outlier_pixels, outlier_prices):
                    print(f"    - Ignored: Price {p_val} at Pixel {p_px:.1f}")
        
        def pixel_to_price(pixel_y):
            return ransac.predict([[pixel_y]])[0]
            
        m = ransac.estimator_.coef_[0]
        c = ransac.estimator_.intercept_
        if verbose:
             print(f"\nSuccessfully created robust mapping (RANSAC): price = {m:.4f} * pixel_y + {c:.4f}")

    except ImportError:
        print("Warning: scikit-learn not found, falling back to simple polyfit (sensitive to outliers).")
        m, c = np.polyfit(pixels, prices, 1)
        def pixel_to_price(pixel_y):
            return m * pixel_y + c
        if verbose:
            print(f"\nSuccessfully created mapping (Polyfit): price = {m:.4f} * pixel_y + {c:.4f}")
    return pixel_to_price, processed_ocr_results


def get_pixel_to_time_map_easyocr(full_image_path, x_axis_bbox, verbose=True):
    """Create a pixel-to-time mapping from x-axis OCR analysis (requires python-dateutil)"""
    # INCOMPLETE, CANT DEAL WITH MONTHS, MISSING CANDLESTICKS
    try:
        from dateutil.parser import parse
    except ImportError:
        print("Error: 'python-dateutil' not installed. Run: pip install python-dateutil")
        return None

    try:
        img = Image.open(full_image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {full_image_path}")
        return None

    x1, y1, x2, y2 = [int(coord) for coord in x_axis_bbox]
    x_axis_image = img.crop((x1, y1, x2, y2))
    results = reader.readtext(np.array(x_axis_image), allowlist='0123456789:- ')
    
    x_axis_image = img.crop((x1, y1, x2, y2))
    results = reader.readtext(np.array(x_axis_image), allowlist='0123456789:- ')
    
    if verbose:
        print("--- EasyOCR X-Axis Results ---")
        for (bbox, text, prob) in results:
            print(f"Detected: '{text}' (prob: {prob:.4f})")

    pixel_time_pairs = []
    for (bbox, text, prob) in results:
        try:
            dt_obj = parse(text)
            timestamp = dt_obj.timestamp()
            pixel_x_center = x1 + (bbox[0][0] + bbox[1][0]) / 2
            pixel_time_pairs.append((pixel_x_center, timestamp))
        except (ValueError, OverflowError):
            continue

    if len(pixel_time_pairs) < 2:
        if verbose:
            print("Error: Need at least two valid time points to create a map.")
        return None
        
    pixels, timestamps = zip(*pixel_time_pairs)
    m, c = np.polyfit(pixels, timestamps, 1)
    
    def pixel_to_timestamp(pixel_x):
        return m * pixel_x + c
        
    if verbose:
        print(f"\nSuccessfully created mapping: timestamp = {m:.4f} * pixel_x + {c:.4f}")
    return pixel_to_timestamp


def extract_structured_data(predictions, image_path, model, verbose=True):
    """
    Extract structured chart data from YOLO predictions and OCR.
    
    Returns:
        dict: Chart data with component boxes, candlesticks, mappers, and OCR results
    """
    if not os.path.exists(image_path):
        if verbose:
            print(f"Error: Image not found at {image_path}")
        return None

    if verbose:
        print(f"Processing predictions for image: {image_path}")
    if isinstance(predictions, dict):
        boxes = predictions.get('boxes', [])
        class_ids = predictions.get('class_ids', [])
        confidences = predictions.get('confidences', [])
        
        if len(boxes) > 0:
            boxes = np.array(boxes) if not isinstance(boxes, np.ndarray) else boxes
            class_ids = np.array(class_ids).astype(int) if not isinstance(class_ids, np.ndarray) else class_ids.astype(int)
            confidences = np.array(confidences) if not isinstance(confidences, np.ndarray) else confidences
        else:
            boxes, class_ids, confidences = np.array([]).reshape(0, 4), np.array([]), np.array([])
    elif hasattr(predictions, 'boxes') and hasattr(predictions.boxes, 'xyxy'):
        boxes = predictions.boxes.xyxy.cpu().numpy()
        class_ids = predictions.boxes.cls.cpu().numpy().astype(int)
        confidences = predictions.boxes.conf.cpu().numpy()
    else:
        boxes, class_ids, confidences = [], [], []
        for box in predictions.boxes:
            if hasattr(box, 'xyxy') and hasattr(box, 'cls'):
                boxes.append(box.xyxy[0].tolist() if hasattr(box.xyxy[0], 'tolist') else box.xyxy[0])
                class_ids.append(int(box.cls[0]))
                if hasattr(box, 'conf'):
                    confidences.append(float(box.conf[0]))
        
        boxes = np.array(boxes) if boxes else np.array([]).reshape(0, 4)
        class_ids = np.array(class_ids) if class_ids else np.array([])
        confidences = np.array(confidences) if confidences else np.array([])
    
    if verbose:
        print(f"Found {len(boxes)} objects to process")
    
    id_to_class_map = {v: k for k, v in CLASS_MAPPING.items()}
    component_boxes = {}
    candlestick_boxes = []
    
    for i, class_id in enumerate(class_ids):
        class_name = id_to_class_map.get(class_id)
        if not class_name:
            continue
        
        if class_name == 'candlestick':
            candlestick_boxes.append(boxes[i])
        elif class_name not in component_boxes:
            component_boxes[class_name] = boxes[i]

    if verbose:
        print(f"\n--- Detected Components ---")
        for name, box in component_boxes.items():
            print(f"Found '{name}' at {box.tolist()}")
        print(f"Found {len(candlestick_boxes)} candlesticks.")

    if len(candlestick_boxes) >= 3:
        candlestick_dicts = [
            {'bbox': box.tolist() if hasattr(box, 'tolist') else list(box), 'confidence': 1.0} 
            for box in candlestick_boxes
        ]
        
        filled_candlesticks = detect_and_fill_gaps(
            candlestick_dicts, 
            gap_threshold_multiplier=1.8,
            enable_filling=True,
            verbose=verbose
        )
        
        candlestick_boxes = [c['bbox'] for c in filled_candlesticks]
        if verbose:
            print(f"After gap filling: {len(candlestick_boxes)} candlesticks total.")

    y_axis_box = component_boxes.get('y_axis')
    price_mapper, ocr_results = None, []
    
    if y_axis_box is not None:
        if verbose:
            print("\n--- Creating Price-Pixel Map from Y-Axis ---")
        price_mapper, ocr_results = get_pixel_to_price_map_easyocr(image_path, y_axis_box, verbose=verbose)
    else:
        if verbose:
            print("\nWarning: Y-Axis not detected. Cannot create price map.")

    time_mapper = None

    # Construct groundtruth path - handle any image extension (.png, .jpg, .jpeg)
    base_path = os.path.splitext(image_path)[0]
    groundtruth_path = base_path.replace('images', 'groundtruth') + '.csv'
    ground_truth_df = None
    
    if os.path.exists(groundtruth_path):
        if verbose:
            print(f"\n--- Loading Ground Truth from {groundtruth_path} ---")
        ground_truth_df = pd.read_csv(groundtruth_path, index_col=0, parse_dates=True)
    else:
        if verbose:
            print(f"\nWarning: Ground truth not found at {groundtruth_path}")

    ohlc_results = []
    
    if price_mapper and candlestick_boxes:
        if verbose:
            print("\n--- OHLC Extraction (Body Detection Method) ---")
        candlestick_boxes.sort(key=lambda box: box[0])
        img = Image.open(image_path)

        for i, candle_box in enumerate(candlestick_boxes):
            # Extract OHLC using body detection
            # detailed debug only for first few
            ohlc_data = extract_ohlc_from_candlestick(img, candle_box, price_mapper, debug=(verbose and i<3))
            
            if ohlc_data is None:
                if verbose:
                    print(f"--- Candle {i+1} --- FAILED: Body detection error")
                continue
            
            high_price = ohlc_data['high']
            low_price = ohlc_data['low']
            open_close_1 = ohlc_data['open_close_1']
            open_close_2 = ohlc_data['open_close_2']
            
            # Try to determine Open vs Close using color
            open_price, close_price = determine_open_close_from_color(
                img, candle_box, open_close_1, open_close_2
            )
            
            # Fallback if color detection fails
            if open_price is None:
                open_price = open_close_2  # Assume bottom = Open (Green-ish default?? or just ambiguous)
                close_price = open_close_1 
                # Actually, blindly assuming might be wrong, but better than None for display
            
            candle_result = {
                'id': i+1,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'confidence': ohlc_data['confidence']
            }
            ohlc_results.append(candle_result)

            # Ground Truth Comparison (if available)
            if ground_truth_df is not None and i < len(ground_truth_df):
                if verbose:
                    gt_row = ground_truth_df.iloc[i]
                    gt_open, gt_high, gt_low, gt_close = gt_row['Open'], gt_row['High'], gt_row['Low'], gt_row['Close']

                    print(f"\n--- Candle {i+1} --- (Body confidence: {ohlc_data['confidence']:.2f})")
                    print(f"          | {'Extracted':<25} | {'Ground Truth':<15} | {'Difference':<15}")
                    print(f"----------|---------------------------|-----------------|-----------------")
                    print(f"High      | ${high_price:<8.2f}                   | ${gt_high:<14.2f} | ${high_price - gt_high:<14.2f}")
                    print(f"Low       | ${low_price:<8.2f}                   | ${gt_low:<14.2f} | ${low_price - gt_low:<14.2f}")
                    print(f"Open      | ${open_price:<24.2f} | ${gt_open:<14.2f} | ${open_price - gt_open:<14.2f}")
                    print(f"Close     | ${close_price:<24.2f} | ${gt_close:<14.2f} | ${close_price - gt_close:<14.2f}")


    if verbose:
        print("\n--- Extraction Complete ---")
    
    return {
        'component_boxes': component_boxes,
        'candlestick_boxes': candlestick_boxes,
        'price_mapper': price_mapper,
        'time_mapper': time_mapper,
        'ground_truth_df': ground_truth_df,
        'ocr_results': ocr_results,
        'ohlc_data': ohlc_results
    }