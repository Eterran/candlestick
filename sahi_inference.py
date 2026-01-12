"""
SAHI (Slicing Aided Hyper Inference) module for improved small object detection.
Combines standard YOLO with SAHI for enhanced candlestick detection.
"""
import numpy as np
from generate.config import CLASS_MAPPING

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

# SAHI Configuration
SAHI_SLICE_HEIGHT = 640
SAHI_SLICE_WIDTH = 640
SAHI_OVERLAP_HEIGHT = 0.2
SAHI_OVERLAP_WIDTH = 0.2
SAHI_MERGE_NMS_THRESHOLD = 0.5


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    intersection = 0 if (x2_inter <= x1_inter or y2_inter <= y1_inter) else (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return 0 if union == 0 else intersection / union


def is_within_bounds(inner_box, outer_box, margin=0):
    """Check if inner_box is within outer_box (with optional margin). Format: [x1, y1, x2, y2]"""
    inner_x1, inner_y1, inner_x2, inner_y2 = inner_box
    outer_x1, outer_y1, outer_x2, outer_y2 = outer_box
    
    outer_x1 -= margin
    outer_y1 -= margin
    outer_x2 += margin
    outer_y2 += margin
    
    return (inner_x1 >= outer_x1 and inner_y1 >= outer_y1 and 
            inner_x2 <= outer_x2 and inner_y2 <= outer_y2)


def filter_standard_predictions(boxes, confs, cls_ids, model):
    """Filter standard predictions to keep only the best plot_area and axes, while keeping all candlesticks"""
    if len(boxes) == 0:
        return boxes, confs, cls_ids
    
    best_plot_area = None
    best_x_axis = None
    best_y_axis = None
    other_objects = []
    
    for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
        class_name = model.names[int(cls_id)]
        
        if class_name == 'plot_area':
            if best_plot_area is None or conf > best_plot_area[1]:
                best_plot_area = (i, conf, box, cls_id)
        elif class_name == 'x_axis':
            if best_x_axis is None or conf > best_x_axis[1]:
                best_x_axis = (i, conf, box, cls_id)
        elif class_name == 'y_axis':
            if best_y_axis is None or conf > best_y_axis[1]:
                best_y_axis = (i, conf, box, cls_id)
        else:
            other_objects.append((i, conf, box, cls_id))
    
    filtered_boxes = []
    filtered_confs = []
    filtered_cls = []
    
    for best_obj in [best_plot_area, best_x_axis, best_y_axis]:
        if best_obj is not None:
            _, conf, box, cls_id = best_obj
            filtered_boxes.append(box)
            filtered_confs.append(conf)
            filtered_cls.append(cls_id)
    
    for _, conf, box, cls_id in other_objects:
        filtered_boxes.append(box)
        filtered_confs.append(conf)
        filtered_cls.append(cls_id)
    
    print(f"    Filtered result: {len(filtered_boxes)} objects")
    return np.array(filtered_boxes), np.array(filtered_confs), np.array(filtered_cls)


def run_sahi_inference(model, image_path, model_path, confidence_threshold=0.5):
    """
    Run SAHI (Slicing Aided Hyper Inference) on an image.
    Returns detection results with better performance on large images.
    """
    if not SAHI_AVAILABLE:
        print("SAHI not available")
        return None
    
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device="cuda:0" if hasattr(model, 'device') and model.device.type == 'cuda' else "cpu"
        )
        
        result = get_sliced_prediction(
            image=image_path,
            detection_model=detection_model,
            slice_height=SAHI_SLICE_HEIGHT,
            slice_width=SAHI_SLICE_WIDTH,
            overlap_height_ratio=SAHI_OVERLAP_HEIGHT,
            overlap_width_ratio=SAHI_OVERLAP_WIDTH,
            postprocess_type="NMS",
            postprocess_match_threshold=SAHI_MERGE_NMS_THRESHOLD,
            verbose=0
        )
        
        # print(f"SAHI inference completed: {len(result.object_prediction_list)} detections")
        return result
        
    except Exception as e:
        print(f"SAHI inference failed: {e}")
        return None


def run_hybrid_inference(model, image_path, model_path, confidence_threshold=0.5, iou_threshold=0.4, use_sahi=True):
    """
    Hybrid inference with IoU-based deduplication.
    Combines standard YOLO with SAHI to find small candlesticks while avoiding duplicates.
    
    Args:
        model: YOLO model instance
        image_path: Path to the image
        model_path: Path to the model weights file
        confidence_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for deduplication
        use_sahi: Whether to use SAHI (if False, only standard YOLO is used)
        
    Returns:
        tuple: (boxes, confidences, class_ids) as numpy arrays
    """
    # print("Running hybrid inference with IoU-based deduplication...")
    
    # print("  -> Step 1: Standard YOLO prediction...")
    standard_results = model.predict(source=image_path, save=False, verbose=False)
    if not standard_results:
        print("No standard results found")
        return None
    
    standard_result = standard_results[0]
    
    standard_boxes = standard_result.boxes.xyxy.cpu().numpy() if len(standard_result.boxes) > 0 else []
    standard_confs = standard_result.boxes.conf.cpu().numpy() if len(standard_result.boxes) > 0 else []
    standard_cls = standard_result.boxes.cls.cpu().numpy() if len(standard_result.boxes) > 0 else []
    
    standard_boxes, standard_confs, standard_cls = filter_standard_predictions(
        standard_boxes, standard_confs, standard_cls, model
    )
    
    if not use_sahi or not SAHI_AVAILABLE:
        return standard_boxes, standard_confs, standard_cls
    
    sahi_boxes = []
    sahi_confs = []
    sahi_cls = []
    
    # print("  -> Step 2: SAHI prediction...")
    try:
        sahi_result = run_sahi_inference(model, image_path, model_path, confidence_threshold)
        if sahi_result and hasattr(sahi_result, 'object_prediction_list'):
            for pred in sahi_result.object_prediction_list:
                bbox = pred.bbox
                sahi_boxes.append([bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
                sahi_confs.append(pred.score.value)
                sahi_cls.append(CLASS_MAPPING.get(pred.category.name, 0))
            
            # print(f"  -> SAHI prediction found: {len(sahi_boxes)} objects")
    except Exception as e:
        print(f"  -> SAHI prediction failed: {e}, continuing with standard only")
    
    # print(f"  -> Step 3: Merging with IoU threshold {iou_threshold}...")
    
    final_boxes = list(standard_boxes)
    final_confs = list(standard_confs)
    final_cls = list(standard_cls)
    
    added_count = 0
    discarded_count = 0
    
    for i, sahi_box in enumerate(sahi_boxes):
        sahi_class_name = model.names[int(sahi_cls[i])]
        
        if sahi_class_name in ['plot_area', 'x_axis', 'y_axis']:
            discarded_count += 1
            continue
        
        if sahi_class_name == 'candlestick':
            plot_area_box = None
            for j, std_class in enumerate(standard_cls):
                if model.names[int(std_class)] == 'plot_area':
                    plot_area_box = standard_boxes[j]
                    break
            
            if plot_area_box is not None and not is_within_bounds(sahi_box, plot_area_box, margin=20):
                discarded_count += 1
                continue
        
        is_duplicate = False
        for standard_box in standard_boxes:
            if calculate_iou(sahi_box, standard_box) > iou_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_boxes.append(sahi_box)
            final_confs.append(sahi_confs[i])
            final_cls.append(sahi_cls[i])
            added_count += 1
        else:
            discarded_count += 1
    
    print(f"  -> Added {added_count} new objects from SAHI, discarded {discarded_count} duplicates")
    print(f"  -> Final result: {len(final_boxes)} total objects")
    
    return np.array(final_boxes), np.array(final_confs), np.array(final_cls)
