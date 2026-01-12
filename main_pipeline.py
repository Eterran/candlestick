"""
Main pipeline for chart data extraction using YOLO model and OCR.
Supports both hybrid inference (from test_control) and basic YOLO inference.
"""
import os
import sys
from ultralytics import YOLO
from axis_mapping import extract_structured_data
from debug import draw_predictions_with_labels, visualize_y_axis_mapping_with_ocr, create_debug_summary
from sahi_inference import run_hybrid_inference, SAHI_AVAILABLE

HYBRID_AVAILABLE = SAHI_AVAILABLE
if HYBRID_AVAILABLE:
    print("Hybrid inference available (YOLO + SAHI)")
else:
    print("SAHI not available, using basic YOLO only")

# Model configuration
# Model configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, 'best.pt')
CONFIDENCE_THRESHOLD = 0.5

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    print(f"Loading model from: {model_path}")
    return YOLO(model_path)

def run_model_predictions(model, image_path):
    """Run model predictions using hybrid inference or basic YOLO"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None

    if HYBRID_AVAILABLE:
        try:
            print("Using hybrid inference...")
            final_boxes, final_confs, final_cls = run_hybrid_inference(
                model, image_path, MODEL_PATH, CONFIDENCE_THRESHOLD
            )
            predictions = {
                'boxes': final_boxes,
                'confidences': final_confs,
                'class_ids': final_cls
            }
            print(f"Hybrid inference: {len(final_boxes)} objects detected")
            return predictions, model
        except Exception as e:
            print(f"Hybrid inference error: {e}, falling back to basic YOLO...")
    
    print("Using basic YOLO inference...")
    results = model.predict(source=image_path, conf=CONFIDENCE_THRESHOLD, save=False, verbose=False)
    
    if not results or len(results[0].boxes) == 0:
        print("No objects detected")
        return None, None
    
    result = results[0]
    predictions = {
        'boxes': result.boxes.xyxy.cpu().numpy(),
        'confidences': result.boxes.conf.cpu().numpy(),
        'class_ids': result.boxes.cls.cpu().numpy()
    }
    
    print(f"Basic inference: {len(predictions['boxes'])} objects detected")
    return predictions, model


def main(image_path=None, show_visualizations=False, model=None):
    """Main pipeline function that orchestrates the entire process"""
    if image_path is None:
        image_path = 'chart_dataset_v2.11/val/images/chart_108.png'
    
    print(f"\n=== Chart Data Extraction Pipeline ===")
    print(f"Image: {image_path}")
    print(f"Model: {MODEL_PATH}")
    
    if model is None:
        model = load_model(MODEL_PATH)

    predictions, model = run_model_predictions(model, image_path)
    if predictions is None:
        print("Failed to get predictions")
        return None
    
    print("\n--- Extracting Structured Chart Data ---")
    chart_data = extract_structured_data(predictions, image_path, model)
    
    if not chart_data:
        print("Extraction failed")
        return None
    
    if show_visualizations:
        print("\n--- Creating Visualizations ---")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        draw_predictions_with_labels(image_path, predictions, model, f"predictions_{base_name}.png")
        visualize_y_axis_mapping_with_ocr(image_path, 
                                         chart_data.get('component_boxes', {}).get('y_axis'),
                                         chart_data, 
                                         f"y_axis_mapping_{base_name}.png")
    
    print("\n--- Extraction Complete ---")
    component_boxes = chart_data.get('component_boxes', {})
    candlestick_boxes = chart_data.get('candlestick_boxes', [])
    
    print("Detected Components:")
    for component, bbox in component_boxes.items():
        print(f"  - {component}: {bbox}")
    print(f"  - candlesticks: {len(candlestick_boxes)} found")
    
    return chart_data


if __name__ == '__main__':
    show_viz = False
    image_path = None
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if len(sys.argv) > 2 and sys.argv[2].lower() in ['--no-viz', '--no-visualizations', 'false']:
            show_viz = False
    else:
        test_image = 'chart_dataset_v2.11/val/images/chart_108.png'
        
        if not os.path.exists(test_image):
            print(f"Default test image not found, searching for alternatives...")
            for item in os.listdir('.'):
                if item.startswith('chart_dataset'):
                    val_images = os.path.join(item, 'val', 'images')
                    if os.path.exists(val_images):
                        images = [f for f in os.listdir(val_images) if f.endswith('.png')]
                        if images:
                            test_image = os.path.join(val_images, images[0])
                            print(f"Using: {test_image}")
                            break
        
        image_path = test_image
    
    result = main(image_path, show_visualizations=show_viz)
    
    if result:
        print("\n=== SUCCESS ===")
        if show_viz:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            print("\nVisualization files:")
            print(f"  - predictions_{base_name}.png")
            print(f"  - y_axis_mapping_{base_name}.png")
    else:
        print("\n=== FAILED ===")
        sys.exit(1)