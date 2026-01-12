"""
Debug visualization functions for chart data extraction pipeline
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from generate.config import CLASS_MAPPING, COLOR_MAPPING
from axis_mapping import get_pixel_to_price_map_easyocr


def draw_predictions_with_labels(image_path, predictions, model, chart_data=None, save_path=None):
    """Draw predicted bounding boxes with class labels and confidence scores"""
    id_to_class_map = {v: k for k, v in CLASS_MAPPING.items()}
    
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    if isinstance(predictions, dict):
        boxes = predictions.get('boxes', [])
        confidences = predictions.get('confidences', [])
        class_ids = predictions.get('class_ids', [])
    else:
        boxes = predictions.boxes.xyxy.cpu().numpy()
        confidences = predictions.boxes.conf.cpu().numpy()
        class_ids = predictions.boxes.cls.cpu().numpy().astype(int)
    
    print(f"\nDrawing {len(boxes)} predicted objects:")
    
    # Draw standard predictions (Skip candlesticks if we have chart_data with detailed info)
    for i, box in enumerate(boxes):
        class_id = int(class_ids[i])
        class_name = id_to_class_map.get(class_id, 'Unknown')
        
        if chart_data and class_name == 'candlestick':
            continue

        confidence = confidences[i]
        
        x1, y1, x2, y2 = box
        color = COLOR_MAPPING.get(class_name, 'yellow')
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        label = f"{confidence:.2f}"
        text_bbox = draw.textbbox((x1, y1 - 25), label, font=font)
        draw.rectangle(text_bbox, fill=color, outline=color)
        draw.text((x1, y1 - 25), label, fill='white', font=font)

    # Draw candlesticks from chart_data with specific coloring for interpolated ones
    if chart_data and 'candlestick_info' in chart_data:
        candlesticks = chart_data['candlestick_info']
        print(f"Drawing {len(candlesticks)} candlesticks (including interpolated)")
        
        for candle in candlesticks:
            x1, y1, x2, y2 = candle['bbox']
            is_interpolated = candle.get('interpolated', False)
            
            if is_interpolated:
                color = 'cyan'
                label = "Interpolated"
            else:
                color = COLOR_MAPPING.get('candlestick', 'green')
                label = "Candle"
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Optional: Draw label only if needed, or maybe just color is enough
            # text_bbox = draw.textbbox((x1, y1 - 15), label, font=font)
            # draw.rectangle(text_bbox, fill=color, outline=color)
            # draw.text((x1, y1 - 15), label, fill='black', font=font)
    
    if save_path:
        img.save(save_path)
        print(f"\nPrediction visualization saved to: {save_path}")
    
    return img

def visualize_y_axis_mapping_with_ocr(image_path, y_axis_bbox, chart_data=None, save_path=None):
    """Visualize the y-axis region with OCR results and price mapping lines on the full chart"""
    if y_axis_bbox is None or len(y_axis_bbox) == 0:
        print("No y-axis detected for price mapping visualization")
        return None
    
    if chart_data and 'ocr_results' in chart_data:
        ocr_results = chart_data['ocr_results']
        price_map_func = chart_data.get('price_mapper')
    else:
        print("Warning: chart_data not provided, running OCR again")
        price_map_func, ocr_results = get_pixel_to_price_map_easyocr(image_path, y_axis_bbox)
    
    if not ocr_results:
        print("No OCR text found in y-axis region")
        return None
    
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        font_large = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_large = ImageFont.load_default()
    
    print(f"\nY-Axis OCR Analysis:")
    print(f"Y-axis bbox: {y_axis_bbox}")
    print(f"Found {len(ocr_results)} OCR text items:")
    
    plot_area_bbox = chart_data.get('component_boxes', {}).get('plot_area') if chart_data else None
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, ocr_item in enumerate(ocr_results):
        text = ocr_item['text']
        confidence = ocr_item['confidence']
        center_y = ocr_item['center_y_absolute']
        price_value = ocr_item.get('price')
        used_for_mapping = ocr_item.get('used_for_mapping', False)
        color = colors[i % len(colors)]
        
        print(f"  {i+1}. Text: '{text}' | Confidence: {confidence:.2f} | Y-pixel: {center_y:.1f}", end="")
        
        if used_for_mapping and price_value is not None:
            print(f" | Price: {price_value}")
        elif not used_for_mapping and confidence < 0.85:
            print(f" | (filtered: low confidence)")
        else:
            print(f" | (filtered: invalid)")
        
        abs_bbox = ocr_item['bbox_absolute']
        bbox_points = [coord for coord in abs_bbox]
        bbox_points.append(abs_bbox[0])
        draw.line(bbox_points, fill=color, width=2)
        draw.text((abs_bbox[0][0] - 50, center_y - 10), f"{i+1}:{text}", fill=color, font=font)
        
        if used_for_mapping and price_value is not None and plot_area_bbox is not None:
            plot_x1, plot_y1, plot_x2, plot_y2 = plot_area_bbox
            draw.line([(plot_x1, center_y), (plot_x2, center_y)], fill=color, width=2)
            draw.text((plot_x2 + 5, center_y - 8), f"${price_value:.2f}", fill=color, font=font_large)
            draw.text((plot_x1 - 80, center_y - 8), f"Y={int(center_y)}", fill=color, font=font)
    
    x1, y1, x2, y2 = y_axis_bbox
    draw.rectangle([x1, y1, x2, y2], outline='cyan', width=3)
    draw.text((x1, y1 - 20), "Y-Axis Region", fill='cyan', font=font_large)
    
    if plot_area_bbox is not None:
        px1, py1, px2, py2 = plot_area_bbox
        draw.rectangle([px1, py1, px2, py2], outline='magenta', width=3)
        draw.text((px1, py1 - 20), "Plot Area", fill='magenta', font=font_large)
    
    if price_map_func:
        print(f"\nDEBUG")
        print(f"\nTesting price mapping function:")
        test_y_pixels = [y1 + 20, (y1 + y2) / 2, y2 - 20]
        for test_y in test_y_pixels:
            try:
                mapped_price = price_map_func(test_y)
                print(f"  Y-pixel {test_y:.0f} -> Price: {mapped_price:.2f}")
            except Exception as e:
                print(f"  Y-pixel {test_y:.0f} -> Error: {e}")
    
    if save_path:
        img.save(save_path)
        print(f"\nY-axis OCR visualization saved to: {save_path}")
    
    return img, ocr_results


def create_debug_summary(image_path, predictions, chart_data, save_path=None):
    """Create a comprehensive debug summary with all visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    img = Image.open(image_path)
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    try:
        pred_img = draw_predictions_with_labels(image_path, predictions, None)
        axes[0, 1].imshow(pred_img)
        axes[0, 1].set_title('Predicted Objects')
        axes[0, 1].axis('off')
    except Exception as e:
        print(f"Error creating predictions overlay: {e}")
        axes[0, 1].text(0.5, 0.5, f'Predictions overlay failed: {e}', ha='center', va='center')
        axes[0, 1].set_title('Predicted Objects (Failed)')
    
    y_axis_bbox = chart_data.get('component_boxes', {}).get('y_axis') if chart_data else None
    if y_axis_bbox is not None and len(y_axis_bbox) > 0:
        try:
            ocr_img, ocr_results = visualize_y_axis_mapping_with_ocr(image_path, y_axis_bbox, chart_data)
            axes[1, 0].imshow(ocr_img)
            axes[1, 0].set_title('Y-Axis OCR Analysis')
            axes[1, 0].axis('off')
            
            axes[1, 1].axis('off')
            summary_text = "OCR Results Summary:\n\n"
            for i, ocr_item in enumerate(ocr_results[:10]):
                summary_text += f"{i+1}. '{ocr_item['text']}' (conf: {ocr_item['confidence']:.2f})\n"
                summary_text += f"    Y-pixel: {ocr_item['center_y_absolute']:.1f}\n\n"
            
            axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                           fontsize=10, verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title('OCR Text Summary')
        except Exception as e:
            print(f"Error in Y-axis analysis: {e}")
            axes[1, 0].text(0.5, 0.5, f'Y-axis analysis failed: {e}', ha='center', va='center')
            axes[1, 0].set_title('Y-Axis Analysis (Error)')
            axes[1, 1].text(0.5, 0.5, 'No OCR data available', ha='center', va='center')
            axes[1, 1].set_title('OCR Summary (Error)')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Y-axis detected', ha='center', va='center')
        axes[1, 0].set_title('Y-Axis Analysis (Failed)')
        axes[1, 1].text(0.5, 0.5, 'No OCR data available', ha='center', va='center')
        axes[1, 1].set_title('OCR Summary (Failed)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Debug summary saved to: {save_path}")
    
    return fig