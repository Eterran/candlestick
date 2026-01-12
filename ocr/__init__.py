import easyocr
from PIL import Image
import numpy as np
import cv2

reader = easyocr.Reader(['en'])

def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """
    Applies a series of pre-processing steps to an image to improve OCR accuracy.
    """
    # Convert PIL Image to an OpenCV array (in BGR format)
    cv_image = np.array(image.convert('RGB'))
    
    # 1. Grayscale
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # 2. Upscale the image - This is often the most critical step
    # INTER_CUBIC is a good general-purpose interpolation for enlarging.
    height, width = gray.shape
    upscaled = cv2.resize(gray, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)
    
    # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This is great for enhancing local contrast without blowing out the whole image.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(upscaled)
    
    # Optional 4: Binarization (can sometimes help, sometimes hurt - experiment with this)
    # _, thresh = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert back to PIL Image
    return Image.fromarray(contrast_enhanced) # or return Image.fromarray(thresh)



def get_candle_color(image, candle_bbox):
    """
    Determines if a candle is 'up' (e.g., green) or 'down' (e.g., red).
    Returns 'up', 'down', or 'unknown'.
    """
    x1, y1, x2, y2 = [int(coord) for coord in candle_bbox]
    # Crop a small central part of the candle body to avoid wicks and edges
    body_x1 = x1 + (x2 - x1) * 0.4
    body_x2 = x1 + (x2 - x1) * 0.6
    body_y1 = y1 + (y2 - y1) * 0.2
    body_y2 = y1 + (y2 - y1) * 0.8
    
    candle_body = image.crop((body_x1, body_y1, body_x2, body_y2))
    
    candle_arr = np.array(candle_body.convert('RGB'))
    if candle_arr.size == 0:
        return 'unknown'
        
    avg_color = np.mean(candle_arr, axis=(0, 1))
    r, g, b = avg_color

    if g > r and g > b:
        return 'up'
    if r > g and r > b:
        return 'down'
    
    return 'unknown'

if __name__ == '__main__':
    print('no main function')