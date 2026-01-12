"""Configuration module for chart dataset generation"""

output_dir = "chart_dataset_v2.11"
verification_dir = "verification_images_v2.11"

NUM_IMAGES_TO_GENERATE = 2000
MAX_VERIFICATION_IMAGES = 20

# Figure configuration for chart generation
FIGURE_WIDTH = 14
FIGURE_HEIGHT = 8
DPI = 80 

# Calculate actual image dimensions based on figure size and DPI
IMG_WIDTH = int(FIGURE_WIDTH * DPI)
IMG_HEIGHT = int(FIGURE_HEIGHT * DPI)

CLASS_MAPPING = {'candlestick': 0, 'plot_area': 1, 'y_axis': 2, 'x_axis': 3}
COLOR_MAPPING = {'candlestick': 'red', 'plot_area': 'blue', 'y_axis': 'green', 'x_axis': 'purple'}

TIMEFRAME_CONFIGS = {
    # num_periods is a (min, max) tuple
    "15 Seconds": {'freq': '15s', 'num_periods': (6, 220), 'volatility': 0.0005, 'drift': 0.0},
    "5 Minutes":  {'freq': '5min', 'num_periods': (6, 220), 'volatility': 0.002, 'drift': 0.0},
    "15 Minutes": {'freq': '15min', 'num_periods': (6, 220), 'volatility': 0.004, 'drift': 0.0},
    "1 Hour":     {'freq': 'h', 'num_periods': (6, 220), 'volatility': 0.008, 'drift': 0.0},
    "4 Hours":    {'freq': '4h', 'num_periods': (6, 220), 'volatility': 0.012, 'drift': 0.0},
    "Daily":      {'freq': 'D', 'num_periods': (6, 220), 'volatility': 0.02, 'drift': 0.0005},
}

TRAIN_VAL_RATIO = 0.5
