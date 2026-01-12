import os
import random
import io
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image, ImageDraw, ImageFont
from matplotlib.transforms import Bbox
from .config import (
    output_dir,
    verification_dir,
    NUM_IMAGES_TO_GENERATE,
    MAX_VERIFICATION_IMAGES,
    IMG_WIDTH,
    IMG_HEIGHT,
    FIGURE_WIDTH,
    FIGURE_HEIGHT,
    DPI,
    CLASS_MAPPING,
    COLOR_MAPPING,
    TIMEFRAME_CONFIGS,
    TRAIN_VAL_RATIO,
)

def generate_random_ohlc_data(freq, num_periods, volatility, drift):
    """Generates random OHLC data with variable candle counts and occasional spikes/gaps."""
    base_price = np.random.uniform(50, 500)

    min_periods, max_periods = num_periods
    final_num_periods = np.random.randint(min_periods, max_periods + 1)

    base_date = pd.to_datetime('today') - pd.to_timedelta(np.random.randint(0, 365*5), 'd')
    time_offset = pd.to_timedelta(np.random.randint(0, 24), 'h') + \
                  pd.to_timedelta(np.random.randint(0, 60), 'm') + \
                  pd.to_timedelta(np.random.randint(0, 60), 's')
    start_date = base_date.replace(hour=0, minute=0, second=0) + time_offset

    dates = pd.date_range(start=start_date, periods=final_num_periods, freq=freq)
    returns = np.random.normal(loc=drift, scale=volatility, size=final_num_periods)

    # Add extreme volatility spikes (2% chance)
    if random.random() < 0.02:
        if final_num_periods > 2:
            spike_index = np.random.randint(1, final_num_periods - 1)
            spike_magnitude = np.random.uniform(8, 15) * volatility
            returns[spike_index] += spike_magnitude * np.random.choice([-1, 1])

    prices = base_price * (1 + returns).cumprod()

    opens = prices / (1 + returns)
    highs = np.maximum(opens, prices) + np.random.uniform(0, volatility/2, size=final_num_periods) * prices
    lows = np.minimum(opens, prices) - np.random.uniform(0, volatility/2, size=final_num_periods) * prices
    
    df = pd.DataFrame({'Open': opens, 'High': highs, 'Low': lows, 'Close': prices}, index=dates)

    # Add price gaps (1% chance)
    if random.random() < 0.01:
        gap_size = np.random.randint(1, max(2, int(final_num_periods * 0.1)))
        if final_num_periods > gap_size + 2:
             gap_start = np.random.randint(1, final_num_periods - gap_size - 1)
             df.iloc[gap_start:gap_start + gap_size] = np.nan

    return df

def calculate_sma(data, window):
    """Calculate Simple Moving Average."""
    return data.rolling(window=window).mean()

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def add_technical_indicators(df):
    """Add technical indicators to the dataframe."""
    indicators = {}
    data_length = len(df)
    
    # Only add indicators if we have enough data points
    if data_length < 10:  # Need minimum data for any indicators
        return indicators
    
    # Random chance to add SMA (10% chance)
    if random.random() < 0.1:
        # Choose SMA period that's appropriate for data length
        possible_periods = [p for p in [5, 10, 20, 50] if p < data_length * 0.8]
        if possible_periods:
            sma_period = random.choice(possible_periods)
            sma = calculate_sma(df['Close'], sma_period)
            # Only add if we have valid data
            if not sma.dropna().empty:
                indicators[f'SMA_{sma_period}'] = sma
    
    # Random chance to add Bollinger Bands (5% chance)
    if random.random() < 0.05:
        # Choose BB period that's appropriate for data length
        possible_periods = [p for p in [20, 25] if p < data_length * 0.8]
        if possible_periods:
            bb_period = random.choice(possible_periods)
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['Close'], bb_period)
            # Only add if we have valid data
            if not bb_middle.dropna().empty:
                indicators['BB_upper'] = bb_upper
                indicators['BB_middle'] = bb_middle
                indicators['BB_lower'] = bb_lower
    
    return indicators

def create_custom_dark_style():
    """Creates a custom dark-themed mplfinance style."""
    mc = mpf.make_marketcolors(up='#089981', down='#f43444', edge='inherit', wick='inherit')
    
    rc = {
        "figure.facecolor": "#151924",
        "grid.color": "#222631",
        "axes.labelcolor": "#b2b4be",
        "xtick.color": "#b2b4be",
        "ytick.color": "#b2b4be",
    }

    return mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors=mc,
        rc=rc,
        facecolor="#151924"
    )

def create_custom_light_style():
    """Creates a custom light-themed style."""
    mc = mpf.make_marketcolors(
        up='#32ea32',
        down='#C62727',
        edge='inherit',
        wick='inherit'
    )
    
    rc = {
        "figure.facecolor": "#ffffff",
        "grid.color": "#FBFBFB",
        "axes.labelcolor": "#505050",
        "xtick.color": "#505050",
        "ytick.color": "#505050",
    }
    
    s = mpf.make_mpf_style(
        base_mpf_style='yahoo',
        marketcolors=mc,
        rc=rc,
        facecolor="#ffffff"
    )
    
    s['candle_width'] = 0.4
    s['wick_width'] = 0.15
    s['volume_width'] = 0.4
    
    return s

def create_metatrader_style():
    """Creates a MetaTrader-style theme."""
    mc = mpf.make_marketcolors(
        up='#000000',
        down='#ffffff',
        edge='#01ff00',
        wick='#01ff00'
    )
    
    rc = {
        "figure.facecolor": "#000000",
        "grid.color": "#778899",
        "axes.labelcolor": "#ffffff",
        "xtick.color": "#ffffff",
        "ytick.color": "#ffffff",
        "axes.facecolor": "#000000",
    }
    
    return mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors=mc,
        rc=rc,
        facecolor="#000000"
    )

def add_chart_text_elements(fig, ax, df, style_choice):
    """Add text elements to charts (instrument names, OHLC values, etc.)."""
    is_dark_theme = ('dark' in getattr(style_choice, 'name', '') or 
                     style_choice == 'nightclouds' or
                     (hasattr(style_choice, '_style') and style_choice._style.get('facecolor') == '#000000'))
    text_color = '#cccccc' if is_dark_theme else '#333333'
    
    if not df.empty:
        latest_candle = df.iloc[-1]
        open_price = latest_candle['Open']
        high_price = latest_candle['High']
        low_price = latest_candle['Low']
        close_price = latest_candle['Close']
        
        instruments = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD', 
                      'NZDUSD', 'EURJPY', 'GBPJPY', 'EURGBP', 'BTCUSD', 'ETHUSD',
                      'AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'SPY']
        instrument_name = random.choice(instruments)
        
        if random.random() < 0.7:  # 70% chance
            fig.suptitle(instrument_name, fontsize=random.uniform(10, 14), 
                        color=text_color, x=random.uniform(0.1, 0.3), y=random.uniform(0.85, 0.95))
        
        if random.random() < 0.6:  # 60% chance
            ohlc_text = f"O: {open_price:.4f}  H: {high_price:.4f}  L: {low_price:.4f}  C: {close_price:.4f}"
            
            text_positions = [
                (0.02, 0.95),  # Top left
                (0.02, 0.90),  # Upper left
                (0.5, 0.95),   # Top center
                (0.7, 0.95),   # Top right
            ]
            x_pos, y_pos = random.choice(text_positions)
            
            fig.text(x_pos, y_pos, ohlc_text, fontsize=random.uniform(7, 10), 
                    color=text_color, transform=fig.transFigure)
        
        if random.random() < 0.4:  # 40% chance for extra text
            extra_texts = [
                f"Vol: {random.randint(10000, 999999)}",
                f"Spread: {random.uniform(0.1, 2.5):.1f}",
                f"1D: {random.uniform(-5, 5):+.2f}%",
                f"Bid: {close_price:.4f}",
                f"Ask: {close_price + random.uniform(0.0001, 0.01):.4f}",
                f"Change: {random.uniform(-0.05, 0.05):+.4f}",
            ]
            
            selected_text = random.choice(extra_texts)
            extra_positions = [
                (0.02, 0.85),  # Left side
                (0.02, 0.10),  # Bottom left
                (0.75, 0.90),  # Top right area
                (0.75, 0.10),  # Bottom right
            ]
            x_pos, y_pos = random.choice(extra_positions)
            
            fig.text(x_pos, y_pos, selected_text, fontsize=random.uniform(6, 9), 
                    color=text_color, transform=fig.transFigure)

def get_bounding_boxes(fig, ax, ohlc_df):
    renderer = fig.canvas.get_renderer()
    boxes = []

    plot_bbox = ax.get_tightbbox(renderer)
    boxes.append({'class_id': CLASS_MAPPING['plot_area'], 'box_pixels': plot_bbox.bounds})

    ytick_labels = ax.yaxis.get_ticklabels()
    if ytick_labels:
        valid_ytick_bboxes = []
        for label in ytick_labels:
            label_bbox = label.get_window_extent(renderer)
            is_valid = (label.get_text() != '' and label_bbox.width > 0 and label_bbox.height > 0 and
                        label_bbox.y1 <= plot_bbox.y1 and label_bbox.y0 >= plot_bbox.y0)
            if is_valid:
                valid_ytick_bboxes.append(label_bbox)
        if valid_ytick_bboxes:
            final_yaxis_bbox = Bbox.union(valid_ytick_bboxes)
            boxes.append({'class_id': CLASS_MAPPING['y_axis'], 'box_pixels': final_yaxis_bbox.bounds})

    xtick_labels = ax.xaxis.get_ticklabels()
    if xtick_labels:
        valid_xtick_bboxes = []
        for label in xtick_labels:
            label_bbox = label.get_window_extent(renderer)
            is_valid = (label.get_text() != '' and label_bbox.width > 0 and label_bbox.height > 0 and
                        label_bbox.x1 <= plot_bbox.x1 and label_bbox.x0 >= plot_bbox.x0)
            if is_valid:
                valid_xtick_bboxes.append(label_bbox)
        if valid_xtick_bboxes:
            final_xaxis_bbox = Bbox.union(valid_xtick_bboxes)
            boxes.append({'class_id': CLASS_MAPPING['x_axis'], 'box_pixels': final_xaxis_bbox.bounds})

    transform = ax.transData.transform
    candle_width_data = 0.8 * 0.5

    for i in range(len(ohlc_df)):
        row = ohlc_df.iloc[i]
        top_left_data = (i - candle_width_data, row['High'])
        bottom_right_data = (i + candle_width_data, row['Low'])
        pixel_coords = transform([top_left_data, bottom_right_data])
        px1, py1 = pixel_coords[0]
        px2, py2 = pixel_coords[1]
        final_bbox = Bbox.from_extents(px1, py2, px2, py1)
        boxes.append({'class_id': CLASS_MAPPING['candlestick'], 'box_pixels': final_bbox.bounds})
        
    return boxes

def convert_to_yolo_format(box, img_width, img_height):
    x, y, w, h = box['box_pixels']
    class_id = box['class_id']
    x_center = (x + w / 2) / img_width
    y_center = (img_height - (y + h / 2)) / img_height
    width_norm = w / img_width
    height_norm = h / img_height
    x_center, y_center = max(0.0, min(1.0, x_center)), max(0.0, min(1.0, y_center))
    width_norm, height_norm = max(0.0, min(1.0, width_norm)), max(0.0, min(1.0, height_norm))
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"

# =============================================================================
# MAIN WORKFLOW
# =============================================================================
if __name__ == '__main__':
    train_images_path = os.path.join(output_dir, "train", "images")
    train_labels_path = os.path.join(output_dir, "train", "labels")
    train_groundtruth_path = os.path.join(output_dir, "train", "groundtruth")
    val_images_path = os.path.join(output_dir, "val", "images")
    val_labels_path = os.path.join(output_dir, "val", "labels")
    val_groundtruth_path = os.path.join(output_dir, "val", "groundtruth")
    
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(train_groundtruth_path, exist_ok=True)
    os.makedirs(val_images_path, exist_ok=True)
    os.makedirs(val_labels_path, exist_ok=True)
    os.makedirs(val_groundtruth_path, exist_ok=True)
    os.makedirs(verification_dir, exist_ok=True)
    
    print(f"Starting data generation...")
    print(f"Output: '{output_dir}' | Verification: '{verification_dir}'")

    REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}
    
    custom_dark_style = create_custom_dark_style()
    custom_light_style = create_custom_light_style()
    metatrader_style = create_metatrader_style()

    all_styles = ['yahoo', 'charles', 'nightclouds', custom_dark_style, custom_light_style, metatrader_style]

    verification_saved_count = 0

    for i in range(NUM_IMAGES_TO_GENERATE):
        img_name = f"chart_{i}.png"
        label_name = f"chart_{i}.txt"
        groundtruth_name = f"chart_{i}.csv"
        
        is_train = random.random() < TRAIN_VAL_RATIO
        if is_train:
            img_filepath = os.path.join(train_images_path, img_name)
            label_filepath = os.path.join(train_labels_path, label_name)
            groundtruth_filepath = os.path.join(train_groundtruth_path, groundtruth_name)
        else:
            img_filepath = os.path.join(val_images_path, img_name)
            label_filepath = os.path.join(val_labels_path, label_name)
            groundtruth_filepath = os.path.join(val_groundtruth_path, groundtruth_name)

        random_tf_name = random.choice(list(TIMEFRAME_CONFIGS.keys()))
        config = TIMEFRAME_CONFIGS[random_tf_name]
        df = generate_random_ohlc_data(**config)
        
        df.to_csv(groundtruth_filepath)
        
        indicators = add_technical_indicators(df)
        
        style_choice = random.choice(all_styles)
        ax_position = 'left' if np.random.rand() > 0.5 else 'right'
        
        addplots = []
        if indicators:
            is_dark_theme = ('dark' in getattr(style_choice, 'name', '') or 
                             style_choice == 'nightclouds' or 
                             (hasattr(style_choice, '_style') and style_choice._style.get('facecolor') == '#000000'))
            
            indicator_colors = {
                'sma': '#FFD700' if is_dark_theme else '#FF8C00',  # Gold/Orange
                'bb_upper': '#FF6B6B' if is_dark_theme else '#DC143C',  # Light/Dark Red
                'bb_middle': '#4ECDC4' if is_dark_theme else '#008B8B',  # Teal
                'bb_lower': '#45B7D1' if is_dark_theme else '#4169E1',  # Blue
            }
            
            for name, data in indicators.items():
                # Only add indicators with valid data
                if data.dropna().empty:
                    continue
                    
                if name.startswith('SMA_'):
                    addplots.append(mpf.make_addplot(data, color=indicator_colors['sma'], width=1.5))
                elif name == 'BB_upper':
                    addplots.append(mpf.make_addplot(data, color=indicator_colors['bb_upper'], width=1))
                elif name == 'BB_middle':
                    addplots.append(mpf.make_addplot(data, color=indicator_colors['bb_middle'], width=1))
                elif name == 'BB_lower':
                    addplots.append(mpf.make_addplot(data, color=indicator_colors['bb_lower'], width=1))
        
        # Create the plot with or without indicators
        if addplots:
            fig, axlist = mpf.plot(df, type='candle', style=style_choice,
                                   figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
                                   addplot=addplots,
                                   returnfig=True, closefig=True)
        else:
            fig, axlist = mpf.plot(df, type='candle', style=style_choice,
                                   figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
                                   returnfig=True, closefig=True)
        
        fig.set_dpi(DPI)
        
        ax = axlist[0]
        ax.yaxis.set_ticks_position(ax_position)
        ax.yaxis.set_label_position(ax_position)

        # Add crosshairs (20% chance)
        if random.random() < 0.2:
            if not df.empty:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                crosshair_x = random.uniform(xlim[0], xlim[1])
                crosshair_y = random.uniform(ylim[0], ylim[1])

                is_dark_theme = ('dark' in getattr(style_choice, 'name', '') or 
                                 style_choice == 'nightclouds' or 
                                 style_choice == custom_dark_style or
                                 (hasattr(style_choice, '_style') and style_choice._style.get('facecolor') == '#000000'))
                line_color = '#b2b4be' if is_dark_theme else '#404040'

                ax.axvline(x=crosshair_x, color=line_color, linestyle='--', linewidth=0.6, alpha=0.8)
                ax.axhline(y=crosshair_y, color=line_color, linestyle='--', linewidth=0.6, alpha=0.8)

        # Add text elements (80% chance)
        if random.random() < 0.8:
            add_chart_text_elements(fig, ax, df, style_choice)

        fig.canvas.draw()
        
        boxes = get_bounding_boxes(fig, ax, df)
        
        with open(label_filepath, 'w') as f:
            for box in boxes:
                yolo_string = convert_to_yolo_format(box, IMG_WIDTH, IMG_HEIGHT)
                f.write(yolo_string + "\n")

        # Add JPEG compression artifacts (50% chance)
        if random.random() < 0.5:
            jpeg_quality = random.randint(40, 85)
            buffer = io.BytesIO()
            fig.savefig(buffer, format='jpg', dpi=DPI, pil_kwargs={'quality': jpeg_quality})
            buffer.seek(0)
            img_with_artifacts = Image.open(buffer).convert('RGB')
            img_with_artifacts.save(img_filepath, 'PNG')
            buffer.close()
        else:
            fig.savefig(img_filepath, dpi=DPI)

        plt.close(fig)

        # Generate verification image
        verify_img_path = img_filepath
        verify_label_path = label_filepath

        img = Image.open(verify_img_path)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        with open(verify_label_path, 'r') as f:
            for line in f.readlines():
                class_id_float, x_c, y_c, w, h = [float(p) for p in line.strip().split()]
                class_id = int(class_id_float)

                x = (x_c - w/2) * IMG_WIDTH
                y = (y_c - h/2) * IMG_HEIGHT
                x2 = x + w * IMG_WIDTH
                y2 = y + h * IMG_HEIGHT

                class_name = REVERSE_CLASS_MAPPING.get(class_id, "Unknown")
                color = COLOR_MAPPING.get(class_name, "yellow")

                draw.rectangle([(x, y), (x2, y2)], outline=color, width=2)

        if verification_saved_count < MAX_VERIFICATION_IMAGES:
            verified_filepath = os.path.join(verification_dir, f"VERIFIED_{img_name}")
            img.save(verified_filepath)
            verification_saved_count += 1

        if (i+1) % 5 == 0:
            print(f"Generated {i+1}/{NUM_IMAGES_TO_GENERATE}")

    print(f"\nProcess complete. Check verification images in '{verification_dir}'")