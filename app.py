import streamlit as st
import main_pipeline as mp
from debug import draw_predictions_with_labels, visualize_y_axis_mapping_with_ocr
from io import StringIO
import contextlib
import tempfile
import os
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
from streamlit_paste_button import paste_image_button

st.set_page_config(layout="wide", page_title="Candlestick Extraction")

class StreamlitOutput:
    def __init__(self):
        self.buffer = StringIO()

    def write(self, text):
        self.buffer.write(text)

    def flush(self):
        pass
    
    def get_value(self):
        return self.buffer.getvalue()

@st.cache_resource
def load_model(model_path):
    return mp.load_model(model_path)

# Initialize Model
model = load_model(mp.MODEL_PATH)

# --- Pages ---

def intro_page():
    st.title("Welcome to Candlestick AI Extractor")
    
    st.markdown("""
    ### 📈 Digitize Financial Charts in Seconds
    
    This application uses advanced Computer Vision and AI to extract structured OHLC (Open, High, Low, Close) data from static images of candlestick charts.
    
    ---
    
    #### 🚀 Key Capabilities
    *   **YOLOv10 Object Detection**: Precisely locates candlesticks, wicks, and chart axes.
    *   **Intelligent OCR Integration**: Reads price scales and automatically maps pixel coordinates to financial values.
    *   **Time Axis Reconstruction**: Infers time intervals or allows manual configuration for accurate timestamping.
    *   **Visual Verification**: Provides detailed debug overlays to trust the AI's output.
    
    #### 📖 How to Use
    1.  Go to the **Extraction Tool** page using the sidebar menu.
    2.  **Upload** an image file (PNG, JPG).
    3.  Open the **Configuration** sidebar to set the chart's *Start Date*, *Time*, and *Interval* (e.g., 1 hour, 1 day).
    4.  Click **Run Extraction Pipeline**.
    5.  Interact with the results:
        *   View the **Reconstructed Chart** (Zoom/Pan).
        *   Inspect the **Detected Objects** visualization.
        *   Download the extracted data as **CSV**.
        
    ---
    *Built with Streamlit, Ultralytics YOLO, and EasyOCR.*
    """)

def extraction_page():
    st.title('Candlestick OHLC Extraction')
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        img_file = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
    with col2:
        st.write("## ") # Spacing to align with uploader
        pasted_img = paste_image_button("Paste Image", key="paster")

    # Create sidebar for configuration (only for extraction page)
    with st.sidebar:
        st.header("Configuration")
        st.subheader("Time Inference")
        use_time = st.checkbox("Generate Time Axis", value=True)
        
        start_date = st.date_input("Start Date", value=pd.to_datetime("today"))
        start_time = st.time_input("Start Time", value=pd.to_datetime("09:30").time())
        interval_str = st.selectbox("Interval", ["1h", "1d", "15m", "30m", "5m", "1m"], index=0)

    image_to_process = None
    if img_file is not None:
        image_to_process = img_file
    elif pasted_img.image_data is not None:
        image_to_process = pasted_img.image_data

    if image_to_process is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        
        if hasattr(image_to_process, 'getbuffer'):
            # It's an UploadedFile from st.file_uploader
            tfile.write(image_to_process.getbuffer())
        else:
            # It's a PIL Image from pasted_img.image_data
            if image_to_process.mode == 'RGBA':
                image_to_process = image_to_process.convert('RGB')
            image_to_process.save(tfile.name, format="JPEG")
            
        temp_image_path = tfile.name
        tfile.close()

        # Display Original Image
        st.subheader("Original Image")
        st.image(image_to_process, use_container_width=True)

        if st.button("Run Extraction Pipeline", type="primary"):
            st.divider()
            
            # Capture logs
            log_capture = StreamlitOutput()
            
            with st.spinner("Running AI Pipeline..."):
                try:
                    with contextlib.redirect_stdout(log_capture):
                        # 1. Prediction
                        print("Step 1: Running Object Detection...")
                        predictions, model_obj = mp.run_model_predictions(model, temp_image_path)
                        
                        chart_data = None
                        if predictions:
                            # 2. Extraction
                            print("Step 2: Extracting Structured Data & OCR...")
                            chart_data = mp.extract_structured_data(predictions, temp_image_path, model_obj)
                        
                    # layout cols
                    col_viz, col_data = st.columns([1, 1])
                    
                    # --- VISUALIZATION COLUMN ---
                    with col_viz:
                        st.subheader("Visual Analysis")
                        
                        if predictions:
                            # Bounding Boxes
                            annotated_img = draw_predictions_with_labels(temp_image_path, predictions, model_obj)
                            st.image(annotated_img, caption="Detected Components", use_container_width=True)
                            
                            # Y-Axis Debug
                            if chart_data:
                                y_axis_box = chart_data.get('component_boxes', {}).get('y_axis')
                                if y_axis_box is not None:
                                    with st.expander("Show Y-Axis OCR Details"):
                                        ocr_img, _ = visualize_y_axis_mapping_with_ocr(temp_image_path, y_axis_box, chart_data)
                                        if ocr_img:
                                            st.image(ocr_img, caption="Y-Axis Mapping Logic", use_container_width=True)
                        else:
                            st.warning("No objects detected.")

                    # --- DATA COLUMN ---
                    with col_data:
                        st.subheader("Extracted Data")
                        
                        if chart_data:
                            ohlc_data = chart_data.get('ohlc_data', [])
                            if ohlc_data:
                                df = pd.DataFrame(ohlc_data)
                                
                                # --- Add Time Column ---
                                if use_time:
                                    try:
                                        # Combine date and time
                                        start_dt = pd.Timestamp.combine(start_date, start_time)
                                        
                                        # Create date range
                                        # Determine frequency offset
                                        freq_map = {
                                            "1m": "T", "5m": "5T", "15m": "15T", "30m": "30T", 
                                            "1h": "H", "1d": "D"
                                        }
                                        freq = freq_map.get(interval_str, "H")
                                        
                                        # Generate timestamps based on count of candles
                                        timestamps = pd.date_range(start=start_dt, periods=len(df), freq=freq)
                                        df.insert(0, 'Date', timestamps)
                                        
                                        # Sort by Date just in case, though id order is assumed correct
                                        # df = df.sort_values('Date')
                                        print(f"Generated {len(df)} timestamps starting from {start_dt}")
                                        
                                    except Exception as e:
                                        st.warning(f"Could not generate time axis: {e}")

                                # Interactive Plotly Chart
                                x_axis = df['Date'] if 'Date' in df.columns else df['id']
                                
                                fig = go.Figure(data=[go.Candlestick(
                                    x=x_axis,
                                    open=df['open'],
                                    high=df['high'],
                                    low=df['low'],
                                    close=df['close']
                                )])
                                
                                title_text = f'Reconstructed Chart ({interval_str})' if use_time else 'Reconstructed Chart'
                                fig.update_layout(title=title_text, xaxis_title='Date' if use_time else 'Candle ID', yaxis_title='Price', height=400)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Data Table
                                st.dataframe(df, use_container_width=True, height=300)
                                
                                # CSV Download
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download CSV",
                                    data=csv,
                                    file_name='extracted_ohlc.csv',
                                    mime='text/csv',
                                )
                            else:
                                st.info("Structure detected, but no OHLC candles extracted.")
                        else:
                            st.error("Extraction failed.")

                    # --- LOGS ---
                    with st.expander("See Execution Logs"):
                        st.code(log_capture.get_value(), language="bash")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    
        # Cleanup
        # os.remove(temp_image_path) # Commented out to allow debugging if needed, or re-enable production cleanup

# --- Main Navigation ---
page_names_to_funcs = {
    "Introduction": intro_page,
    "Extraction Tool": extraction_page,
}

selected_page = st.sidebar.radio("Go to", list(page_names_to_funcs.keys()))
page_names_to_funcs[selected_page]()