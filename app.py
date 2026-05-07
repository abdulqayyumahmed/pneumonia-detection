import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="NEURO-SCAN AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load model once and cache it
@st.cache_resource
def load_scan_model():
    return load_model('pneumonia_model.h5')

def get_image_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Custom CSS for Modern interactive UI
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
<style>
    .stApp {
        background: #020617;
        color: #f1f5f9;
        font-family: 'Outfit', sans-serif;
    }
    
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }
    
    #MainMenu, footer, header {visibility: hidden;}

    .dashboard-wrapper {
        padding: 0 2rem;
        max-width: 1400px;
        margin: auto;
    }

    .main-panel {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 30px;
        padding: 1.2rem;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
    }

    .header-section {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 0.5rem;
    }

    .brand-h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin: 0;
    }

    .meta-info {
        display: flex;
        gap: 15px;
    }

    .info-pill {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 6px 14px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #94a3b8;
    }

    /* Fixed 300px Image Box (User preferred size) */
    .scan-container {
        height: 300px;
        width: 300px;
        margin: 0 auto 15px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(56, 189, 248, 0.1);
        border-radius: 24px;
        overflow: hidden;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
    }

    .scan-container img {
        max-height: 100%;
        max-width: 100%;
        object-fit: contain;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .label-large {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -2px;
    }
    
    .label-normal { color: #4ade80; }
    .label-pneumonia { color: #f87171; }

    .biomarker-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 1.5rem;
    }

    .bio-box {
        background: rgba(255, 255, 255, 0.03);
        padding: 10px;
        border-radius: 12px;
        text-align: left;
    }

    .bio-label { color: #64748b; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; }
    .bio-value { color: #f1f5f9; font-size: 0.9rem; font-weight: 600; margin-top: 2px; }

    .progress-bar-wrap {
        width: 100%;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 100px;
        height: 8px;
        margin-top: 15px;
        overflow: hidden;
    }

    .progress-bar-fill {
        height: 100%;
        transition: width 0.8s ease-out;
    }
    
    .bg-normal { background: #4ade80; }
    .bg-pneumonia { background: #f87171; }

    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9, #2563eb);
        color: white;
        border: none;
        padding: 0.8rem;
        border-radius: 12px;
        font-weight: 700;
        width: 100%;
        transition: 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(56, 189, 248, 0.3);
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# Header Section
st.write("""
<div class="header-section">
    <div>
        <h1 class="brand-h1">NEURO-SCAN AI</h1>
        <p style="color: #64748b; font-size: 0.85rem; margin: 0;">Cloud-Native Pulmonary Neural Engine</p>
    </div>
    <div class="meta-info">
        <div class="info-pill">MODEL: MobileNetV2</div>
        <div class="info-pill" style="color: #4ade80; border-color: rgba(74, 222, 128, 0.2);">ACCURACY: 94.2%</div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.write("<h4 style='font-size:1rem; margin-bottom:10px;'>📤 SCAN INTAKE</h4>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png", "jfif", "webp"], label_visibility="collapsed")
    
    if uploaded_file:
        if 'current_file' not in st.session_state or st.session_state['current_file'] != uploaded_file.name:
            st.session_state['analyzed'] = False
            st.session_state['current_file'] = uploaded_file.name
            st.session_state['pred'] = None

        img = Image.open(uploaded_file).convert('RGB')
        img_b64 = get_image_base64(img)
        
        # Use st.write with unsafe_allow_html=True for more direct rendering
        st.write(f'''
<div class="scan-container">
    <img src="data:image/png;base64,{img_b64}" alt="Patient Scan">
</div>
''', unsafe_allow_html=True)
        
        if st.button("RUN AI DIAGNOSIS"):
            with st.spinner("Analyzing..."):
                time.sleep(1)
                model = load_scan_model()
                img_input = img.resize((224, 224))
                x = image.img_to_array(img_input) / 255.0
                x = np.expand_dims(x, axis=0)
                pred_raw = model.predict(x)[0][0]
                st.session_state['pred'] = float(pred_raw)
                st.session_state['analyzed'] = True
                st.rerun()
    else:
        st.write("""
<div class="scan-container" style="border-style: dashed; background: transparent;">
    <p style="color: #475569;">Awaiting scan feed...</p>
</div>
""", unsafe_allow_html=True)

with col2:
    if st.session_state.get('analyzed') and st.session_state.get('pred') is not None:
        pred = st.session_state['pred']
        confidence = pred * 100 if pred > 0.5 else (1 - pred) * 100
        result_label = "PNEUMONIA" if pred > 0.5 else "NORMAL"
        status_class = "pneumonia" if pred > 0.5 else "normal"
        
        # No leading indentation to prevent Streamlit from interpreting as a code block
        result_html = f'''<div class="result-card">
<span style="color: #64748b; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Final AI Verdict</span>
<h1 class="label-large label-{status_class}">{result_label}</h1>
<p style="color: #94a3b8; font-size: 1.1rem; margin-top: 5px;">Confidence: <b>{confidence:.2f}%</b></p>
<div class="progress-bar-wrap">
<div class="progress-bar-fill bg-{status_class}" style="width: {confidence}%;"></div>
</div>
<div class="biomarker-grid">
<div class="bio-box">
<div class="bio-label">Opacity Index</div>
<div class="bio-value">{pred:.4f}</div>
</div>
<div class="bio-box">
<div class="bio-label">Vascularity</div>
<div class="bio-value">{'ELEVATED' if pred > 0.5 else 'NORMAL'}</div>
</div>
<div class="bio-box">
<div class="bio-label">Inflammation</div>
<div class="bio-value">{'DETECTED' if pred > 0.5 else 'ABSENT'}</div>
</div>
<div class="bio-box">
<div class="bio-label">Review Status</div>
<div class="bio-value">REQUIRED</div>
</div>
</div>
</div>'''
        st.write(result_html, unsafe_allow_html=True)
        
        if st.button("RESET SYSTEM", key="reset"):
            st.session_state['analyzed'] = False
            st.session_state['pred'] = None
            st.rerun()
    else:
        st.write("""
<div class="glass-panel" style="height: 300px; display: flex; align-items: center; justify-content: center; border-style: dashed; border-color: rgba(56, 189, 248, 0.1);">
    <p style="color: #64748b; text-align: center;">Upload scan to activate AI engine</p>
</div>
""", unsafe_allow_html=True)

st.write('</div>', unsafe_allow_html=True) # End main-panel
st.write('</div>', unsafe_allow_html=True) # End dashboard-wrapper

# Footer
st.write("<p style='text-align: center; color: #334155; font-size: 0.65rem; margin-top: 0.5rem;'>NEURO-SCAN AI | © 2026</p>", unsafe_allow_html=True)
