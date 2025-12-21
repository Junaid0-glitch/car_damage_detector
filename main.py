import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import io

# ---------------- CONFIG ----------------
API_URL = "https://junaid17-car-damage-detector.hf.space/predict_damage"

st.set_page_config(
    page_title="Car Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%) !important;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 1200px !important;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stFileUploader"] label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: white !important;
    }
    
    [data-testid="stFileUploader"] section {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    
    [data-testid="stFileUploader"] section > div {
        color: rgba(255, 255, 255, 0.8) !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: rgba(255, 255, 255, 0.6) !important;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 2rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        margin-top: 1rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 25px rgba(15, 32, 39, 0.5) !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        width: 100%;
        background: linear-gradient(135deg, #134E5E 0%, #71B280 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        margin-top: 1rem !important;
    }
    
    /* Image containers */
    [data-testid="stImage"] {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: white;
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Markdown text color */
    .stMarkdown {
        color: white;
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
    <div style="text-align: center; padding: 2rem 0 3rem 0;">
        <h1 style="font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🚗 Car Damage Detection
        </h1>
        <p style="font-size: 1.3rem; color: rgba(255, 255, 255, 0.9); font-weight: 400;">
            AI-powered damage analysis with visual insights
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------------- INFO BOXES ----------------
col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.08); backdrop-filter: blur(10px); border: 2px solid rgba(255, 255, 255, 0.15); padding: 1.8rem; border-radius: 12px; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);">
            <h3 style="color: white; margin-top: 0; font-size: 1.3rem;">📋 How it works</h3>
            <ol style="color: rgba(255, 255, 255, 0.9); line-height: 1.8; font-size: 1rem; margin-bottom: 0;">
                <li>Upload a clear image of your car</li>
                <li>Click the analyze button</li>
                <li>Get instant damage detection results</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("""
        <div style="background: linear-gradient(135deg, #FFA726 0%, #FB8C00 100%); padding: 1.8rem; border-radius: 12px; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);">
            <h3 style="color: white; margin-top: 0; font-size: 1.3rem;">⏱️ Please Note</h3>
            <p style="color: white; line-height: 1.8; font-size: 1rem; margin-bottom: 0; font-weight: 500;">
                The first request may take <strong>30-60 seconds</strong> due to API cold start. 
                Subsequent requests will be much faster!
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📁 Drag and drop or click to upload a car image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of your car for best results"
)

if uploaded_file:
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.markdown("""
            <h3 style="color: white; margin-top: 0; margin-bottom: 1rem; text-align: center; font-size: 1.5rem;">📸 Uploaded Image</h3>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("""
            <h3 style="color: white; margin-top: 0; margin-bottom: 1.5rem; font-size: 1.5rem;">📋 Image Details</h3>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2); margin-bottom: 1rem;">
                <p style="color: white; font-size: 1rem; margin: 0;">
                    <strong style="color: rgba(255, 255, 255, 0.8);">📄 Filename</strong><br>
                    <span style="font-size: 0.95rem; color: rgba(255, 255, 255, 0.9);">{uploaded_file.name}</span>
                </p>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2); margin-bottom: 1rem;">
                <p style="color: white; font-size: 1rem; margin: 0;">
                    <strong style="color: rgba(255, 255, 255, 0.8);">💾 File Size</strong><br>
                    <span style="font-size: 0.95rem; color: rgba(255, 255, 255, 0.9);">{uploaded_file.size / 1024:.2f} KB</span>
                </p>
            </div>
            
            <div style="background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); padding: 1.5rem; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.2); margin-bottom: 1rem;">
                <p style="color: white; font-size: 1rem; margin: 0;">
                    <strong style="color: rgba(255, 255, 255, 0.8);">📐 Dimensions</strong><br>
                    <span style="font-size: 0.95rem; color: rgba(255, 255, 255, 0.9);">{image.size[0]} x {image.size[1]} px</span>
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔍 Analyze Damage"):
            with st.spinner("🔄 Analyzing image with AI..."):
                try:
                    files = {
                        "image": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    }

                    response = requests.post(
                        API_URL,
                        files=files,
                        timeout=120
                    )
                    response.raise_for_status()
                    result = response.json()

                    st.session_state['result'] = result
                    st.session_state['original_image'] = image

                except requests.exceptions.RequestException as e:
                    st.error(f"❌ API Error: {str(e)}")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Unexpected Error: {str(e)}")
                    st.stop()

# ---------------- RESULTS DISPLAY ----------------
if 'result' in st.session_state:
    result = st.session_state['result']
    image = st.session_state['original_image']
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("""
        <h2 style="text-align: center; color: white; font-size: 2.5rem; margin: 2rem 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            📊 Analysis Results
        </h2>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%); padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);">
                <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;">Damage Type</p>
                <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0;">{result['damage_type']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%); padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);">
                <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;">Confidence</p>
                <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0;">{result['confidence']:.1%}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%); padding: 2rem; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);">
                <p style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 1px;">Detections</p>
                <p style="color: white; font-size: 2rem; font-weight: bold; margin: 0;">{len(result['bboxes'])}</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Draw bounding boxes
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    for i, box in enumerate(result["bboxes"]):
        x1, y1, x2, y2 = box["bbox"]
        conf = box["confidence"]

        # Draw rectangle with thicker lines
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 4)

        # Background for text
        label = f"Damage {i+1}: {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        )
        
        cv2.rectangle(
            img_np,
            (x1, y1 - text_height - 15),
            (x1 + text_width + 15, y1),
            (0, 255, 0),
            -1
        )
        
        cv2.putText(
            img_np,
            label,
            (x1 + 7, y1 - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    st.markdown("""
        <h3 style="color: white; font-size: 1.8rem; margin: 2rem 0 1rem 0; text-align: center;">
            🎯 Detected Damage Visualization
        </h3>
    """, unsafe_allow_html=True)
    
    st.image(img_rgb, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detection details
    with st.expander("📋 View Detection Details", expanded=False):
        st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 10px;">
        """, unsafe_allow_html=True)
        
        for i, box in enumerate(result["bboxes"]):
            st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 1rem 0; border-left: 4px solid #2c5364;">
                    <h4 style="color: #0f2027; margin-top: 0;">Detection {i+1}</h4>
                    <p style="color: #333; margin: 0.5rem 0;"><strong>Confidence:</strong> {box['confidence']:.2%}</p>
                    <p style="color: #333; margin: 0.5rem 0;"><strong>Bounding Box:</strong> {box['bbox']}</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Download button
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="PNG")
    st.download_button(
        label="⬇️ Download Annotated Image",
        data=buf.getvalue(),
        file_name="damage_detection_result.png",
        mime="image/png",
        use_container_width=True
    )

# ---------------- FOOTER ----------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: rgba(255, 255, 255, 0.8); padding: 2rem;">
        <p style="font-size: 1rem;">Powered by AI | Built with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)