import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# Cache the model loading for better performance
@st.cache_resource
def load_ml_model():
    return load_model("ham10000_model.h5")

model = load_ml_model()

# Configuration
CLASS_NAMES = {
    0: "Melanocytic Nevi",
    1: "Melanoma",
    2: "Benign Keratosis",
    3: "Basal Cell Carcinoma",
    4: "Actinic Keratoses",
    5: "Vascular Lesions",
    6: "Dermatofibroma"
}

SEVERITY_COLORS = {
    "Melanoma": "#ff4b4b",
    "Basal Cell Carcinoma": "#ff4b4b",
    "Actinic Keratoses": "#ff914d",
    "Vascular Lesions": "#ff914d",
    "Melanocytic Nevi": "#2ecc71",
    "Benign Keratosis": "#2ecc71",
    "Dermatofibroma": "#2ecc71"
}

RECOMMENDATIONS = {
    "Melanoma": "Urgent dermatologist consultation required. Please seek medical attention immediately.",
    "Basal Cell Carcinoma": "Urgent consultation needed. Schedule with a dermatologist within 48 hours.",
    "Actinic Keratoses": "Consult a dermatologist within 1 week. Use sun protection.",
    "Vascular Lesions": "Recommend clinical evaluation. Monitor for changes.",
    "Melanocytic Nevi": "Routine monitoring suggested. Annual skin check recommended.",
    "Benign Keratosis": "No immediate action needed. Monitor for changes.",
    "Dermatofibroma": "No treatment required. Cosmetic removal options available."
}

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 8px; }
    .highlight-box { 
        padding: 1rem; 
        border-radius: 10px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .warning { background-color: #fff3cd; border-left: 4px solid #ffc107; }
    .danger { background-color: #f8d7da; border-left: 4px solid #dc3545; }
    .safe { background-color: #d4edda; border-left: 4px solid #28a745; }
</style>
""", unsafe_allow_html=True)

def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

def generate_pdf_report(image, diagnosis, confidence, recommendation):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Convert PIL Image to ReportLab Image
    img_buffer = BytesIO()
    image.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    rl_img = RLImage(img_buffer, width=200, height=200)
    
    story.append(Paragraph("DermaScan AI Diagnostic Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(rl_img)
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Diagnosis: {diagnosis}", styles['Heading2']))
    story.append(Paragraph(f"Confidence: {confidence:.2f}%", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Recommendation:", styles['Heading3']))
    story.append(Paragraph(recommendation, styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Sidebar
with st.sidebar:
    st.header("About DermaScan AI")
    st.markdown("""
    - AI-powered skin condition analysis
    - Developed by Dermatologists & AI Experts
    - Version 1.2
    """)
    
    st.divider()
    st.subheader("Skin Care Tips")
    st.markdown("""
    - Regular self-examinations
    - Use SPF 30+ sunscreen daily
    - Stay hydrated
    - Monitor changes in moles
    """)

# Main Interface
st.title("🔬 DermaScan AI")
st.markdown("Upload a clear photo of your skin concern for instant analysis")

uploaded_file = st.file_uploader("Choose skin image", type=["png", "jpg", "jpeg"], 
                               help="Max file size: 5MB, Supported formats: JPG/PNG/JPEG")

if uploaded_file is not None:
    try:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("File size exceeds 5MB limit")
            st.stop()
            
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)
            
        with col2:
            if st.button("Analyze Now", type="primary"):
                with st.spinner("🔍 Analyzing skin patterns..."):
                    start_time = time.time()
                    processed_img = preprocess_image(image)
                    prediction = model.predict(processed_img)
                    processing_time = time.time() - start_time
                    
                probabilities = prediction[0]
                sorted_indices = np.argsort(probabilities)[::-1]
                
                # Main Diagnosis
                main_class = CLASS_NAMES[sorted_indices[0]]
                confidence = probabilities[sorted_indices[0]] * 100
                
                # Result Display
                st.subheader("Analysis Results")
                color = SEVERITY_COLORS.get(main_class, "#000000")
                st.markdown(f"""
                <div class="highlight-box" style="border-left-color: {color};">
                    <h3 style="color: {color}; margin-top: 0;">{main_class}</h3>
                    <p>Confidence: <strong>{confidence:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations
                st.subheader("Recommended Actions")
                rec_class = "danger" if "Urgent" in RECOMMENDATIONS[main_class] else "warning" if "Consult" in RECOMMENDATIONS[main_class] else "safe"
                st.markdown(f"""
                <div class="highlight-box {rec_class}">
                    <p>{RECOMMENDATIONS[main_class]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed Probabilities
                with st.expander("Detailed Breakdown"):
                    prob_df = pd.DataFrame({
                        "Condition": [CLASS_NAMES[i] for i in sorted_indices],
                        "Probability": [f"{probabilities[i]*100:.2f}%" for i in sorted_indices]
                    })
                    st.dataframe(prob_df, hide_index=True, use_container_width=True)
                
                # Report Generation
                pdf_report = generate_pdf_report(image, main_class, confidence, RECOMMENDATIONS[main_class])
                st.download_button(
                    label="📄 Download Full Report",
                    data=pdf_report,
                    file_name="dermascan_report.pdf",
                    mime="application/pdf"
                )
                
                st.markdown(f"*Analysis completed in {processing_time:.2f} seconds*")
                
                # Help Section
                st.divider()
                st.markdown("Need help? [Find a dermatologist](https://www.aad.org/)")
                
                # Store in session history
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append({
                    "image": image,
                    "diagnosis": main_class,
                    "confidence": confidence
                })
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# History Section
if 'history' in st.session_state and st.session_state.history:
    st.sidebar.divider()
    st.sidebar.subheader("Analysis History")
    for idx, entry in enumerate(st.session_state.history[::-1]):
        with st.sidebar:
            st.caption(f"Scan {idx+1}: {entry['diagnosis']} ({entry['confidence']:.1f}%)")
            st.image(entry['image'], width=100)