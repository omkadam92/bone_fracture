import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from streamlit_option_menu import option_menu
import plotly.graph_objects as go

# Load the TensorFlow model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("C:\\Users\\hp\\Desktop\\All_ Codes_and_Papers\\PROJECTS_ONLY\\bone_fracture\\bone_fracture\\best_model.keras")
    return model

# Function to predict fracture
def predict_fracture(image, model):
    # Preprocessing
    image = image.resize((128, 128))
    image = np.array(image, dtype='float32')
    
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    
    if image.shape[-1] == 4:
        image = image[:,:,:3]
    
    # Normalize to [0,1]
    image = image / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    # Get predictions with error handling
    try:
        prediction = model.predict(image)
        st.write("Raw prediction shape:", prediction.shape)
        st.write("Raw prediction values:", prediction)
        
        # Since we have a sigmoid output (shape 1,1)
        # prediction close to 0 means fracture
        # prediction close to 1 means no fracture
        fracture_prob = 1 - prediction[0][0]  # Invert the probability
        no_fracture_prob = prediction[0][0]
            
        st.write(f"No Fracture probability: {no_fracture_prob:.4%}")
        st.write(f"Fracture probability: {fracture_prob:.4%}")
        
        # Threshold at 0.5 after inverting
        return "Fracture Detected" if fracture_prob > 0.5 else "No Fracture Detected"
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return "Error in prediction"

# Frontend starts here
# Configure page layout
st.set_page_config(
    page_title="X-Ray Fracture Detection",
    page_icon="🩻",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .insight-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Navigation
selected = option_menu(
    menu_title=None,
    options=["Home", "Detection Tool", "About"],
    icons=["house", "image", "info-circle"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

if selected == "Home":
    # Main Page Content
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>AI-Powered Bone Fracture Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em;'>Advanced X-ray Analysis Using Deep Learning</p>", unsafe_allow_html=True)
    
    # Project Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='insight-card'>
            <h3>🎯 Accuracy</h3>
            <p>Our model achieves good accuracy in detecting bone fractures from X-ray images (The Model can be more accurate if trained on better dataset.)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='insight-card'>
            <h3>⚡ Speed</h3>
            <p>Real-time analysis with results in under 5 seconds per image.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='insight-card'>
            <h3>🔍 Reliability</h3>
            <p>Trained on thousands of X-ray images for reliable results.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown("## 🌟 Key Features")
    st.markdown("""
    - **Instant Analysis**: Upload and get results in seconds
    - **High Accuracy**: State-of-the-art deep learning model
    - **User-Friendly**: Simple interface for medical professionals
    - **Multiple Format Support**: Handles various image formats
    """)
    

elif selected == "Detection Tool":
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>X-Ray Fracture Detection Tool</h2>", unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload your X-ray image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
    
    with col2:
        if uploaded_file is not None:
            if st.button('Analyze X-ray'):
                with st.spinner('Analyzing image...'):
                    model = load_model()
                    prediction = predict_fracture(image, model)
                    
                    # Display result with custom styling
                    result_color = "#FF5733" if "Fracture Detected" in prediction else "#4CAF50"
                    st.markdown(f"""
                        <div style='background-color: {result_color}; padding: 2rem; border-radius: 10px; text-align: center;'>
                            <h2 style='color: white;'>{prediction}</h2>
                        </div>
                    """, unsafe_allow_html=True)

elif selected == "About":
    st.markdown("## 📚 About the Project")
    st.write("""
    This AI-powered bone fracture detection system was developed to assist medical professionals
    in quickly and accurately identifying fractures in X-ray images. The system uses a deep
    learning model trained on thousands of X-ray images to provide reliable results.
    
    ### Technical Details
    - Built using TensorFlow and Deep Learning
    - Trained on a dataset of X-ray images found on kaggle
    - Implements fine tuned image processing techniques
    
    ### Team
    Developed by Om Kadam And Yashwant Ingle.
    """)

# Sidebar remains the same but with enhanced styling
with st.sidebar:
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Select 'Detection Tool' from the navigation menu
    2. Upload a valid X-ray image (PNG/JPEG format)
    3. Click 'Analyze X-ray' to get results
    4. Review the prediction
    
    ### 🎯 Best Practices
    - Use clear, high-quality X-ray images
    - Ensure proper image orientation
    
    ### ⚠️ Disclaimer
    This tool is meant to assist, not replace, professional medical diagnosis.
    """)
