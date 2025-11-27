"""
Urban Growth Prediction - Streamlit Web Application
"""
import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import _0_config as cfg
from PIL import Image
import io


st.set_page_config(
    page_title="Urban Growth Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    """Load the trained model (cached)"""
    
    possible_paths = [
        os.path.join(cfg.OUT_DIR, 'best_model.keras'),
        os.path.join(cfg.OUT_DIR, 'best_model.h5'),
        os.path.join(cfg.OUT_DIR, 'final_model.keras'),
        os.path.join(cfg.OUT_DIR, 'final_model.h5'),
    ]
    
    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                return load_model(model_path, compile=False)
            except:
                continue
    
    return None


@st.cache_data
def load_test_data():
    """Load test data (cached)"""
    test_X = np.load(os.path.join(cfg.OUT_NP_DIR, 'test_X.npy'))
    test_Y = np.load(os.path.join(cfg.OUT_NP_DIR, 'test_Y.npy'))
    test_X = test_X[..., None]
    test_Y = np.squeeze(test_Y)
    return test_X, test_Y


@st.cache_data
def load_years_data():
    """Load years information"""
    years = np.load(os.path.join(cfg.OUT_NP_DIR, 'all_years.npy'))
    return years


@st.cache_data
def load_metrics():
    """Load evaluation metrics if available"""
    metrics_file = os.path.join(cfg.OUT_DIR, 'evaluation_metrics.txt')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return f.read()
    return None


def predict_future_year(model, target_year, years):
    """Predict future year"""
    last_available_year = years[-1]
    
    if target_year <= last_available_year:
        return None, None, "Target year must be in the future"
    
    
    test_X = np.load(os.path.join(cfg.OUT_NP_DIR, 'test_X.npy'))
    last_seq = test_X[-1]
    
    
    if last_seq.ndim == 3:  # (seq, H, W)
        last_seq = last_seq[..., None]  # Add channel: (seq, H, W, 1)
    
    current_seq = last_seq.copy()
    current_year = last_available_year
    
    future_predictions = []
    future_years = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = target_year - current_year
    step = 0
    
    while current_year < target_year:
        status_text.text(f"Predicting year {current_year + 1}...")
        
        pred = model.predict(current_seq[np.newaxis, ...], verbose=0)
        pred = (pred[0, ..., 0] > 0.5).astype(int)
        
        future_predictions.append(pred)
        future_years.append(current_year + 1)
        
        current_year += 1
        step += 1
        progress_bar.progress(step / steps)
        
       
        pred_with_channel = pred[..., None]
        pred_for_concat = pred_with_channel[np.newaxis, ...]
        current_seq = np.concatenate([current_seq[1:], pred_for_concat], axis=0)
    
    progress_bar.empty()
    status_text.empty()
    
    return np.array(future_years), np.array(future_predictions), None


def plot_prediction(image, title):
    """Create a matplotlib figure for prediction"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    #conv to img
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)


def main():
    #header
    st.markdown('<h1 class="main-header">🏙️ Urban Growth Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Model Accuracy", "Test Predictions", "Future Predictions"])
    
    
    model = load_trained_model()
    
    if model is None:
        st.error("❌ Model not found! Please train the model first by running 3_train.py")
        return
    
    
    if page == "Home":
        st.header("📊 Project Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("About")
            st.write("""
            This machine learning system analyzes historical satellite imagery to predict urban growth patterns.
            
            **Key Features:**
            - Uses ConvLSTM neural network architecture
            - Trained on yearly satellite data
            - Predicts future urban expansion
            - Provides accuracy metrics and visualizations
            """)
            
            st.subheader("Configuration")
            st.write(f"**Region:** Delhi (Approx)")
            st.write(f"**Image Size:** {cfg.IMG_SHAPE[0]} x {cfg.IMG_SHAPE[1]}")
            st.write(f"**Input Sequence:** {cfg.IN_SEQ} years")
            st.write(f"**AOI Bounding Box:** {cfg.AOI_BBOX}")
        
        with col2:
            st.subheader("Model Architecture")
            st.write("""
            **ConvLSTM2D Neural Network**
            - Layer 1: 32 filters, ConvLSTM2D
            - Batch Normalization
            - Layer 2: 16 filters, ConvLSTM2D
            - Batch Normalization
            - Output: Single-channel probability map
            - Activation: Sigmoid
            - Loss: Binary Crossentropy
            """)
            
            st.subheader("Quick Stats")
            try:
                years = load_years_data()
                test_X, test_Y = load_test_data()
                st.metric("Available Years", f"{years[0]} - {years[-1]}")
                st.metric("Test Samples", test_X.shape[0])
            except:
                st.warning("Data not loaded")
    
   
    elif page == "Model Accuracy":
        st.header("📈 Model Performance Metrics")
        
        metrics_text = load_metrics()
        
        if metrics_text:
            st.success("✅ Evaluation completed successfully!")
            
            
            lines = metrics_text.split('\n')
            metrics_dict = {}
            for line in lines:
                if ':' in line and any(m in line for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'IoU']):
                    key = line.split(':')[0].strip()
                    value = line.split(':')[1].strip().split()[0]
                    metrics_dict[key] = float(value)
            
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Accuracy", f"{metrics_dict.get('Accuracy', 0):.2%}")
            with col2:
                st.metric("Precision", f"{metrics_dict.get('Precision', 0):.3f}")
            with col3:
                st.metric("Recall", f"{metrics_dict.get('Recall', 0):.3f}")
            with col4:
                st.metric("F1-Score", f"{metrics_dict.get('F1-Score', 0):.3f}")
            with col5:
                st.metric("IoU", f"{metrics_dict.get('IoU', 0):.3f}")
            
            st.markdown("---")
            
            #conf matr
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                cm_path = os.path.join(cfg.OUT_DIR, 'confusion_matrix.png')
                if os.path.exists(cm_path):
                    st.image(cm_path, use_column_width=True)
                else:
                    st.warning("Confusion matrix not found")
            
            with col2:
                st.subheader("Sample Predictions")
                samples_path = os.path.join(cfg.OUT_DIR, 'sample_predictions.png')
                if os.path.exists(samples_path):
                    st.image(samples_path, use_column_width=True)
                else:
                    st.warning("Sample predictions not found")
        else:
            st.warning("⚠️ No evaluation metrics found. Run 5_evaluate_model.py first!")
            
            if st.button("Run Evaluation Now"):
                with st.spinner("Evaluating model..."):
                    from _5_evaluate_model import evaluate_model
                    evaluate_model(visualize=True)
                    st.success("Evaluation complete! Refresh the page.")
                    st.experimental_rerun()
    
    
    elif page == "Test Predictions":
        st.header("🔍 Test Set Predictions")
        
        try:
            test_X, test_Y = load_test_data()
            
            st.write(f"Total test samples: {test_X.shape[0]}")
            
            sample_idx = st.slider("Select test sample", 0, test_X.shape[0]-1, 0)
            
            if st.button("Generate Prediction"):
                with st.spinner("Generating prediction..."):
                    pred = model.predict(test_X[sample_idx:sample_idx+1], verbose=0)
                    pred = (pred[0, ..., 0] > 0.5).astype(int)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Last Input Year")
                        img = plot_prediction(test_X[sample_idx, -1, ..., 0], "Last Input")
                        st.image(img)
                    
                    with col2:
                        st.subheader("Ground Truth")
                        img = plot_prediction(test_Y[sample_idx], "True Next Year")
                        st.image(img)
                    
                    with col3:
                        st.subheader("Prediction")
                        img = plot_prediction(pred, "Predicted Next Year")
                        st.image(img)
                    
                    # Calculate accuracy
                    from sklearn.metrics import accuracy_score, jaccard_score
                    acc = accuracy_score(test_Y[sample_idx].flatten(), pred.flatten())
                    iou = jaccard_score(test_Y[sample_idx].flatten(), pred.flatten(), zero_division=0)
                    
                    st.success(f"✅ Sample Accuracy: {acc:.2%} | IoU: {iou:.3f}")
        
        except Exception as e:
            st.error(f"Error loading test data: {e}")
    
    
    elif page == "Future Predictions":
        st.header("🔮 Future Urban Growth Prediction")
        
        try:
            years = load_years_data()
            last_year = years[-1]
            
            st.info(f"📅 Latest available data: {last_year}")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                target_year = st.number_input(
                    "Enter target year to predict",
                    min_value=int(last_year)+1,
                    max_value=2100,
                    value=int(last_year)+10,
                    step=1
                )
            
            with col2:
                st.write("")
                st.write("")
                predict_btn = st.button("🚀 Predict Future", type="primary")
            
            if predict_btn:
                with st.spinner(f"Predicting year {target_year}..."):
                    future_years, predictions, error = predict_future_year(model, target_year, years)
                    
                    if error:
                        st.error(error)
                    else:
                        st.success(f"✅ Successfully predicted up to year {target_year}!")
                        
                        # Display final prediction
                        st.subheader(f"Predicted Urban Map - {target_year}")
                        
                        final_pred = predictions[-1]
                        img = plot_prediction(final_pred, f"Urban Growth Prediction - {target_year}")
                        st.image(img, use_column_width=True)
                        
                        # Statistics
                        urban_pixels = np.sum(final_pred)
                        total_pixels = final_pred.size
                        urban_percentage = (urban_pixels / total_pixels) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Urban Pixels", f"{urban_pixels:,}")
                        with col2:
                            st.metric("Total Pixels", f"{total_pixels:,}")
                        with col3:
                            st.metric("Urban Coverage", f"{urban_percentage:.2f}%")
                        
                        
                        if len(predictions) > 1:
                            show_all = st.checkbox("Show all intermediate predictions")
                            
                            if show_all:
                                st.subheader("Year-by-Year Predictions")
                                cols = st.columns(min(4, len(predictions)))
                                
                                for i, (year, pred) in enumerate(zip(future_years, predictions)):
                                    with cols[i % 4]:
                                        img = plot_prediction(pred, f"{year}")
                                        st.image(img)
        
        except Exception as e:
            st.error(f"Error: {e}")
    
   
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>Urban Growth Prediction System | "
        "Powered by ConvLSTM & TensorFlow</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()