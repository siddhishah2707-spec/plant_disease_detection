import os
import warnings
import contextlib
import io
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
import pandas as pd
from datetime import datetime
from tensorflow.keras.preprocessing import image
from utils.gradcam import get_gradcam_heatmap, overlay_heatmap

# ------------------ SUPPRESS WARNINGS ------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ------------------ LOAD MODEL & CLASSES ------------------
@st.cache_resource
def load_model_and_labels():
    with contextlib.redirect_stdout(io.StringIO()):
        try:
    model = tf.keras.models.load_model("models/best_model.keras", compile=False)
except:
    model = tf.keras.models.load_model("models/best_model.h5", compile=False)
    with open("models/class_labels.pkl", "rb") as f:
        class_labels = pickle.load(f)
    return model, class_labels

model, class_labels = load_model_and_labels()
last_conv_layer_name = "Conv_1"

# ------------------ APP CONFIG ------------------
st.set_page_config(
    page_title="🌿 Plant Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ CUSTOM STYLING ------------------
st.markdown("""
    <style>
        .block-container {
            max-width: 1300px !important;
            padding: 2rem 4rem;
        }
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }
        .main-title {
            background-color: #e8f5e9;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            font-size: 30px;
            font-weight: 700;
            color: #2e7d32;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR NAVIGATION ------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to Page:",
    ["🔍 Prediction", "🌈 Grad-CAM & Remedy", "📊 Disease Class Chart", "📁 Prediction History"]
)

# ------------------ PAGE 1: PREDICTION ------------------
if page == "🔍 Prediction":
    st.markdown("<div class='main-title'>🌿 Plant Disease Detection</div>", unsafe_allow_html=True)
    st.markdown("Upload a plant leaf image to identify disease type and confidence level.")

    uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save uploaded image
        img_path = "temp_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Preprocess
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        preds = model.predict(img_array)
        class_idx = np.argmax(preds[0])
        confidence = preds[0][class_idx] * 100
        predicted_label = class_labels[class_idx]

        # Store prediction in session state
        st.session_state["predicted_label"] = predicted_label
        st.session_state["confidence"] = confidence
        st.session_state["img_path"] = img_path
        st.session_state["img_array"] = img_array

        # Display result
        st.success(f"✅ **Prediction:** {predicted_label}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        # Save to CSV history
        os.makedirs("logs", exist_ok=True)
        history_file = "logs/prediction_history.csv"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        new_entry = pd.DataFrame({
            "DateTime": [timestamp],
            "Predicted Disease": [predicted_label],
            "Confidence (%)": [f"{confidence:.2f}"],
            "Image Path": [img_path]
        })

        if os.path.exists(history_file):
            existing = pd.read_csv(history_file)
            updated = pd.concat([existing, new_entry], ignore_index=True)
        else:
            updated = new_entry
        updated.to_csv(history_file, index=False)

        st.markdown("---")
        st.write("➡️ Go to **'🌈 Grad-CAM & Remedy'** from the sidebar for visualization and treatment advice.")

# ------------------ PAGE 2: GRAD-CAM + REMEDY ------------------
elif page == "🌈 Grad-CAM & Remedy":
    st.markdown("<div class='main-title'>🌈 Grad-CAM Visualization & Remedies</div>", unsafe_allow_html=True)

    if "predicted_label" not in st.session_state:
        st.warning("⚠️ Please make a prediction first on the '🔍 Prediction' page.")
    else:
        predicted_label = st.session_state["predicted_label"]
        confidence = st.session_state["confidence"]
        img_path = st.session_state["img_path"]
        img_array = st.session_state["img_array"]

        remedies = {
            "Apple___Apple_scab": "Use fungicides like Captan or Mancozeb; prune infected leaves and ensure proper air circulation.",
            "Apple___Black_rot": "Remove mummified fruits, prune cankers, and apply fungicide sprays containing copper.",
            "Apple___Cedar_apple_rust": "Remove nearby juniper hosts; apply myclobutanil-based fungicides early in the season.",
            "Apple___healthy": "No disease detected — continue regular monitoring and maintain balanced fertilization.",
            "Blueberry___healthy": "No disease detected — maintain good irrigation and avoid overhead watering.",
            "Corn_(maize)___Common_rust_": "Use resistant varieties; avoid dense planting and ensure nitrogen balance.",
            "Corn_(maize)___healthy": "Healthy crop — maintain regular pest and nutrient management.",
            "Grape___Black_rot": "Remove infected parts; apply copper-based fungicides and improve ventilation.",
            "Peach___Bacterial_spot": "Use copper sprays; avoid overhead watering and remove infected material.",
            "Pepper,_bell___Bacterial_spot": "Apply copper-based bactericides; ensure field sanitation and avoid working on wet plants.",
        }

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🖼️ Uploaded Image")
            st.image(img_path, use_column_width=True)

        with col2:
            st.subheader("💡 Prediction Summary")
            st.success(f"**Disease:** {predicted_label}")
            st.info(f"**Confidence:** {confidence:.2f}%")

            st.markdown("---")
            st.subheader("🌿 Suggested Remedy")
            if predicted_label in remedies:
                st.markdown(
                    f"<div style='background-color:#eafbea;padding:15px;border-radius:10px;'>💊 {remedies[predicted_label]}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.warning("No specific remedy found for this disease.")

        st.markdown("---")
        st.subheader("🌈 Grad-CAM Heatmap Visualization")

        heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)
        overlayed = overlay_heatmap(img_path, heatmap)
        st.image(overlayed, caption="Regions influencing the model prediction", use_column_width=True)

        st.markdown("---")
        st.info("➡️ Move to **'📊 Disease Class Chart'** page for an overview of all supported classes.")

# ------------------ PAGE 3: CLASS CHART ------------------
elif page == "📊 Disease Class Chart":
    st.markdown("<div class='main-title'>📊 All Recognized Disease Classes</div>", unsafe_allow_html=True)
    st.write("Below is the list of all plant diseases the model can identify:")

    classes_df = pd.DataFrame({
        "Class Index": list(range(len(class_labels))),
        "Disease Name": class_labels
    })

    st.dataframe(classes_df, use_container_width=True)
    st.markdown("---")
    st.info("Use the sidebar to return to **Prediction**, **Grad-CAM**, or **History** pages.")
# ------------------ PAGE 4: PREDICTION HISTORY ------------------
elif page == "📁 Prediction History":
    st.markdown("<div class='main-title'>📁 Prediction History</div>", unsafe_allow_html=True)

    history_file = "logs/prediction_history.csv"
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)

        # --- Filter option ---
        filter_option = st.radio(
            "🩺 Filter by Type:",
            ["All", "Healthy", "Diseased"],
            horizontal=True
        )

        if filter_option == "Healthy":
            filtered_df = history_df[history_df["Predicted Disease"].str.contains("healthy", case=False)]
        elif filter_option == "Diseased":
            filtered_df = history_df[~history_df["Predicted Disease"].str.contains("healthy", case=False)]
        else:
            filtered_df = history_df

        st.dataframe(filtered_df, use_container_width=True)
        st.success(f"✅ {len(filtered_df)} records found ({filter_option} leaves).")

        # --- Summary counts ---
        healthy_count = len(history_df[history_df["Predicted Disease"].str.contains("healthy", case=False)])
        diseased_count = len(history_df) - healthy_count
        total = len(history_df)

        st.markdown("---")
        st.subheader("📊 Health Summary")

        # --- Create pie chart ---
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        labels = ["Healthy", "Diseased"]
        values = [healthy_count, diseased_count]
        colors = ["#66bb6a", "#ef5350"]

        ax.pie(
            values,
            labels=labels,
            autopct=lambda p: f"{p:.1f}%\n({int(p * total / 100)})",
            startangle=90,
            colors=colors,
            textprops={"color": "black", "fontsize": 11},
        )
        ax.axis("equal")  # Equal aspect ratio ensures the pie is a circle
        st.pyplot(fig)

        st.markdown(f"**Total Predictions:** {total} | 🟢 Healthy: {healthy_count} | 🔴 Diseased: {diseased_count}")
    else:
        st.warning("No prediction history found yet. Make some predictions first.")
