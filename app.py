"""
==========================================================
Plant Disease Detection Using Computer Vision & Deep Learning

File: streamlit_app.py

Description:
Professional Streamlit Dashboard for Plant Disease Detection.

==========================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import numpy as np
import streamlit as st

from PIL import Image

from src.prediction import (predict_disease, load_class_names)

from src.disease_info import DISEASE_INFO


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """
<style>

/* ---------------- Main Background ---------------- */

.main{
    background-color:#F7FAFC;
}


/* ---------------- Header ---------------- */

.hero{

    background:linear-gradient(
        90deg,
        #0F766E,
        #16A34A
    );

    padding:30px;

    border-radius:18px;

    text-align:center;

    color:white;

    margin-bottom:25px;
}

.hero h1{

    font-size:42px;

    font-weight:800;

    margin-bottom:10px;
}

.hero p{

    font-size:18px;

}


/* ---------------- KPI Cards ---------------- */

.metric-card{

    background:white;

    border-radius:15px;

    padding:18px;

    text-align:center;

    box-shadow:0px 4px 12px rgba(0,0,0,0.10);

    margin-bottom:20px;

}


/* ---------------- Upload Box ---------------- */

.upload-box{

    background:white;

    border-radius:15px;

    padding:20px;

    box-shadow:0px 4px 10px rgba(0,0,0,0.10);

}


/* ---------------- Footer ---------------- */

.footer{

    text-align:center;

    color:gray;

    margin-top:50px;

    font-size:15px;

}

</style>
""",
    unsafe_allow_html=True,
)


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/240/plant-under-sun.png",
        width=120,
    )

    st.title("🌿 Plant Disease Detection")

    st.markdown("---")

    st.subheader("Project Overview")

    st.write(
        """
This application uses **Computer Vision**
and **Deep Learning (MobileNetV2)** to
identify diseases in plant leaves.

Upload a leaf image and the model will
predict the disease along with its
confidence score.
"""
    )

    st.markdown("---")

    st.subheader("Model")

    st.success("MobileNetV2 Transfer Learning")

    st.subheader("Framework")

    st.info("TensorFlow + Streamlit")

    st.markdown("---")

    st.caption("Version 1.0")


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
<div class="hero">

<h1>
🌿 Plant Disease Detection System
</h1>

<p>
Computer Vision & Deep Learning Powered Crop Disease Identification
</p>

</div>
""",
    unsafe_allow_html=True,
)


# ==========================================================
# PROJECT INFORMATION
# ==========================================================

st.markdown("## 📖 Project Information")

st.write(
"""
This application detects diseases from plant
leaf images using a **MobileNetV2 Transfer
Learning Model** trained on the PlantVillage
dataset.

Supported Crops:

- 🍅 Tomato
- 🥔 Potato
- 🫑 Pepper

The system predicts the disease category
and provides the confidence score.
"""
)

st.markdown("---")


# ==========================================================
# MAIN LAYOUT
# ==========================================================

left_column, right_column = st.columns(
    [1,1],
)

with left_column:

    st.subheader("📤 Upload Leaf Image")

    uploaded_file = st.file_uploader(

        "Choose a Plant Leaf Image",

        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

with right_column:

    st.subheader("🖼 Image Preview")

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.image(

            image,

            use_container_width=True,

        )

    else:

        st.info(
            "Upload an image to preview."
        )

st.markdown("---")

# ==========================================================
# PREDICTION SECTION
# ==========================================================

st.subheader("🔍 Disease Prediction")

if uploaded_file is not None:

    if st.button(
        "🚀 Detect Disease",
        use_container_width=True,
    ):

        with st.spinner("Analyzing leaf image..."):

            predicted_class, confidence, predictions = predict_disease(
                uploaded_file
            )

        st.success("Prediction Completed Successfully!")

        st.markdown("---")

        # ======================================================
        # RESULT CARDS
        # ======================================================

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                label="🌿 Predicted Disease",
                value=predicted_class.replace("_", " "),
            )

        with result_col2:

            st.metric(
                label="🎯 Confidence",
                value=f"{confidence:.2f} %",
            )

        st.markdown("### Confidence Score")

        st.progress(
            min(confidence / 100, 1.0)
        )

        st.write(
            f"Model Confidence : **{confidence:.2f}%**"
        )

        st.markdown("---")

        # ======================================================
        # TOP 3 PREDICTIONS
        # ======================================================

        st.subheader("📊 Top 3 Predictions")

        class_names = load_class_names()

        prediction_scores = predictions[0]

        top_indices = np.argsort(
            prediction_scores
        )[::-1][:3]

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):

            disease_name = class_names[index]

            probability = prediction_scores[index] * 100

            st.write(
                f"**{rank}. {disease_name.replace('_',' ')}**"
            )

            st.progress(
                min(probability / 100, 1.0)
            )

            st.write(
                f"{probability:.2f}%"
            )

            st.write("")

        # ==========================================================
        # DISEASE INFORMATION
        # ==========================================================

        st.markdown("---")

        st.subheader("🌿 Disease Information")

        info = DISEASE_INFO.get(predicted_class)

        if info:

            st.write(f"### 📖 Description")
            st.write(info["description"])

            col1, col2 = st.columns(2)

            with col1:

                st.write("### 🦠 Symptoms")

                for symptom in info["symptoms"]:
                    st.write(f"• {symptom}")

            with col2:

                st.write("### 💊 Treatment")

                for treatment in info["treatment"]:
                    st.write(f"• {treatment}")

            st.write("### 🛡 Prevention")

            for prevention in info["prevention"]:
                st.write(f"• {prevention}")

        # ==========================================================
        # PREDICTION PROBABILITIES
        # ==========================================================

        st.markdown("---")

        st.subheader("📊 Prediction Probabilities")

        df = pd.DataFrame({

            "Disease": class_names,

            "Probability (%)": predictions[0] * 100

        })

        df = df.sort_values(

            by="Probability (%)",

            ascending=False,

        )

        st.dataframe(

            df,

            use_container_width=True,

        )

else:

    st.info(
        "Please upload a plant leaf image to begin prediction."
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown(
    """
<div class="footer">

Developed with ❤️ using TensorFlow, Streamlit and MobileNetV2

<br><br>

<strong>Harish Ragavendra</strong>

</div>
""",
    unsafe_allow_html=True,
)