
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

HEART_MODEL_PATH = BASE_DIR / "models" / "heart_model.joblib"
DIABETES_MODEL_PATH = BASE_DIR / "models" / "diabetes_model.joblib"

st.set_page_config(
    page_title="MediCare AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# PREMIUM DESIGN
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background: #F3F7FB;
        font-family: 'Inter', sans-serif;
        color: #10243A;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10243A, #183D56);
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    .brand {
        font-size: 27px;
        font-weight: 800;
        color: white;
        margin-bottom: 0;
    }

    .brand span {
        color: #5EEAD4;
    }

    .sidebar-caption {
        color: #B8C7D8;
        font-size: 11px;
        letter-spacing: 2px;
        margin-bottom: 25px;
    }

    .hero {
        background: linear-gradient(120deg, #10243A, #18515B);
        padding: 32px;
        border-radius: 20px;
        color: white;
        margin-bottom: 25px;
    }

    .hero .eyebrow {
        color: #5EEAD4;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .hero h1 {
        color: white;
        font-size: 35px;
        margin: 12px 0;
    }

    .hero p {
        color: #D6E5EF;
        font-size: 15px;
        line-height: 1.7;
    }

    .section-title {
        font-size: 23px;
        font-weight: 800;
        color: #10243A;
        margin: 20px 0 5px;
    }

    .section-subtitle {
        color: #64748B;
        margin-bottom: 18px;
    }

    .metric-card, .info-card {
        background: white;
        border: 1px solid #E0E8F0;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 12px;
        box-shadow: 0 5px 18px rgba(16, 36, 58, 0.04);
    }

    .metric-label {
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
    }

    .metric-value {
        color: #10243A;
        font-size: 25px;
        font-weight: 800;
        margin: 10px 0;
    }

    .metric-note {
        color: #0D9488;
        font-size: 12px;
    }

    .info-card h3 {
        color: #10243A;
        font-size: 18px;
    }

    .info-card p {
        color: #64748B;
        line-height: 1.7;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        background: #0D9488;
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background: #0F766E;
        color: white;
        border: none;
    }

    [data-testid="stForm"] {
        background: white;
        border: 1px solid #E0E8F0;
        border-radius: 16px;
        padding: 22px;
    }

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 12px;
        padding: 30px 0 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------
@st.cache_resource
def load_model(model_path_string):
    """
    The saved .joblib files contain dictionaries.
    Extract the sklearn pipeline and its training features.
    """
    model_path = Path(model_path_string)

    if not model_path.exists():
        return None, None, f"Model file not found: {model_path}"

    try:
        bundle = joblib.load(model_path)

        if isinstance(bundle, dict):
            pipeline = bundle.get("pipeline")
            features = bundle.get("features")
        else:
            pipeline = bundle
            features = None

        if pipeline is None or not hasattr(pipeline, "predict"):
            return None, None, (
                f"{model_path.name} does not contain a valid prediction pipeline."
            )

        if not features:
            return None, None, (
                f"Training feature names are missing in {model_path.name}."
            )

        return pipeline, list(features), None

    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


heart_model, heart_features, heart_error = load_model(
    str(HEART_MODEL_PATH)
)

diabetes_model, diabetes_features, diabetes_error = load_model(
    str(DIABETES_MODEL_PATH)
)


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------
def predict_safely(model, values, expected_features, model_name):
    if model is None:
        st.error(f"{model_name} model is not loaded.")
        return

    if not expected_features:
        st.error("Model feature information is missing.")
        return

    if len(values) != len(expected_features):
        st.error(
            f"{model_name} expects {len(expected_features)} inputs, "
            f"but the form provided {len(values)}."
        )
        st.write("Expected features:", expected_features)
        return

    # Preserve the exact feature names and order used during training.
    input_data = pd.DataFrame(
        [values],
        columns=expected_features,
    )

    try:
        # IMPORTANT: model is the extracted pipeline, not the dictionary.
        prediction = model.predict(input_data)[0]

        positive = str(prediction).strip().lower() in {
            "1", "1.0", "true", "yes", "positive", "disease"
        }

        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)

            for index, label in enumerate(classes):
                if str(label).strip().lower() in {
                    "1", "1.0", "true", "yes"
                }:
                    probability = float(probabilities[index]) * 100
                    break

        st.markdown("### Prediction Result")

        if positive:
            st.warning(
                f"**{model_name}: The model flagged a potential risk.** "
                "This is not a diagnosis. Consult a qualified healthcare "
                "professional for proper assessment."
            )
        else:
            st.success(
                f"**{model_name}: The model did not flag a positive result "
                "for these inputs.** This does not rule out disease."
            )

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric("Model output", str(prediction))

        with result_col2:
            if probability is not None:
                st.metric(
                    "Estimated positive-class probability",
                    f"{probability:.1f}%"
                )
            else:
                st.caption("Probability output is not available.")

        with st.expander("View submitted input values"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True,
            )

    except Exception as exc:
        st.error(
            f"Prediction failed: {type(exc).__name__}: {exc}"
        )
        st.info(
            "Check the model's expected input format and the installed "
            "scikit-learn version."
        )


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------
def page_header(eyebrow, title, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand">MediCare <span>AI</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-caption">SMART HEALTH INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Heart Health",
            "Diabetes Check",
            "About MediCare AI",
        ],
        key="navigation_page",
    )

    st.markdown("---")
    st.markdown("### Model Status")

    if heart_model is not None:
        st.success("Heart model loaded")
    else:
        st.error("Heart model unavailable")
        if heart_error:
            st.caption(heart_error)

    if diabetes_model is not None:
        st.success("Diabetes model loaded")
    else:
        st.error("Diabetes model unavailable")
        if diabetes_error:
            st.caption(diabetes_error)

    st.markdown("---")
    st.caption("Educational screening tool")


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
if page == "Dashboard":
    page_header(
        "AI-POWERED HEALTH SCREENING",
        "Your Health, Informed by AI.",
        "Explore machine-learning screening demonstrations for heart "
        "disease and diabetes using your locally trained models."
    )

    st.markdown(
        '<div class="section-title">Health Overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">Your screening workspace</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("SCREENING MODULES", "02", "Heart health + diabetes")

    with c2:
        metric_card(
            "HEART MODEL",
            "Ready" if heart_model is not None else "Unavailable",
            "Trained local model",
        )

    with c3:
        metric_card(
            "DIABETES MODEL",
            "Ready" if diabetes_model is not None else "Unavailable",
            "Trained local model",
        )

    st.markdown(
        '<div class="section-title">Choose a Screening Tool</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h3>♥ Heart Health Screening</h3>
                <p>Enter heart-related measurements and health indicators
                to run the trained heart classification model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button(
            "Open Heart Health →",
            key="open_heart",
            use_container_width=True,
            on_click=lambda: st.session_state.update(
                navigation_page="Heart Health"
            ),
        )

    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h3>✚ Diabetes Screening</h3>
                <p>Enter the required measurements to run the trained
                diabetes classification model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.button(
            "Open Diabetes Check →",
            key="open_diabetes",
            use_container_width=True,
            on_click=lambda: st.session_state.update(
                navigation_page="Diabetes Check"
            ),
        )

    st.warning(
        "This application is for educational purposes only. "
        "Predictions may be inaccurate and must not be used for medical decisions."
    )


# --------------------------------------------------
# HEART HEALTH PAGE
# --------------------------------------------------
elif page == "Heart Health":
    page_header(
        "HEART HEALTH",
        "Heart Disease Screening",
        "Enter the values below using the coding and units expected "
        "by the dataset used to train the model."
    )

    if heart_model is None:
        st.error(heart_error or "Heart model could not be loaded.")
    else:
        with st.form("heart_prediction_form"):
            st.subheader("Patient Measurements")

            a, b, c = st.columns(3)

            with a:
                age = st.number_input("Age (years)", 1, 120, 45)
                sex = st.selectbox("Sex (dataset code)", [0, 1], index=1)
                cp = st.selectbox("Chest pain type (0–3)", [0, 1, 2, 3])
                trestbps = st.number_input(
                    "Resting blood pressure", 50, 250, 120
                )
                chol = st.number_input(
                    "Serum cholesterol", 50, 700, 200
                )

            with b:
                fbs = st.selectbox(
                    "Fasting blood sugar > 120 mg/dl", [0, 1]
                )
                restecg = st.selectbox(
                    "Resting ECG (dataset code)", [0, 1, 2]
                )
                thalach = st.number_input(
                    "Maximum heart rate", 50, 250, 150
                )
                exang = st.selectbox(
                    "Exercise-induced angina", [0, 1]
                )

            with c:
                oldpeak = st.number_input(
                    "ST depression (oldpeak)",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.1,
                )
                slope = st.selectbox(
                    "ST slope (dataset code)", [0, 1, 2]
                )
                ca = st.selectbox(
                    "Major vessels (ca)", [0, 1, 2, 3, 4]
                )
                thal = st.selectbox(
                    "Thal (dataset code)", [0, 1, 2, 3]
                )

            submitted = st.form_submit_button(
                "Run Heart Screening",
                use_container_width=True,
            )

        if submitted:
            heart_values = [
                age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal
            ]

            predict_safely(
                heart_model,
                heart_values,
                heart_features,
                "Heart Disease",
            )


# --------------------------------------------------
# DIABETES PAGE
# --------------------------------------------------
elif page == "Diabetes Check":
    page_header(
        "DIABETES SCREENING",
        "Diabetes Risk Assessment",
        "Enter the measurements using the same units as the dataset "
        "used to train your diabetes model."
    )

    if diabetes_model is None:
        st.error(
            diabetes_error or "Diabetes model could not be loaded."
        )
    else:
        with st.form("diabetes_prediction_form"):
            st.subheader("Patient Measurements")

            a, b = st.columns(2)

            with a:
                pregnancies = st.number_input(
                    "Pregnancies", 0, 30, 1
                )
                glucose = st.number_input(
                    "Glucose", 0, 400, 120
                )
                blood_pressure = st.number_input(
                    "Blood pressure", 0, 250, 70
                )
                skin_thickness = st.number_input(
                    "Skin thickness", 0, 150, 20
                )

            with b:
                insulin = st.number_input(
                    "Insulin", 0, 1000, 80
                )
                bmi = st.number_input(
                    "BMI",
                    min_value=0.0,
                    max_value=100.0,
                    value=25.0,
                    step=0.1,
                )
                pedigree = st.number_input(
                    "Diabetes pedigree function",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.5,
                    step=0.01,
                )
                diabetes_age = st.number_input(
                    "Age (years)", 1, 120, 30
                )

            submitted = st.form_submit_button(
                "Run Diabetes Screening",
                use_container_width=True,
            )

        if submitted:
            diabetes_values = [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                pedigree,
                diabetes_age,
            ]

            predict_safely(
                diabetes_model,
                diabetes_values,
                diabetes_features,
                "Diabetes",
            )


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------
elif page == "About MediCare AI":
    page_header(
        "ABOUT THE PROJECT",
        "Technology with a Health Focus.",
        "MediCare AI is a machine-learning demonstration project "
        "built with Python, Streamlit and scikit-learn."
    )

    st.markdown(
        """
        <div class="info-card">
            <h3>🧠 Machine Learning</h3>
            <p>The application uses trained pipelines for heart disease
            and diabetes classification.</p>
        </div>

        <div class="info-card">
            <h3>🔒 Local Model Execution</h3>
            <p>The application loads model files from the project's
            models folder on the computer running Streamlit.</p>
        </div>

        <div class="info-card">
            <h3>⚕️ Medical Disclaimer</h3>
            <p>This project is not a medical device and cannot diagnose
            or rule out a medical condition. Consult a qualified
            healthcare professional for medical advice.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        MediCare AI · Built with Python and Streamlit ·
        For educational purposes only
    </div>
    """,
    unsafe_allow_html=True,
)
