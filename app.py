from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from src.nlp_pipeline import preprocess_text
    from src.train_model import DATA_PATH, FIGURE_DIR, MODEL_PATH, load_data, train_and_evaluate
except ModuleNotFoundError:
    from nlp_pipeline import preprocess_text
    from train_model import DATA_PATH, FIGURE_DIR, MODEL_PATH, load_data, train_and_evaluate


st.set_page_config(
    page_title="Mental Health Post Detector",
    page_icon="",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, lavender 0%, mintcream 45%, aliceblue 100%);
        color: darkslategray;
    }
    .main-title {
        color: darkslateblue;
        font-size: 42px;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 8px;
    }
    .sub-title {
        color: dimgray;
        font-size: 17px;
        margin-bottom: 24px;
    }
    .result-box {
        background: white;
        border: 1px solid lightsteelblue;
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 8px 24px rgba(72, 61, 139, 0.10);
    }
    .small-label {
        color: slategray;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        train_and_evaluate(DATA_PATH)
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_data():
    return load_data(DATA_PATH)


def prediction_color(label):
    return {
        "stress": "tomato",
        "anxiety": "darkorange",
        "depression": "mediumslateblue",
        "normal": "seagreen",
    }.get(label, "darkslategray")


model = get_model()
df = get_data()

st.markdown('<div class="main-title">Mental Health Detection from Social Media</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">A simple NLP and machine learning app for classifying posts as stress, anxiety, depression, or normal.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 0.95], gap="large")
