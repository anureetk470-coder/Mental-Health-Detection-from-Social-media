from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.nlp_pipeline import preprocess_text
from src.train_model import DATA_PATH, FIGURE_DIR, MODEL_PATH, load_data, train_and_evaluate


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

with left:
    st.subheader("Try a post")
    sample_text = st.text_area(
        "Social media post",
        value="I feel overwhelmed by deadlines and cannot relax.",
        height=150,
    )

    if st.button("Predict state", use_container_width=True):
        prediction = model.predict(pd.DataFrame({"post": [sample_text]}))[0]
        color = prediction_color(prediction)
        cleaned = preprocess_text(sample_text)

        st.markdown(
            f"""
            <div class="result-box">
                <div class="small-label">Predicted class</div>
                <h2 style="color:{color}; margin-top:6px;">{prediction.title()}</h2>
                <div class="small-label">Cleaned text</div>
                <p style="color:darkslategray;">{cleaned or "No useful tokens after preprocessing."}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Preprocessing pipeline")
    st.write(
        "Lowercasing, URL removal, emoji removal, punctuation removal, tokenization, stopword removal, and lemmatization."
    )

with right:
    st.subheader("Dataset snapshot")
    st.dataframe(df[["post", "label", "clean_post"]], use_container_width=True, height=280)

    st.subheader("Class counts")
    st.bar_chart(df["label"].value_counts())

tabs = st.tabs(["Visualizations", "Model notes", "Ethics"])

with tabs[0]:
    col1, col2 = st.columns(2)
    class_plot = FIGURE_DIR / "class_distribution.png"
    matrix_plot = FIGURE_DIR / "confusion_matrix.png"
    if not class_plot.exists() or not matrix_plot.exists():
        train_and_evaluate(DATA_PATH)
    with col1:
        st.image(str(class_plot), caption="Class distribution", use_container_width=True)
    with col2:
        st.image(str(matrix_plot), caption="Confusion matrix", use_container_width=True)

with tabs[1]:
    st.write(
        "The project compares Logistic Regression, Linear SVM, and Naive Bayes. "
        "Features combine TF-IDF n-grams with simple numeric text statistics such as word count, punctuation counts, and keyword signals."
    )
    report_path = Path("reports/model_report.txt")
    if report_path.exists():
        st.code(report_path.read_text(encoding="utf-8"), language="text")

with tabs[2]:
    st.info(
        "This app is for learning only. It should not be used to diagnose people, monitor individuals, or replace professional mental health support."
    )
