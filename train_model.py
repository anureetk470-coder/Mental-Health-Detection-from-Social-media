from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

try:
    from .nlp_pipeline import TextPreprocessor, TextStats, preprocess_text
except ImportError:
    from nlp_pipeline import TextPreprocessor, TextStats, preprocess_text


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_mental_health_posts.csv"
MODEL_PATH = ROOT / "models" / "mental_health_model.joblib"
REPORT_PATH = ROOT / "reports" / "model_report.txt"
FIGURE_DIR = ROOT / "reports" / "figures"


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    expected = {"post", "label"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")

    df = df.dropna(subset=["post", "label"]).copy()
    df["label"] = df["label"].str.lower().str.strip()
    df["clean_post"] = df["post"].apply(preprocess_text)
    return df


def build_pipeline(model):
    features = ColumnTransformer(
        transformers=[
            (
                "tfidf",
                Pipeline(
                    steps=[
                        ("cleaner", TextPreprocessor()),
                        (
                            "vectorizer",
                            TfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=1,
                                max_features=2000,
                            ),
                        ),
                    ]
                ),
                "post",
            ),
            (
                "stats",
                Pipeline(steps=[("stats", TextStats()), ("scale", MinMaxScaler())]),
                "post",
            ),
        ]
    )

    return Pipeline(steps=[("features", features), ("model", model)])


def candidate_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
    }


def make_visuals(df, y_test, y_pred, labels):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="label", order=labels, palette="Set2")
    plt.title("Class Distribution")
    plt.xlabel("Mental Health State")
    plt.ylabel("Number of Posts")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=160)
    plt.close()

    matrix = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "confusion_matrix.png", dpi=160)
    plt.close()


def train_and_evaluate(data_path=DATA_PATH):
    df = load_data(data_path)
    labels = sorted(df["label"].unique())

    stratify = df["label"] if df["label"].value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        df[["post"]],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    results = {}
    fitted_models = {}
    for name, model in candidate_models().items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results[name] = accuracy_score(y_test, predictions)
        fitted_models[name] = (pipeline, predictions)

    best_name = max(results, key=results.get)
    best_model, best_predictions = fitted_models[best_name]

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    report = classification_report(y_test, best_predictions, zero_division=0)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        file.write(f"Best model: {best_name}\n")
        file.write(f"Accuracy: {results[best_name]:.3f}\n\n")
        file.write("All model accuracies:\n")
        for name, score in results.items():
            file.write(f"- {name}: {score:.3f}\n")
        file.write("\nClassification report:\n")
        file.write(report)

    make_visuals(df, y_test, best_predictions, labels)
    return best_name, results, report


if __name__ == "__main__":
    name, scores, report_text = train_and_evaluate()
    print(f"Best model: {name}")
    for model_name, score in scores.items():
        print(f"{model_name}: {score:.3f}")
    print(report_text)

