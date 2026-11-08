import json
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from src.evaluation import (
    build_classification_report,
    confusion_table,
    summarize_predictions,
)
from src.settings import (
    ARTIFACT_DIR,
    MODEL_ARTIFACTS,
    TARGET_COLUMN,
)


@st.cache_resource
def load_encoder() -> Any:
    return joblib.load(ARTIFACT_DIR / "label_encoder.joblib")


@st.cache_resource
def load_model(artifact_name: str) -> Any:
    return joblib.load(ARTIFACT_DIR / artifact_name)


@st.cache_data
def load_training_metrics() -> dict | None:
    metrics_path = ARTIFACT_DIR / "training_metrics.json"
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def get_available_models() -> dict[str, str]:
    return {
        display_name: artifact_name
        for display_name, artifact_name in MODEL_ARTIFACTS.items()
        if (ARTIFACT_DIR / artifact_name).exists()
    }


def extract_scores(model: Any, features: pd.DataFrame) -> Any | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)
    if hasattr(model, "decision_function"):
        return model.decision_function(features)
    return None


def render_dashboard() -> None:
    st.set_page_config(
        page_title="Dry Bean Classification",
        page_icon="Beans",
        layout="wide",
    )
    st.title("Dry Bean Classification System")
    st.write(
        "Upload test data to evaluate the five saved machine-learning "
        "models. No models are retrained from the uploaded file."
    )

    try:
        encoder = load_encoder()
    except FileNotFoundError:
        st.error(
            "Missing model/label_encoder.joblib. "
            "Run train_models.py before starting the app."
        )
        st.stop()

    training_metrics = load_training_metrics()
    if training_metrics:
        st.subheader("Training/test performance comparison")
        comparison = (
            pd.DataFrame(training_metrics)
            .T.rename_axis("Model")
            .reset_index()
        )
        st.dataframe(comparison, use_container_width=True)

    uploaded_file = st.file_uploader(
        "Upload test CSV (include all input features and the Class column)",
        type=["csv"],
    )
    if uploaded_file is None:
        return

    test_data = pd.read_csv(uploaded_file)
    if TARGET_COLUMN not in test_data.columns:
        st.error(
            f"The uploaded CSV must contain the target column "
            f"'{TARGET_COLUMN}'."
        )
        st.stop()

    true_labels_text = test_data[TARGET_COLUMN]
    test_features = test_data.drop(columns=[TARGET_COLUMN])

    known_labels = set(map(str, encoder.classes_))
    missing_labels = sorted(
        set(true_labels_text.astype(str)) - known_labels
    )
    if missing_labels:
        st.error(
            "Unknown class labels found: " + ", ".join(missing_labels)
        )
        st.stop()

    true_labels = encoder.transform(true_labels_text)
    available_models = get_available_models()
    if not available_models:
        st.error("No trained model files found. Run train_models.py first.")
        st.stop()

    selected_name = st.selectbox("Select a model", list(available_models))
    selected_model = load_model(available_models[selected_name])

    try:
        predicted_labels = selected_model.predict(test_features)
    except ValueError as error:
        st.error(
            "The uploaded CSV does not match the trained feature format: "
            f"{error}"
        )
        st.stop()

    scores = extract_scores(selected_model, test_features)
    results = summarize_predictions(
        true_labels,
        predicted_labels,
        scores,
        len(encoder.classes_),
    )

    st.subheader(f"Evaluation results: {selected_name}")
    metric_columns = st.columns(3)
    for index, (metric_name, metric_value) in enumerate(results.items()):
        display_value = (
            f"{metric_value:.4f}"
            if isinstance(metric_value, float)
            else str(metric_value)
        )
        metric_columns[index % 3].metric(metric_name, display_value)

    matrix = confusion_table(
        true_labels,
        predicted_labels,
        encoder.classes_,
    )
    matrix_frame = pd.DataFrame(
        matrix,
        index=encoder.classes_,
        columns=encoder.classes_,
    )

    st.subheader("Confusion Matrix")
    st.dataframe(matrix_frame, use_container_width=True)

    st.subheader("Classification Report")
    report = build_classification_report(
        true_labels,
        predicted_labels,
        encoder.classes_,
    )
    st.code(report)
