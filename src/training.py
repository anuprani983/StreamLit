import json

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.settings import (
    ARTIFACT_DIR,
    DATASET_PATH,
    MODEL_ARTIFACTS,
    TARGET_COLUMN,
    TEST_DATA_PATH,
)


RANDOM_STATE = 42


def create_estimators() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_STATE,
            )),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def train_models() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_excel(DATASET_PATH, sheet_name="Dry_Beans_Dataset")
    features = dataset.drop(columns=[TARGET_COLUMN])

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(dataset[TARGET_COLUMN])
    joblib.dump(label_encoder, ARTIFACT_DIR / "label_encoder.joblib")

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    metrics = {}
    for model_name, estimator in create_estimators().items():
        estimator.fit(train_features, train_labels)
        predictions = estimator.predict(test_features)
        probabilities = (
            estimator.predict_proba(test_features)
            if hasattr(estimator, "predict_proba")
            else estimator.decision_function(test_features)
        )

        try:
            auc = roc_auc_score(
                test_labels,
                probabilities,
                multi_class="ovr",
                average="weighted",
            )
        except ValueError:
            auc = None

        metrics[model_name] = {
            "Accuracy": round(
                accuracy_score(test_labels, predictions),
                4,
            ),
            "AUC": round(auc, 4) if auc is not None else None,
            "Precision": round(
                precision_score(
                    test_labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                4,
            ),
            "Recall": round(
                recall_score(
                    test_labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                4,
            ),
            "F1": round(
                f1_score(
                    test_labels,
                    predictions,
                    average="weighted",
                    zero_division=0,
                ),
                4,
            ),
            "MCC": round(
                matthews_corrcoef(test_labels, predictions),
                4,
            ),
        }
        artifact_name = MODEL_ARTIFACTS[model_name]
        joblib.dump(estimator, ARTIFACT_DIR / artifact_name)

    with (ARTIFACT_DIR / "training_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    held_out_data = test_features.copy()
    held_out_data[TARGET_COLUMN] = label_encoder.inverse_transform(test_labels)
    held_out_data.to_csv(TEST_DATA_PATH, index=False)

    print(f"Dataset shape: {dataset.shape}")
    print(json.dumps(metrics, indent=2))
    print(
        f"Saved {TEST_DATA_PATH.name} with "
        f"{len(held_out_data)} rows"
    )
    return metrics
