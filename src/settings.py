from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "model"
DATASET_PATH = PROJECT_ROOT / "Dry_Bean_Dataset.xlsx"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
TARGET_COLUMN = "Class"

MODEL_ARTIFACTS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
