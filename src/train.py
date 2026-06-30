"""Train baseline machine learning models for audio deepfake detection."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models() -> dict[str, Pipeline]:
    """Create baseline model pipelines with standardization where appropriate."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        ),
        "kNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", MLPClassifier(max_iter=500, random_state=42)),
            ]
        ),
    }


def train_and_compare(features, labels, test_size: float = 0.2, random_state: int = 42) -> list[dict]:
    """Train baseline models and return evaluation metrics."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    results = []
    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions, zero_division=0),
                "recall": recall_score(y_test, predictions, zero_division=0),
                "f1": f1_score(y_test, predictions, zero_division=0),
            }
        )

    return results

