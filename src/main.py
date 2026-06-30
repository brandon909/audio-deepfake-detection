import os
import time
from pathlib import Path

import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

TEST_LIMIT = None  # Use 100 for testing, None for full dataset
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results"


def extract_features(file_path):
    """Extract MFCC, delta, and spectral summary features from one audio file."""
    y, sr = librosa.load(file_path, sr=16000)

    # If the file is empty, create a short silent signal
    if len(y) == 0:
        y = np.zeros(2048)

    # If the file is silence, keep it as silence
    elif np.max(np.abs(y)) == 0:
        y = np.zeros(2048)

    else:
        y_trimmed, _ = librosa.effects.trim(y)

        # If trimming removes everything, keep a silent signal
        if len(y_trimmed) == 0:
            y = np.zeros(2048)
        else:
            y = y_trimmed

    # Make sure signal is long enough for feature extraction
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc, width=3)
    delta2 = librosa.feature.delta(mfcc, order=2, width=3)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = []

    for feature in [mfcc, delta, delta2, centroid, bandwidth, rolloff, zcr, rms]:
        features.extend(np.mean(feature, axis=1))
        features.extend(np.std(feature, axis=1))
        features.extend(np.min(feature, axis=1))
        features.extend(np.max(feature, axis=1))

    return np.array(features)


def load_split(split_path):
    X = []
    y = []

    labels = {
        "real": 0,
        "fake": 1
    }

    for label_name, label in labels.items():
        folder_path = os.path.join(split_path, label_name)

        files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        files.sort()

        if TEST_LIMIT is not None:
            files = files[:TEST_LIMIT]

        print(f"\nLoading {label_name} files ({len(files)} files)...")

        for file_name in tqdm(
                files,
                desc=f"Processing {label_name}",
                unit="file"
        ):
            file_path = os.path.join(folder_path, file_name)

            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(label)

            except Exception as e:
                print(f"\nError processing {file_name}")
                print(e)

        print(f"Finished loading {label_name}")

    return np.array(X), np.array(y)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print("\n" + name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

    cm = confusion_matrix(y_test, pred)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Real", "Fake"]
    )

    display.plot()
    plt.title(name + " Confusion Matrix")
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / (name.replace(" ", "_") + "_confusion_matrix.png"))
    plt.close()

    return accuracy, precision, recall, f1


def plot_model_comparison(results):
    model_names = list(results.keys())

    accuracies = [results[name]["accuracy"] for name in model_names]
    precisions = [results[name]["precision"] for name in model_names]
    recalls = [results[name]["recall"] for name in model_names]
    f1_scores = [results[name]["f1"] for name in model_names]

    x = np.arange(len(model_names))
    width = 0.2

    plt.figure(figsize=(10, 6))
    plt.bar(x - 1.5 * width, accuracies, width, label="Accuracy")
    plt.bar(x - 0.5 * width, precisions, width, label="Precision")
    plt.bar(x + 0.5 * width, recalls, width, label="Recall")
    plt.bar(x + 1.5 * width, f1_scores, width, label="F1-score")

    plt.xticks(x, model_names, rotation=20)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "model_performance_comparison.png")
    plt.close()


def main():
    start = time.time()
    base_path = os.environ.get(
        "AUDIO_DATASET_DIR",
        "/Users/brandon/Downloads/archive/for-norm/for-norm",
    )

    train_path = os.path.join(base_path, "training")
    val_path = os.path.join(base_path, "validation")
    test_path = os.path.join(base_path, "testing")

    print("Loading training data...")
    X_train, y_train = load_split(train_path)

    print("Loading validation data...")
    X_val, y_val = load_split(val_path)

    print("Loading testing data...")
    X_test, y_test = load_split(test_path)

    X_train = np.vstack((X_train, X_val))
    y_train = np.concatenate((y_train, y_val))

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Feature vector length:", X_train.shape[1])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "kNN": KNeighborsClassifier(n_neighbors=5),

        "Logistic Regression": LogisticRegression(max_iter=3000),

        "Naive Bayes": GaussianNB(),

        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        accuracy, precision, recall, f1 = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results[name] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    plot_model_comparison(results)

    print("\nDone. Plots saved as PNG files.")
    print(f"\nTotal runtime: {(time.time() - start) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
