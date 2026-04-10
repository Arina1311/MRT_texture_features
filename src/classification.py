import json
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import umap

from classification_utils.class_mapping import map_classname_to_numeric
from classification_utils.plot_3d import plot_misclassifications_3d


def train_evaluate_classifier(
    classifier,
    X_scaled: np.ndarray,
    y: np.ndarray,
    sample_ids: pd.Index,
) -> Dict:
    """
    Train and evaluate a classifier using Stratified K-Fold CV on pre-scaled data.
    Returns a dict with accuracy, misclassified samples, classification_report, etc.
    """
    # We will store predictions & confidences for each sample
    all_preds = np.zeros_like(y, dtype=int)
    all_confidences = np.zeros(len(y), dtype=float)

    skf = StratifiedKFold(n_splits=5)
    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        classifier.fit(X_train, y_train)
        fold_preds = classifier.predict(X_test)

        # Confidence
        if hasattr(classifier, "predict_proba"):
            proba = classifier.predict_proba(X_test)
            fold_confidences = np.max(proba, axis=1)
        elif hasattr(classifier, "decision_function"):
            dec_vals = classifier.decision_function(X_test)
            fold_confidences = 1 / (1 + np.exp(-np.abs(dec_vals)))
        else:
            fold_confidences = np.full(len(X_test), 0.75)

        all_preds[test_idx] = fold_preds
        all_confidences[test_idx] = fold_confidences

    # Accuracy, misclassifications, and report
    accuracy = np.mean(all_preds == y)
    misclassified = []
    for i, (true_lbl, pred_lbl) in enumerate(zip(y, all_preds)):
        if true_lbl != pred_lbl:
            misclassified.append(
                {
                    "patient_id": sample_ids[i],
                    "true_label": int(true_lbl),
                    "predicted": int(pred_lbl),
                    "confidence": all_confidences[i],
                    "confidence_pct": f"{all_confidences[i]*100:.2f}%",
                }
            )

    report = classification_report(y, all_preds, zero_division=0)

    return {
        "accuracy": accuracy,
        "misclassified": misclassified,
        "full_report": report,
        # Return data needed for plotting
        "y_true": y,
        "y_pred": all_preds,
        "confidence": all_confidences,
        "mean_confidence": np.mean(all_confidences),
        "min_confidence": np.min(all_confidences),
    }


def evaluate_classifiers(
    data_path="features/patient_features_phases_1_to_25.json",
    selected_features=[],
):
    # 1. Load data
    with open(data_path, "r") as file:
        data = json.load(file)
    df = pd.DataFrame(data).T

    # 2. Map class names (ASxx, COxx) to numeric labels
    df["label"] = df.index.map(map_classname_to_numeric)

    # Shuffle for randomness
    df = df.sample(frac=1, random_state=42)

    if not selected_features:
        feature_columns = df.columns.drop("label")
    else:
        feature_columns = selected_features

    X = df[feature_columns].values

    y = df["label"].values
    sample_ids = df.index

    # 3. Scale features once
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Compute a single UMAP embedding (3D) for all samples
    reducer = umap.UMAP(n_components=3, random_state=42)
    X_3d = reducer.fit_transform(X_scaled)

    # 5. Define classifiers
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM linear": SVC(kernel="linear", probability=True, random_state=42),
        "SVM poly": SVC(kernel="poly", probability=True, random_state=42),
        "SVM rbf": SVC(kernel="rbf", probability=True, random_state=42),
        "SVM sigmoid": SVC(kernel="sigmoid", probability=True, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
    }

    # 6. Evaluate each classifier + Plot
    results = {}
    for name, clf in classifiers.items():
        result = train_evaluate_classifier(clf, X_scaled, y, sample_ids)
        results[name] = result
        print(f"\n\n\n\n{name} Results:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Mean confidence: {result['mean_confidence']:.4f}")
        print(f"  Min confidence: {result['min_confidence']:.4f}")
        print(f"  Classification report:\n{result['full_report']}")

        # if result["misclassified"]:
        #     print("  Misclassified samples:")
        #     for mc in result["misclassified"]:
        #         print(
        #             f"   - {mc['patient_id']} (true={mc['true_label']}, pred={mc['predicted']})"
        #         )

        plot_misclassifications_3d(
            X_3d=X_3d,
            y_true=result["y_true"],
            y_pred=result["y_pred"],
            sample_ids=sample_ids,
            confidences=result["confidence"],
            method_name=name,
            output_dir="plots",
        )

    return results


if __name__ == "__main__":
    evaluate_classifiers(selected_features=[])
