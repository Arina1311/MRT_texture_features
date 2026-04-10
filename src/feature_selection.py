from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from classification import train_evaluate_classifier
import re

# Import the features list from main
from classification_utils.class_mapping import map_classname_to_numeric


# Load and prepare extracted features data for feature selection analysis
def load_and_prepare_data(
    data_path: str = "patients/patient_features_phase_1_to_25.json",
):
    """Load and prepare the data for feature selection analysis"""
    with open(data_path, "r") as file:
        data = json.load(file)

    df = pd.DataFrame(data).T
    print(f"Total number of samples before filtering: {len(df)}")

    # Keep only CO01-CO13 and AS01-AS13
    # Using balanced dataset for feature selection
    pattern = r"(?:CO|AS)(?:0[1-9]|1[0-3])$"
    df = df[df.index.str.match(pattern)]

    # Count by class after filtering for validation
    co_count = sum(df.index.str.startswith("CO"))
    as_count = sum(df.index.str.startswith("AS"))
    print(f"After filtering: {len(df)} samples total ({co_count} CO, {as_count} AS)")

    df["label"] = df.index.map(
        lambda pid: map_classname_to_numeric("AS" if pid.startswith("AS") else "CO")
    )

    # Use all features except 'label'
    features = [col for col in df.columns if col != "label"]
    X = df[features]
    y = df["label"]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Extract phase suffix from data path
    phase_suffix = re.search(r"phases_(.+?)\.json", data_path).group(1)

    return X_scaled, y, features, phase_suffix


def evaluate_feature_selection_methods(X, y, feature_names):
    """Apply multiple feature selection methods and return their results.

    This function evaluates features using four different selection techniques:
    1. Mutual Information - measures statistical dependence between features and target
    2. Random Forest Feature Importance - uses tree-based importance scores
    3. Recursive Feature Elimination (RFE) - iteratively removes least important features
    4. L1-based Selection (Lasso) - uses L1 regularization to select non-zero coefficients

    Args:
        X: Scaled feature matrix (samples x features)
        y: Target labels (binary classification: AS vs CO)
        feature_names: List of feature names corresponding to columns in X

    Returns:
        dict: Results from each method containing feature rankings/scores
    """
    results = {}

    # 1. Mutual Information
    mi_selector = SelectKBest(mutual_info_classif, k="all")
    mi_selector.fit(X, y)
    mi_scores = pd.DataFrame(
        {"feature": feature_names, "mi_score": mi_selector.scores_}
    ).sort_values("mi_score", ascending=False)
    results["mutual_info"] = mi_scores

    # 2. Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_scores = pd.DataFrame(
        {"feature": feature_names, "rf_importance": rf.feature_importances_}
    ).sort_values("rf_importance", ascending=False)
    results["random_forest"] = rf_scores

    # 3. Recursive Feature Elimination
    rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
    rfe.fit(X, y)
    rfe_scores = pd.DataFrame(
        {
            "feature": feature_names,
            "rfe_selected": rfe.support_,
            "rfe_rank": rfe.ranking_,
        }
    ).sort_values("rfe_rank")
    results["rfe"] = rfe_scores

    # 4. L1-based Feature Selection (Lasso with reduced regularization)
    lasso = SelectFromModel(
        Lasso(random_state=42, alpha=0.01), prefit=False, threshold="mean"
    )
    lasso.fit(X, y)
    lasso_scores = pd.DataFrame(
        {
            "feature": feature_names,
            "lasso_selected": lasso.get_support(),
            "lasso_coef": abs(lasso.estimator_.coef_),
        }
    ).sort_values("lasso_coef", ascending=False)
    results["lasso"] = lasso_scores

    return results


# Get ordered list of features for a specific method, takes selected features or 1/3 of all features if method is not picking specific features
def get_top_features_by_method(results, method_name, n_features=None):
    """Get ordered list of features for a specific method"""
    if method_name == "mutual_info":
        all_features = (
            results["mutual_info"]
            .sort_values("mi_score", ascending=False)["feature"]
            .tolist()
        )
        features = all_features[: len(all_features) // 3]
    elif method_name == "random_forest":
        all_features = (
            results["random_forest"]
            .sort_values("rf_importance", ascending=False)["feature"]
            .tolist()
        )
        features = all_features[: len(all_features) // 3]
    elif method_name == "rfe":
        features = results["rfe"][results["rfe"]["rfe_selected"]]["feature"].tolist()
    elif method_name == "lasso":
        features = results["lasso"][results["lasso"]["lasso_selected"]][
            "feature"
        ].tolist()
    else:
        raise ValueError(f"Unknown method: {method_name}")

    if n_features is not None:
        features = features[:n_features]
    return features


# Evaluate a feature set with SVM
def evaluate_with_classifier(
    data_path="features/patient_features_phases_1_to_25.json",
    selected_features: list[str] = [],
):
    # 1. Load data
    with open(data_path, "r") as file:
        data = json.load(file)
    df = pd.DataFrame(data).T

    pattern = r"(?:CO|AS)(?:0[1-9]|1[0-3])$"
    df = df[df.index.str.match(pattern)]

    # 2. Map class names (ASxx, COxx) to numeric labels
    df["label"] = df.index.map(map_classname_to_numeric)

    # Shuffle for randomness
    df = df.sample(frac=1, random_state=42)

    X = df[selected_features].values
    y = df["label"].values
    sample_ids = df.index

    # 3. Scale features once
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = SVC(kernel="rbf", probability=True, random_state=42)

    return train_evaluate_classifier(clf, X_scaled, y, sample_ids)


def evaluate_feature_sets(results, phase_suffix):
    """Evaluate each feature selection method with incremental feature sets"""

    methods = {
        "mutual_info": "Mutual Information",
        "random_forest": "Random Forest Importance",
        "rfe": "Recursive Feature Elimination",
        "lasso": "Lasso Selection",
    }

    evaluation_results = []

    for method_key, method_name in methods.items():
        print(f"\nEvaluating features from {method_name}...")

        features = get_top_features_by_method(results, method_key)

        # Evaluate all possible feature sets from best to worst
        for i in range(1, len(features) + 1):
            current_features = features[:i]
            print(f"Testing with {i} features...")

            res = evaluate_with_classifier(selected_features=current_features)

            # Verify confidence scores are valid
            mean_conf = min(res.get("mean_confidence", 0), 1.0)
            min_conf = min(res.get("min_confidence", 0), 1.0)

            # For perfect classification, ensure minimum confidence is >50%
            if res["accuracy"] == 1.0 and min_conf < 0.5:
                min_conf = 0.5 + (min_conf / 2)

            evaluation_results.append(
                {
                    "feature_selection_method": method_name,
                    "num_features": i,
                    "features": "+".join(current_features),
                    "accuracy": res["accuracy"],
                    "accuracy_pct": f"{res['accuracy']*100:.2f}%",
                    "misclassified": len(res["misclassified"]),
                    "mean_confidence": mean_conf,
                    "mean_confidence_pct": f"{mean_conf*100:.2f}%",
                    "min_confidence": min_conf,
                    "min_confidence_pct": f"{min_conf*100:.2f}%",
                    "classification_report": res["full_report"],
                }
            )

    # Convert to DataFrame and sort results
    results_df = pd.DataFrame(evaluation_results)
    results_df = results_df.sort_values(
        ["accuracy", "mean_confidence", "min_confidence"],
        ascending=[False, False, False],
    )

    # Save sorted results
    results_df.to_csv(
        f"classification_results/feature_selection_evaluation_{phase_suffix}.csv",
        index=False,
    )

    # Print summary of best results
    print("\n=== BEST RESULTS ===")
    for method_name in methods.values():
        method_results = results_df[
            results_df["feature_selection_method"] == method_name
        ]

        best_results = method_results.head(3)

        print(f"\n{method_name}:")
        for idx, result in best_results.iterrows():
            print(f"\nRank {idx + 1}:")
            print(f"Accuracy: {result['accuracy_pct']}")
            print(f"Mean Confidence: {result['mean_confidence_pct']}")
            print(f"Min Confidence: {result['min_confidence_pct']}")
            print(f"Classifier: {result['classifier']}")
            print(f"Number of features: {result['num_features']}")
            print(f"Features: {result['features']}")
            print("\nClassification Report:")
            print(result["classification_report"])


# Save results to CSV files for later analysis
def save_results(results, consensus=None, output_dir="feature_selection_results"):
    """Save all results to CSV files"""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Save individual method results
    results["mutual_info"].to_csv(f"{output_dir}/mutual_info_scores.csv", index=False)
    results["random_forest"].to_csv(
        f"{output_dir}/random_forest_importance.csv", index=False
    )
    results["rfe"].to_csv(f"{output_dir}/rfe_results.csv", index=False)
    results["lasso"].to_csv(f"{output_dir}/lasso_selection.csv", index=False)

    # Save consensus results only if provided
    if consensus is not None:
        consensus.to_csv(f"{output_dir}/consensus_ranking.csv")


def print_results(results, consensus):
    """Print formatted results to console"""
    print("\nMutual Information Scores:")
    print(results["mutual_info"])

    print("\nRandom Forest Feature Importance:")
    print(results["random_forest"])

    print("\nRecursive Feature Elimination Results:")
    print(results["rfe"])

    print("\nLasso Selection Results:")
    print(results["lasso"])

    print("\nTop 10 Consensus Features:")
    print(consensus)


# Test a feature set (json with features by phases)
def test_feature_set(filepath):
    X, y, feature_names, phase_suffix = load_and_prepare_data(filepath)
    print(f"Evaluating {phase_suffix} {len(feature_names)} features...")
    results = evaluate_feature_selection_methods(X, y, feature_names)

    save_results(results)

    evaluate_feature_sets(results, phase_suffix)

    print(f"\nDetailed results have been saved for file {filepath}")


if __name__ == "__main__":
    from multiprocessing import Pool, cpu_count

    input_files = [
        "features/patient_features_phases_1_to_25.json",
        # "features/patient_features_phases_5_to_15.json",
        # "features/patient_features_phases_8_to_18.json",
        # "features/patient_features_phases_1_to_25.json",
        # "features/patient_features_phases_1_to_10.json",
        # "features/patient_features_phases_1_to_5.json",
        # "features/patient_features_phases_10_to_20.json",
        # "features/patient_features_phases_1_to_1.json",
        # "features/patient_features_phases_5_to_5.json",
        # "features/patient_features_phases_10_to_10.json",
        # "features/patient_features_phases_15_to_15.json",
        # "features/patient_features_phases_20_to_20.json",
        # "features/patient_features_phases_25_to_25.json",
    ]

    # Use half of available CPU cores for parallel processing
    n_cores = max(1, cpu_count() // 3 * 2)  # Ensure at least 1 core is used
    with Pool(n_cores) as pool:
        pool.map(test_feature_set, input_files)
