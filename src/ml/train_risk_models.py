
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "forecasting"

OUTPUT_DIR = BASE_DIR / "models" / "risk_forecasting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = BASE_DIR / "results" / "risk_forecasting"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()

X_val = pd.read_csv(DATA_DIR / "X_validation.csv")
y_val = pd.read_csv(DATA_DIR / "y_validation.csv").squeeze()

X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# Ensure labels are integers
y_train = y_train.astype(int)
y_val = y_val.astype(int)
y_test = y_test.astype(int)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Testing shape:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())


# --------------------------------------------------
# 3. DEFINE MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight=None,
            random_state=42
        ))
    ]),

    "Balanced Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 4. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_model(model, X, y, threshold=0.5):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y, predictions),
        "Precision": precision_score(
            y, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y, probabilities),
        "PR-AUC": average_precision_score(y, probabilities)
    }

    return metrics, probabilities, predictions


# --------------------------------------------------
# 5. TRAIN AND VALIDATE
# --------------------------------------------------

validation_results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    metrics, probabilities, predictions = evaluate_model(
        model, X_val, y_val
    )

    validation_results.append({
        "Model": name,
        **metrics
    })

    trained_models[name] = model

    print("\nValidation metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, predictions))

    print("\nClassification Report:")
    print(classification_report(
        y_val,
        predictions,
        zero_division=0
    ))

    # Save each trained model
    filename = name.lower().replace(" ", "_")

    joblib.dump(
        model,
        OUTPUT_DIR / f"{filename}.pkl"
    )


# --------------------------------------------------
# 6. COMPARE VALIDATION RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(validation_results)

results_df = results_df.sort_values(
    by="PR-AUC",
    ascending=False
)

results_df.to_csv(
    RESULTS_DIR / "validation_model_comparison.csv",
    index=False
)

print("\nVALIDATION MODEL COMPARISON")
print(results_df.to_string(index=False))


# --------------------------------------------------
# 7. SELECT BEST MODEL
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print(f"\nSelected model: {best_model_name}")


# --------------------------------------------------
# 8. THRESHOLD TUNING USING VALIDATION DATA
# --------------------------------------------------

from sklearn.metrics import precision_recall_curve

best_metrics = None
best_threshold = 0.5
best_f1 = -1

# Select the model's probabilities on validation data
val_probabilities = best_model.predict_proba(X_val)[:, 1]

# Try different thresholds
thresholds = np.arange(0.05, 0.96, 0.01)

threshold_results = []

for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val, predictions, zero_division=0
    )

    recall = recall_score(
        y_val, predictions, zero_division=0
    )

    f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    threshold_results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    RESULTS_DIR / "threshold_comparison.csv",
    index=False
)

print("\nTHRESHOLD COMPARISON")
print(
    threshold_df.sort_values(
        "f1_score",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\nSelected threshold:", best_threshold)


# --------------------------------------------------
# FEATURE DISTRIBUTION DIAGNOSTICS
# --------------------------------------------------

print("\nFEATURE DISTRIBUTION COMPARISON")

for feature in X_train.columns:

    print(f"\nFeature: {feature}")

    print(
        "Train mean:",
        round(X_train[feature].mean(), 6)
    )

    print(
        "Validation mean:",
        round(X_val[feature].mean(), 6)
    )

    print(
        "Test mean:",
        round(X_test[feature].mean(), 6)
    )

    print(
        "Train std:",
        round(X_train[feature].std(), 6)
    )

    print(
        "Test std:",
        round(X_test[feature].std(), 6)
    )
    

# --------------------------------------------------
# PORTFOLIO RETURN ANOMALY CHECK
# --------------------------------------------------

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

FORECASTING_DIR = BASE_DIR / "data" / "processed" / "forecasting"

train_df = pd.read_csv(FORECASTING_DIR / "train.csv")
val_df = pd.read_csv(FORECASTING_DIR / "validation.csv")
test_df = pd.read_csv(FORECASTING_DIR / "test.csv")

print("\nPORTFOLIO RETURN ANOMALY CHECK")

for name, part in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    print(f"\n{name}")

    print(
        part[
            [
                "trading_date",
                "portfolio_return",
                "portfolio_volatility_20d",
                "annualized_portfolio_volatility"
            ]
        ]
        .sort_values("portfolio_return")
        .head(5)
        .to_string(index=False)
    )

    print("\nLargest positive returns:")

    print(
        part[
            [
                "trading_date",
                "portfolio_return",
                "portfolio_volatility_20d",
                "annualized_portfolio_volatility"
            ]
        ]
        .sort_values("portfolio_return", ascending=False)
        .head(5)
        .to_string(index=False)
    )


# --------------------------------------------------
# 9. FINAL TEST EVALUATION
# --------------------------------------------------

test_probabilities = best_model.predict_proba(X_test)[:, 1]

print("\nPROBABILITY DISTRIBUTION")

print("\nValidation probabilities:")
print(pd.Series(val_probabilities).describe())

print("\nTest probabilities:")
print(pd.Series(test_probabilities).describe())

print("\nActual test labels:")
print(y_test.value_counts())

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_metrics = {
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "F1 Score": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(y_test, test_probabilities),
    "PR-AUC": average_precision_score(
        y_test, test_probabilities
    )
}

print("\nFINAL TEST RESULTS")

for metric, value in test_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nTest Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=[0, 1]
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        zero_division=0
    )
)


# --------------------------------------------------
# 10. SAVE FINAL MODEL AND RESULTS
# --------------------------------------------------

joblib.dump(
    best_model,
    OUTPUT_DIR / "best_risk_model.pkl"
)

predictions_df = pd.DataFrame({
    "actual_risk": y_test,
    "predicted_risk": test_predictions,
    "risk_probability": test_probabilities
})

predictions_df.to_csv(
    RESULTS_DIR / "test_predictions.csv",
    index=False
)

pd.DataFrame([{
    "Selected Model": best_model_name,
    "Selected Threshold": best_threshold,
    **test_metrics
}]).to_csv(
    RESULTS_DIR / "final_test_metrics.csv",
    index=False
)

print("\nRisk forecasting pipeline completed!")


# --------------------------------------------------
# TARGET AND THRESHOLD DIAGNOSTICS
# --------------------------------------------------

print("\nTARGET AND THRESHOLD DIAGNOSTICS")

for name, part in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:

    print(f"\n{name}")

    print("\nFuture volatility:")
    print(part["future_volatility_10d"].describe())

    print("\nRolling risk threshold:")
    print(part["risk_threshold"].describe())

    print("\nHigh-risk observations:")
    print(part["high_future_risk"].value_counts())

    print("\nAverage future volatility by risk class:")
    print(
        part.groupby("high_future_risk")[
            "future_volatility_10d"
        ].mean()
    )