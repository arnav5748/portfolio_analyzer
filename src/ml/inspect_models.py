
import joblib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def inspect_model(name, model_path, scaler_path=None):

    print(f"\n{'=' * 55}")
    print(f"MODEL: {name}")
    print(f"{'=' * 55}")

    model = joblib.load(PROJECT_ROOT / model_path)

    print("Model type:", type(model).__name__)

    if hasattr(model, "n_features_in_"):
        print("Expected features:", model.n_features_in_)

    if hasattr(model, "feature_names_in_"):
        print("Feature names:", list(model.feature_names_in_))

    if hasattr(model, "classes_"):
        print("Classes:", model.classes_)

    if hasattr(model, "means_"):
        print("Number of clusters:", model.means_.shape[0])

    if hasattr(model, "transmat_"):
        print("Number of HMM states:", model.transmat_.shape[0])

    if scaler_path:
        scaler = joblib.load(PROJECT_ROOT / scaler_path)

        print("\nScaler type:", type(scaler).__name__)
        print("Scaler expected features:", scaler.n_features_in_)

        if hasattr(scaler, "feature_names_in_"):
            print("Scaler feature names:", list(scaler.feature_names_in_))


inspect_model(
    "GMM",
    "data/processed/regime/gmm/gmm_model.pkl",
    "data/processed/regime/gmm/gmm_scaler.pkl"
)

inspect_model(
    "HMM",
    "data/processed/regime/hmm/hmm_model.pkl",
    "data/processed/regime/hmm/hmm_scaler.pkl"
)

inspect_model(
    "Risk Regression",
    "models/regression/best_regression_model.pkl"
)

inspect_model(
    "Risk Classification",
    "models/risk_forecasting/best_risk_model.pkl"
)