
import joblib
import pandas as pd
import numpy as np
import shap

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

GMM_MODEL_PATH = ROOT / "data/processed/regime/gmm/gmm_model.pkl"
GMM_SCALER_PATH = ROOT / "data/processed/regime/gmm/gmm_scaler.pkl"

HMM_MODEL_PATH = ROOT / "data/processed/regime/hmm/hmm_model.pkl"
HMM_SCALER_PATH = ROOT / "data/processed/regime/hmm/hmm_scaler.pkl"

REGRESSION_MODEL_PATH = ROOT / "models/regression/best_regression_model.pkl"

CLASSIFICATION_MODEL_PATH = (
    ROOT / "models/risk_forecasting/balanced_logistic_regression.pkl"
)


# --------------------------------------------------
# FEATURE DEFINITIONS
# --------------------------------------------------

REGIME_FEATURES = [
    "portfolio_return",
    "portfolio_volatility_20d",
    "portfolio_drawdown",
    "VIX",
    "mkt_rf",
    "TREASURY_10Y",
]

FORECAST_FEATURES = [
    "portfolio_return",
    "portfolio_volatility_20d",
    "annualized_portfolio_volatility",
    "portfolio_drawdown",
    "mkt_rf",
    "smb",
    "hml",
    "rmw",
    "cma",
    "mom",
    "rf",
    "CPI",
    "FEDERAL_FUNDS_RATE",
    "TREASURY_10Y",
    "UNEMPLOYMENT_RATE",
    "VIX",
]


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

def load_models():

    return {
        "gmm": joblib.load(GMM_MODEL_PATH),
        "gmm_scaler": joblib.load(GMM_SCALER_PATH),

        "hmm": joblib.load(HMM_MODEL_PATH),
        "hmm_scaler": joblib.load(HMM_SCALER_PATH),

        "regressor": joblib.load(REGRESSION_MODEL_PATH),
        "classifier": joblib.load(CLASSIFICATION_MODEL_PATH),
    }


# --------------------------------------------------
# REGIME LABELS
# --------------------------------------------------

GMM_LABELS = {
    0: "Calm / Low Volatility",
    1: "Favorable / Low Drawdown",
    2: "Stressed / Elevated Volatility",
    3: "Extreme Volatility / Severe Stress",
}

HMM_LABELS = {
    0: "Favorable / Low Drawdown",
    1: "Stressed / Elevated Volatility",
    2: "Extreme Volatility / Severe Stress",
    3: "Calm / Low Volatility",
}


# --------------------------------------------------
# SHAP EXPLANATION
# --------------------------------------------------

def explain_prediction(model, X_forecast, top_n=5):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_forecast)

    # Handle the single-row prediction
    contributions = np.asarray(shap_values).reshape(-1)

    # Expected / baseline prediction
    expected_value = np.asarray(
        explainer.expected_value
    ).reshape(-1)[0]

    explanation_df = pd.DataFrame({
        "feature": X_forecast.columns,
        "shap_value": contributions,
    })

    # Sort by absolute contribution
    explanation_df["absolute_impact"] = (
        explanation_df["shap_value"].abs()
    )

    explanation_df = explanation_df.sort_values(
        "absolute_impact",
        ascending=False
    ).head(top_n)

    explanation = []

    for _, row in explanation_df.iterrows():

        shap_value = float(row["shap_value"])

        explanation.append({
            "feature": row["feature"],
            "shap_value": shap_value,
            "impact": (
                "increases predicted volatility"
                if shap_value > 0
                else "decreases predicted volatility"
                if shap_value < 0
                else "no meaningful impact"
            ),
        })

    return {
        "baseline_prediction": float(expected_value),
        "top_features": explanation,
    }


# --------------------------------------------------
# INFERENCE
# --------------------------------------------------

def analyze_custom_portfolio(features):

    models = load_models()

    df = features.copy()

    df = df.sort_values(
        "trading_date"
    ).reset_index(drop=True)

    # Ensure all required features exist
    required = set(REGIME_FEATURES + FORECAST_FEATURES)

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    # Remove rows with missing model inputs
    valid = df.dropna(
        subset=REGIME_FEATURES + FORECAST_FEATURES
    ).copy()

    if valid.empty:
        raise ValueError(
            "No valid rows available for ML inference."
        )

    # ----------------------------------------------
    # GMM REGIME
    # ----------------------------------------------

    X_regime = valid[REGIME_FEATURES]

    X_gmm = models["gmm_scaler"].transform(X_regime)

    gmm_states = models["gmm"].predict(X_gmm)

    gmm_state = int(gmm_states[-1])

    # ----------------------------------------------
    # HMM REGIME
    # ----------------------------------------------

    X_hmm = models["hmm_scaler"].transform(X_regime)

    hmm_states = models["hmm"].predict(X_hmm)

    hmm_state = int(hmm_states[-1])

    # ----------------------------------------------
    # RISK FORECASTING
    # ----------------------------------------------

    latest = valid.iloc[[-1]]

    X_forecast = latest[FORECAST_FEATURES]

    predicted_volatility = float(
        models["regressor"].predict(X_forecast)[0]
    )

    risk_probability = float(
        models["classifier"].predict_proba(X_forecast)[0][1]
    )

    # ----------------------------------------------
    # SHAP EXPLANATION
    # ----------------------------------------------

    shap_explanation = explain_prediction(
        models["regressor"],
        X_forecast,
        top_n=5
    )

    # ----------------------------------------------
    # RETURN RESULTS
    # ----------------------------------------------

    return {

        "latest_date": str(
            pd.to_datetime(
                latest["trading_date"].iloc[0]
            ).date()
        ),

        "market_regime": {
            "gmm_state": gmm_state,
            "gmm_label": GMM_LABELS[gmm_state],

            "hmm_state": hmm_state,
            "hmm_label": HMM_LABELS[hmm_state],
        },

        "risk_forecast": {
            "predicted_future_volatility": predicted_volatility,
            "high_risk_probability": risk_probability,
        },

        "explanation": shap_explanation,
    }