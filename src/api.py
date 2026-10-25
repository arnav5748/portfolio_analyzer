
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict

from src.portfolio_engine import analyze_portfolio
from src.ml.custom_portfolio_features import build_custom_portfolio_features
from src.ml.inference_engine import analyze_custom_portfolio


app = FastAPI(
    title="Portfolio Risk Analyzer API",
    description="API for custom portfolio performance and ML-based risk analysis",
    version="1.1.0"
)


# Enable communication between React and FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PortfolioRequest(BaseModel):
    weights: Dict[str, float] = Field(
        ...,
        description="Ticker-to-weight mapping"
    )


@app.get("/")
def home():
    return {
        "message": "Portfolio Risk Analyzer API is running"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
def analyze(request: PortfolioRequest):

    try:
        # 1. Calculate traditional portfolio metrics
        portfolio_result = analyze_portfolio(request.weights)

        # 2. Generate features for the custom-weight portfolio
        features = build_custom_portfolio_features(
            request.weights
        )

        # 3. Run trained ML models
        ml_result = analyze_custom_portfolio(features)

        # 4. Combine both results
        return {
            "portfolio": portfolio_result,
            "ml_analysis": ml_result
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )