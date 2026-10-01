from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import pandas as pd
import sys
from pathlib import Path
import logging

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from inference import RiceInferencePipeline, InsufficientHistoryError

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the pipeline instance
pipeline_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model once
    global pipeline_instance
    logger.info("Loading RiceInferencePipeline...")
    try:
        pipeline_instance = RiceInferencePipeline()
        logger.info("RiceInferencePipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError("Failed to load ML model during startup.")
    yield
    # Shutdown logic
    logger.info("Shutting down API and releasing resources.")
    pipeline_instance = None

app = FastAPI(title="Rice Price and Volatility API", lifespan=lifespan)

# --- Pydantic Schemas ---

class HistoricalObservation(BaseModel):
    reported_date: str = Field(..., description="Date in YYYY-MM-DD format")
    modal_price: float = Field(..., description="Modal price in Rs./Quintal")
    arrivals: float = Field(..., description="Arrivals in Tonnes")

class PredictRequest(BaseModel):
    market: str
    variety: str
    prediction_date: str = Field(..., description="Date in YYYY-MM-DD format")
    history: List[HistoricalObservation]

class PredictResponse(BaseModel):
    market: str
    variety: str
    prediction_date: str
    predicted_modal_price: float
    predicted_volatility_range: float
    predicted_volatility_class: str

# --- Endpoints ---

@app.get("/health")
async def health_check():
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "status": "ok",
        "model": "rice-price-xgb",
        "volatility": "persistence"
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    if not request.history:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty history provided.")
        
    # Convert Pydantic request to pandas DataFrame expected by inference.py
    data = []
    for obs in request.history:
        data.append({
            "Market Name": request.market,
            "Variety": request.variety,
            "Reported Date": obs.reported_date,
            "Modal Price (Rs./Quintal)": obs.modal_price,
            "Arrivals (Tonnes)": obs.arrivals
        })
        
    df = pd.DataFrame(data)
    
    try:
        result = pipeline_instance.predict(
            df=df,
            market=request.market,
            variety=request.variety,
            prediction_date=request.prediction_date
        )
        return PredictResponse(**result)
        
    except InsufficientHistoryError as e:
        logger.warning(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
    except ValueError as e:
        logger.warning(f"Value error during prediction: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during prediction.")
