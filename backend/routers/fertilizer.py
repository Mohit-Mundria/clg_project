"""
KisanAI - Fertilizer Recommendation Router
ML-powered fertilizer suggestion based on soil and crop data
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from backend.services.ml_service import predict_fertilizer, SOIL_TYPE_MAP, CROP_TYPE_MAP
from backend.services.groq_service import get_groq_response

router = APIRouter(prefix="/api/fertilizer", tags=["Fertilizer Recommendation"])


class FertilizerRequest(BaseModel):
    soil_type: str = Field(..., description="Soil type: sandy, loamy, black, red, clayey, alluvial, laterite")
    crop_type: str = Field(..., description="Crop type: rice, wheat, maize, sugarcane, cotton, pulses, vegetables, fruits, oilseeds")
    nitrogen: float = Field(..., ge=0, le=140, description="Nitrogen (N) level in soil (kg/ha)")
    phosphorus: float = Field(..., ge=0, le=145, description="Phosphorus (P) level in soil (kg/ha)")
    potassium: float = Field(..., ge=0, le=205, description="Potassium (K) level in soil (kg/ha)")
    temperature: float = Field(default=25.0, ge=0, le=50, description="Temperature (°C)")
    humidity: float = Field(default=60.0, ge=0, le=100, description="Humidity (%)")
    moisture: float = Field(default=30.0, ge=0, le=100, description="Soil moisture (%)")


@router.post("/recommend")
async def recommend_fertilizer(request: FertilizerRequest):
    """
    Predict the best fertilizer based on soil conditions and crop type.
    Uses Gradient Boosting Classifier trained on fertilizer recommendation dataset.
    """
    result = predict_fertilizer(
        soil_type=request.soil_type,
        crop_type=request.crop_type,
        nitrogen=request.nitrogen,
        phosphorus=request.phosphorus,
        potassium=request.potassium,
        temperature=request.temperature,
        humidity=request.humidity,
        moisture=request.moisture,
    )

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Prediction failed"))

    # Generate detailed AI advice for the fertilizer
    fertilizer_name = result["recommended_fertilizer"]
    advice_prompt = (
        f"For {request.crop_type} grown in {request.soil_type} soil with "
        f"N={request.nitrogen}, P={request.phosphorus}, K={request.potassium} kg/ha, "
        f"I recommend using {fertilizer_name} fertilizer. "
        f"Please provide: 1) Application rate and timing, "
        f"2) How to apply (broadcast/side-dress/fertigation), "
        f"3) Complementary organic inputs, "
        f"4) Signs of over/under application. Be concise and practical for Indian farmers."
    )

    ai_advice = get_groq_response(user_message=advice_prompt, topic="fertilizer")
    result["ai_advice"] = ai_advice.get("message", "")

    return result


@router.get("/soil-types")
async def get_soil_types():
    """Return list of supported soil types."""
    return {
        "success": True,
        "soil_types": list(SOIL_TYPE_MAP.keys()),
        "descriptions": {
            "sandy": "Light, dry soil — good drainage, low nutrients",
            "loamy": "Ideal mix of sand, silt, clay — best for most crops",
            "black": "Black cotton soil — high water retention, rich in minerals",
            "red": "Iron-rich soil — found in peninsular India, well-drained",
            "clayey": "Heavy soil — poor drainage, high nutrient retention",
            "alluvial": "River-deposited soil — very fertile, ideal for rice/wheat",
            "laterite": "Weathered tropical soil — poor in nutrients, needs amendments"
        }
    }


@router.get("/crop-types")
async def get_crop_types():
    """Return list of supported crop types."""
    return {
        "success": True,
        "crop_types": list(CROP_TYPE_MAP.keys())
    }
