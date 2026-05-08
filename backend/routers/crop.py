"""
KisanAI - Crop Recommendation Router
ML-powered crop recommendation based on soil and climate data
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from backend.services.ml_service import predict_crop
from backend.services.groq_service import get_groq_response

router = APIRouter(prefix="/api/crop", tags=["Crop Recommendation"])


class CropRequest(BaseModel):
    nitrogen: float = Field(..., ge=0, le=140, description="Nitrogen content (kg/ha)")
    phosphorus: float = Field(..., ge=0, le=145, description="Phosphorus content (kg/ha)")
    potassium: float = Field(..., ge=0, le=205, description="Potassium content (kg/ha)")
    temperature: float = Field(..., ge=0, le=50, description="Temperature (°C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    ph: float = Field(..., ge=0, le=14, description="Soil pH level")
    rainfall: float = Field(..., ge=0, le=300, description="Rainfall (mm)")


@router.post("/recommend")
async def recommend_crop(request: CropRequest):
    """
    Predict the best crop to grow based on soil nutrients and climate data.
    Uses a Random Forest Classifier trained on the Crop Recommendation Dataset.
    """
    result = predict_crop(
        nitrogen=request.nitrogen,
        phosphorus=request.phosphorus,
        potassium=request.potassium,
        temperature=request.temperature,
        humidity=request.humidity,
        ph=request.ph,
        rainfall=request.rainfall,
    )

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Prediction failed"))

    # Generate detailed AI advice for the recommended crop
    crop_name = result["recommended_crop"]
    advice_prompt = (
        f"I recommend growing {crop_name} based on these soil conditions: "
        f"N={request.nitrogen}, P={request.phosphorus}, K={request.potassium}, "
        f"pH={request.ph}, Temperature={request.temperature}°C, "
        f"Humidity={request.humidity}%, Rainfall={request.rainfall}mm. "
        f"Please provide: 1) Why this crop is ideal, 2) Key cultivation tips, "
        f"3) Expected yield, 4) Market demand in India. Be concise and practical."
    )

    ai_advice = get_groq_response(user_message=advice_prompt, topic="crop")
    result["ai_advice"] = ai_advice.get("message", "")

    return result


@router.get("/list")
async def get_crop_list():
    """Return list of all 22 supported crops with info."""
    crops = [
        {"name": "Rice", "season": "Kharif", "category": "Cereal"},
        {"name": "Wheat", "season": "Rabi", "category": "Cereal"},
        {"name": "Maize", "season": "Kharif/Rabi", "category": "Cereal"},
        {"name": "Chickpea", "season": "Rabi", "category": "Pulse"},
        {"name": "Kidneybeans", "season": "Kharif", "category": "Pulse"},
        {"name": "Pigeonpeas", "season": "Kharif", "category": "Pulse"},
        {"name": "Mothbeans", "season": "Kharif", "category": "Pulse"},
        {"name": "Mungbean", "season": "Kharif/Zaid", "category": "Pulse"},
        {"name": "Blackgram", "season": "Kharif", "category": "Pulse"},
        {"name": "Lentil", "season": "Rabi", "category": "Pulse"},
        {"name": "Pomegranate", "season": "Perennial", "category": "Fruit"},
        {"name": "Banana", "season": "Perennial", "category": "Fruit"},
        {"name": "Mango", "season": "Perennial", "category": "Fruit"},
        {"name": "Grapes", "season": "Perennial", "category": "Fruit"},
        {"name": "Watermelon", "season": "Zaid", "category": "Fruit"},
        {"name": "Muskmelon", "season": "Zaid", "category": "Fruit"},
        {"name": "Apple", "season": "Temperate", "category": "Fruit"},
        {"name": "Orange", "season": "Perennial", "category": "Fruit"},
        {"name": "Papaya", "season": "Perennial", "category": "Fruit"},
        {"name": "Coconut", "season": "Perennial", "category": "Plantation"},
        {"name": "Cotton", "season": "Kharif", "category": "Fiber"},
        {"name": "Jute", "season": "Kharif", "category": "Fiber"},
        {"name": "Coffee", "season": "Perennial", "category": "Plantation"},
    ]
    return {"success": True, "crops": crops, "total": len(crops)}
