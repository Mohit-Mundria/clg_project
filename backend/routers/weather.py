"""
KisanAI - Weather Router
Real-time weather data and farming advisory
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.services.weather_service import get_weather_data
from backend.services.groq_service import get_groq_response

router = APIRouter(prefix="/api/weather", tags=["Weather"])


@router.get("/{city}")
async def get_weather(city: str, language: str = "english"):
    """
    Get real-time weather data for a city and generate farming advisory.
    """
    if not city.strip():
        raise HTTPException(status_code=400, detail="City name cannot be empty")

    weather = await get_weather_data(city.strip())

    if not weather.get("success"):
        raise HTTPException(status_code=404, detail=weather.get("error", "Weather data not found"))

    return weather


class SoilAnalysisRequest(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    organic_matter: float = 2.0
    soil_type: str = "loamy"
    rainfall: float = 100.0
    temperature: float = 25.0


@router.post("/soil-analysis")
async def analyze_soil(request: SoilAnalysisRequest):
    """Analyze soil health and generate AI recommendations."""
    from backend.services.groq_service import generate_soil_advice

    soil_data = {
        "nitrogen": request.nitrogen,
        "phosphorus": request.phosphorus,
        "potassium": request.potassium,
        "ph": request.ph,
        "organic_matter": request.organic_matter,
        "soil_type": request.soil_type,
        "rainfall": request.rainfall,
        "temperature": request.temperature,
    }

    advice = generate_soil_advice(soil_data)

    # Calculate soil health score (0-10)
    score = calculate_soil_score(
        n=request.nitrogen, p=request.phosphorus, k=request.potassium,
        ph=request.ph, om=request.organic_matter
    )

    return {
        "success": True,
        "soil_health_score": score,
        "score_label": get_score_label(score),
        "ai_advice": advice,
        "inputs": soil_data,
        "nutrient_levels": {
            "nitrogen": classify_nutrient(request.nitrogen, "N"),
            "phosphorus": classify_nutrient(request.phosphorus, "P"),
            "potassium": classify_nutrient(request.potassium, "K"),
        }
    }


def calculate_soil_score(n, p, k, ph, om) -> float:
    score = 0
    # Nitrogen score (optimal: 80-120)
    if 80 <= n <= 120: score += 2
    elif 60 <= n < 80 or 120 < n <= 140: score += 1.5
    elif 40 <= n < 60: score += 1
    else: score += 0.5
    # Phosphorus score (optimal: 40-80)
    if 40 <= p <= 80: score += 2
    elif 25 <= p < 40 or 80 < p <= 100: score += 1.5
    elif 15 <= p < 25: score += 1
    else: score += 0.5
    # Potassium score (optimal: 80-120)
    if 80 <= k <= 120: score += 2
    elif 60 <= k < 80 or 120 < k <= 140: score += 1.5
    elif 40 <= k < 60: score += 1
    else: score += 0.5
    # pH score (optimal: 6.0-7.5)
    if 6.0 <= ph <= 7.5: score += 2
    elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0: score += 1.5
    elif 5.0 <= ph < 5.5: score += 1
    else: score += 0.3
    # Organic matter score (optimal: >2%)
    if om >= 3.0: score += 2
    elif om >= 2.0: score += 1.5
    elif om >= 1.0: score += 1
    else: score += 0.3

    return round(min(score, 10), 1)


def get_score_label(score: float) -> str:
    if score >= 8.5: return "Excellent 🌟"
    if score >= 7.0: return "Good ✅"
    if score >= 5.5: return "Moderate ⚠️"
    if score >= 4.0: return "Poor 🔴"
    return "Critical ❌"


def classify_nutrient(value: float, nutrient: str) -> str:
    thresholds = {"N": (50, 100), "P": (25, 60), "K": (50, 100)}
    low, high = thresholds.get(nutrient, (50, 100))
    if value < low: return "Low"
    if value > high: return "High"
    return "Optimal"
