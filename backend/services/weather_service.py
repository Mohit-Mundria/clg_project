"""
KisanAI - Weather Service
Fetches real-time weather data using WeatherAPI.com and generates farming advisory
"""
import httpx
import logging
from typing import Optional
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

WEATHERAPI_BASE_URL = "https://api.weatherapi.com/v1"


async def get_weather_data(city: str) -> dict:
    """Fetch weather from WeatherAPI.com API."""
    if not settings.weather_api_key:
        return get_mock_weather(city)

    # forecast.json gives current + N-day forecast in one call
    url = f"{WEATHERAPI_BASE_URL}/forecast.json"
    params = {
        "key": settings.weather_api_key,
        "q": f"{city},India",
        "days": 5,
        "aqi": "no",
        "alerts": "no",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        return parse_weather_data(data)

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401 or e.response.status_code == 403:
            logger.error("Invalid WeatherAPI key")
            return {"success": False, "error": "Invalid API key. Please check your WEATHER_API_KEY."}
        elif e.response.status_code == 400:
            return {"success": False, "error": f"City '{city}' not found."}
        logger.error(f"Weather HTTP error: {e}")
        return get_mock_weather(city)
    except Exception as e:
        logger.error(f"Weather fetch error: {e}")
        return get_mock_weather(city)


def parse_weather_data(data: dict) -> dict:
    """Parse WeatherAPI.com API response."""
    location = data.get("location", {})
    current = data.get("current", {})
    condition = current.get("condition", {})
    forecast_days = data.get("forecast", {}).get("forecastday", [])

    temperature = current.get("temp_c", 25)
    feels_like = current.get("feelslike_c", temperature)
    humidity = current.get("humidity", 60)
    pressure = current.get("pressure_mb", 1013)
    description = condition.get("text", "Clear").title()
    icon_url = condition.get("icon", "")
    if icon_url.startswith("//"):
        icon_url = "https:" + icon_url
    wind_kph = current.get("wind_kph", 0)
    cloudiness = current.get("cloud", 0)
    rainfall_mm = current.get("precip_mm", 0)
    visibility_km = current.get("vis_km", 10.0)
    uv_index = current.get("uv", "N/A")

    # Map WeatherAPI icon code from URL (e.g. /113.png → 113)
    icon_code = icon_url.split("/")[-1].replace(".png", "") if icon_url else "113"

    advisory = generate_weather_advisory(
        temperature=temperature,
        humidity=humidity,
        description=description,
        wind_speed=wind_kph,
        rainfall=rainfall_mm,
        cloudiness=cloudiness,
    )

    # Build 5-day forecast list
    forecast_list = []
    for day in forecast_days:
        day_data = day.get("day", {})
        day_condition = day_data.get("condition", {})
        day_icon = day_condition.get("icon", "")
        if day_icon.startswith("//"):
            day_icon = "https:" + day_icon

        forecast_list.append({
            "date": day.get("date", ""),
            "temp_max": day_data.get("maxtemp_c", 0),
            "temp_min": day_data.get("mintemp_c", 0),
            "description": day_condition.get("text", "").title(),
            "icon": day_icon,
            "humidity": day_data.get("avghumidity", 0),
        })

    return {
        "success": True,
        "city": location.get("name", "Unknown"),
        "country": location.get("country", "India"),
        "temperature": round(temperature, 1),
        "feels_like": round(feels_like, 1),
        "humidity": humidity,
        "pressure": pressure,
        "description": description,
        "icon_code": icon_code,
        "icon_url": icon_url,
        "wind_speed": round(wind_kph, 1),
        "cloudiness": cloudiness,
        "rainfall_1h": rainfall_mm,
        "uv_index": uv_index,
        "visibility": visibility_km,
        "advisory": advisory,
        "forecast": forecast_list,
        "is_mock": False,
    }


def generate_weather_advisory(temperature: float, humidity: float,
                               description: str, wind_speed: float,
                               rainfall: float, cloudiness: int) -> list[dict]:
    """Generate farming advisories based on weather conditions."""
    advisories = []

    # Temperature advisories
    if temperature > 40:
        advisories.append({
            "type": "warning",
            "icon": "🌡️",
            "title": "Extreme Heat Alert",
            "message": "Irrigate crops in early morning or evening. Mulch soil to retain moisture. Avoid pesticide spraying.",
        })
    elif temperature < 5:
        advisories.append({
            "type": "warning",
            "icon": "❄️",
            "title": "Frost Risk",
            "message": "Protect sensitive crops with covers. Irrigate before frost as wet soil holds more heat.",
        })
    elif 20 <= temperature <= 30:
        advisories.append({
            "type": "success",
            "icon": "✅",
            "title": "Ideal Temperature",
            "message": "Great conditions for most crops. Consider sowing or transplanting today.",
        })

    # Humidity advisories
    if humidity > 80:
        advisories.append({
            "type": "caution",
            "icon": "💧",
            "title": "High Humidity",
            "message": "Risk of fungal diseases. Ensure good ventilation in fields. Avoid overhead irrigation.",
        })
    elif humidity < 30:
        advisories.append({
            "type": "caution",
            "icon": "🏜️",
            "title": "Low Humidity",
            "message": "Increase irrigation frequency. Consider mulching to conserve soil moisture.",
        })

    # Rain advisories
    if "rain" in description.lower() or rainfall > 0:
        advisories.append({
            "type": "info",
            "icon": "🌧️",
            "title": "Rainfall Expected",
            "message": "Hold off on irrigation today. Ensure proper drainage in your fields to prevent waterlogging.",
        })

    # Wind advisories
    if wind_speed > 40:
        advisories.append({
            "type": "warning",
            "icon": "💨",
            "title": "Strong Winds",
            "message": "Avoid spraying pesticides or fertilizers. Stake tall crops to prevent lodging.",
        })

    # Default positive advisory
    if not advisories:
        advisories.append({
            "type": "info",
            "icon": "🌤️",
            "title": "Good Farming Day",
            "message": "Weather conditions are suitable for regular farm activities. Great day for field inspection.",
        })

    return advisories


def get_mock_weather(city: str) -> dict:
    """Return mock weather data when API key is not available."""
    return {
        "success": True,
        "city": city,
        "country": "India",
        "temperature": 28.5,
        "feels_like": 31.0,
        "humidity": 65,
        "pressure": 1010,
        "description": "Partly Cloudy",
        "icon_code": "116",
        "icon_url": "https://cdn.weatherapi.com/weather/64x64/day/116.png",
        "wind_speed": 12.5,
        "cloudiness": 40,
        "rainfall_1h": 0,
        "uv_index": "7",
        "visibility": 10.0,
        "advisory": [
            {
                "type": "info",
                "icon": "🌤️",
                "title": "Good Farming Day",
                "message": "Conditions are favorable for farming activities. Live weather requires WeatherAPI key.",
            },
            {
                "type": "caution",
                "icon": "⚙️",
                "title": "Demo Mode",
                "message": "Add your WeatherAPI key to .env for real-time weather data.",
            },
        ],
        "forecast": [
            {"date": "2024-06-01", "temp_max": 32, "temp_min": 24, "description": "Sunny", "icon": "https://cdn.weatherapi.com/weather/64x64/day/113.png", "humidity": 60},
            {"date": "2024-06-02", "temp_max": 30, "temp_min": 22, "description": "Partly Cloudy", "icon": "https://cdn.weatherapi.com/weather/64x64/day/116.png", "humidity": 65},
            {"date": "2024-06-03", "temp_max": 27, "temp_min": 21, "description": "Light Rain", "icon": "https://cdn.weatherapi.com/weather/64x64/day/296.png", "humidity": 75},
            {"date": "2024-06-04", "temp_max": 29, "temp_min": 23, "description": "Cloudy", "icon": "https://cdn.weatherapi.com/weather/64x64/day/119.png", "humidity": 70},
            {"date": "2024-06-05", "temp_max": 33, "temp_min": 25, "description": "Sunny", "icon": "https://cdn.weatherapi.com/weather/64x64/day/113.png", "humidity": 55},
        ],
        "is_mock": True,
    }
