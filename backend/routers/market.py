"""
KisanAI - Market Prices Router
Simulated mandi (agricultural market) price data for Indian crops
"""
from fastapi import APIRouter
from datetime import datetime, timedelta
import random

router = APIRouter(prefix="/api/market", tags=["Market Prices"])

# Base prices in ₹/quintal (100 kg) — approximate MSP + market rates
BASE_PRICES = {
    "Wheat": {"msp": 2275, "market": 2350, "unit": "₹/quintal", "state": "Punjab"},
    "Rice (Common)": {"msp": 2183, "market": 2100, "unit": "₹/quintal", "state": "Haryana"},
    "Rice (Grade A)": {"msp": 2203, "market": 2280, "unit": "₹/quintal", "state": "Uttar Pradesh"},
    "Maize": {"msp": 1962, "market": 1850, "unit": "₹/quintal", "state": "Bihar"},
    "Bajra": {"msp": 2500, "market": 2400, "unit": "₹/quintal", "state": "Rajasthan"},
    "Jowar": {"msp": 3180, "market": 3050, "unit": "₹/quintal", "state": "Maharashtra"},
    "Tur/Arhar Dal": {"msp": 7000, "market": 9500, "unit": "₹/quintal", "state": "Madhya Pradesh"},
    "Moong Dal": {"msp": 8558, "market": 9200, "unit": "₹/quintal", "state": "Rajasthan"},
    "Urad Dal": {"msp": 7400, "market": 8800, "unit": "₹/quintal", "state": "Andhra Pradesh"},
    "Chana Dal": {"msp": 5440, "market": 5600, "unit": "₹/quintal", "state": "Madhya Pradesh"},
    "Masoor Dal": {"msp": 6425, "market": 6800, "unit": "₹/quintal", "state": "Uttar Pradesh"},
    "Groundnut": {"msp": 6783, "market": 7200, "unit": "₹/quintal", "state": "Gujarat"},
    "Sunflower": {"msp": 6760, "market": 6400, "unit": "₹/quintal", "state": "Karnataka"},
    "Soybean": {"msp": 4600, "market": 4400, "unit": "₹/quintal", "state": "Madhya Pradesh"},
    "Mustard": {"msp": 5650, "market": 5900, "unit": "₹/quintal", "state": "Rajasthan"},
    "Sugarcane": {"msp": 340, "market": 350, "unit": "₹/quintal", "state": "Uttar Pradesh"},
    "Cotton (Long)": {"msp": 7020, "market": 7200, "unit": "₹/quintal", "state": "Gujarat"},
    "Cotton (Medium)": {"msp": 6620, "market": 6800, "unit": "₹/quintal", "state": "Maharashtra"},
    "Jute": {"msp": 5335, "market": 5100, "unit": "₹/quintal", "state": "West Bengal"},
    "Onion": {"msp": 0, "market": 1800, "unit": "₹/quintal", "state": "Maharashtra"},
    "Potato": {"msp": 0, "market": 900, "unit": "₹/quintal", "state": "Uttar Pradesh"},
    "Tomato": {"msp": 0, "market": 2500, "unit": "₹/quintal", "state": "Karnataka"},
}


def get_live_price(base_price: int) -> dict:
    """Simulate live market price with small daily variation."""
    variation = random.uniform(-0.05, 0.08)  # -5% to +8%
    current = int(base_price * (1 + variation))
    change = current - base_price
    change_pct = round((change / base_price) * 100, 2) if base_price > 0 else 0
    return {
        "current": current,
        "change": change,
        "change_pct": change_pct,
        "trend": "up" if change > 0 else "down" if change < 0 else "stable"
    }


@router.get("/prices")
async def get_market_prices():
    """
    Get current market prices for major Indian agricultural commodities.
    Data includes MSP (Minimum Support Price) and market prices with daily variation.
    """
    prices = []
    for crop, data in BASE_PRICES.items():
        live = get_live_price(data["market"])
        prices.append({
            "crop": crop,
            "msp": data["msp"],
            "market_price": live["current"],
            "change": live["change"],
            "change_pct": live["change_pct"],
            "trend": live["trend"],
            "unit": data["unit"],
            "major_market": data["state"],
            "above_msp": live["current"] > data["msp"] if data["msp"] > 0 else None,
        })

    return {
        "success": True,
        "last_updated": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "source": "e-NAM (National Agriculture Market) — Simulated",
        "note": "Prices are indicative. Check your local mandi for actual rates.",
        "prices": prices,
        "total": len(prices),
    }


@router.get("/prices/{crop_name}")
async def get_crop_price(crop_name: str):
    """Get market price for a specific crop."""
    # Case-insensitive search
    found = None
    for crop, data in BASE_PRICES.items():
        if crop.lower() == crop_name.lower() or crop_name.lower() in crop.lower():
            found = (crop, data)
            break

    if not found:
        return {
            "success": False,
            "error": f"Crop '{crop_name}' not found in market data",
            "available_crops": list(BASE_PRICES.keys())
        }

    crop, data = found
    live = get_live_price(data["market"])

    return {
        "success": True,
        "crop": crop,
        "msp": data["msp"],
        "market_price": live["current"],
        "change": live["change"],
        "change_pct": live["change_pct"],
        "trend": live["trend"],
        "unit": data["unit"],
        "major_market": data["state"],
        "last_updated": datetime.now().strftime("%d %b %Y, %I:%M %p"),
    }


@router.get("/schemes")
async def get_government_schemes():
    """Return major government schemes for Indian farmers."""
    schemes = [
        {
            "name": "PM-KISAN",
            "full_name": "Pradhan Mantri Kisan Samman Nidhi",
            "benefit": "₹6,000/year direct income support (3 installments of ₹2,000)",
            "eligibility": "All land-holding farmer families",
            "apply_at": "pmkisan.gov.in",
            "category": "Income Support"
        },
        {
            "name": "PMFBY",
            "full_name": "Pradhan Mantri Fasal Bima Yojana",
            "benefit": "Crop insurance at low premium (2% for Kharif, 1.5% for Rabi)",
            "eligibility": "All farmers growing notified crops",
            "apply_at": "pmfby.gov.in",
            "category": "Insurance"
        },
        {
            "name": "KCC",
            "full_name": "Kisan Credit Card",
            "benefit": "Short-term crop loan at 4% interest (with subsidy)",
            "eligibility": "All farmers, sharecroppers, tenant farmers",
            "apply_at": "Nearest bank branch",
            "category": "Credit"
        },
        {
            "name": "e-NAM",
            "full_name": "National Agriculture Market",
            "benefit": "Online trading platform for better crop prices",
            "eligibility": "All farmers",
            "apply_at": "enam.gov.in",
            "category": "Market Access"
        },
        {
            "name": "PKVY",
            "full_name": "Paramparagat Krishi Vikas Yojana",
            "benefit": "₹50,000/hectare for organic farming cluster",
            "eligibility": "Farmer groups (minimum 50 farmers, 50 acres)",
            "apply_at": "State Agriculture Department",
            "category": "Organic Farming"
        },
    ]
    return {"success": True, "schemes": schemes, "total": len(schemes)}
