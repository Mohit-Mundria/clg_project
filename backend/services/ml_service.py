"""
KisanAI - ML Model Service
Handles Crop Recommendation, Fertilizer Recommendation, and Plant Disease Detection
"""
import os
import joblib
import numpy as np
import logging
from pathlib import Path
from typing import Optional
from PIL import Image
import io

logger = logging.getLogger(__name__)

# ── Model Paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

CROP_MODEL_PATH = MODEL_DIR / "crop_model.pkl"
FERTILIZER_MODEL_PATH = MODEL_DIR / "fertilizer_model.pkl"
CROP_ENCODER_PATH = MODEL_DIR / "crop_label_encoder.pkl"
FERTILIZER_ENCODER_PATH = MODEL_DIR / "fertilizer_label_encoder.pkl"
CROP_SCALER_PATH = MODEL_DIR / "crop_scaler.pkl"
FERTILIZER_SCALER_PATH = MODEL_DIR / "fertilizer_scaler.pkl"

# ── Global Model Cache ────────────────────────────────────────────────────────
_crop_model = None
_fertilizer_model = None
_crop_encoder = None
_fertilizer_encoder = None
_crop_scaler = None
_fertilizer_scaler = None
_disease_pipeline = None


# ── Crop Recommendation ───────────────────────────────────────────────────────
def load_crop_model():
    global _crop_model, _crop_encoder, _crop_scaler
    if _crop_model is None:
        if not CROP_MODEL_PATH.exists():
            logger.warning("Crop model not found. Please run ml_training/train_models.py first.")
            return None, None, None
        _crop_model = joblib.load(CROP_MODEL_PATH)
        _crop_encoder = joblib.load(CROP_ENCODER_PATH)
        _crop_scaler = joblib.load(CROP_SCALER_PATH)
    return _crop_model, _crop_encoder, _crop_scaler


def predict_crop(nitrogen: float, phosphorus: float, potassium: float,
                 temperature: float, humidity: float, ph: float, rainfall: float) -> dict:
    """
    Predict the best crop to grow based on soil and weather parameters.
    Uses Random Forest Classifier.
    """
    model, encoder, scaler = load_crop_model()

    if model is None:
        return {
            "success": False,
            "error": "Crop recommendation model not loaded. Please run ml_training/train_models.py"
        }

    try:
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        features_scaled = scaler.transform(features)

        # Predict with probabilities
        pred_encoded = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # Top 3 crops
        top3_indices = np.argsort(probabilities)[::-1][:3]
        top3_crops = [
            {
                "crop": encoder.inverse_transform([idx])[0].title(),
                "confidence": float(probabilities[idx]),
                "confidence_pct": f"{probabilities[idx]*100:.1f}%"
            }
            for idx in top3_indices
        ]

        recommended_crop = encoder.inverse_transform([pred_encoded])[0].title()
        confidence = float(max(probabilities))

        # Crop information dictionary
        crop_info = {
            "Rice": {"season": "Kharif (Jun-Nov)", "water": "High", "soil": "Clay/Loam"},
            "Wheat": {"season": "Rabi (Nov-Apr)", "water": "Medium", "soil": "Loam/Clay Loam"},
            "Maize": {"season": "Kharif & Rabi", "water": "Medium", "soil": "Well-drained Loam"},
            "Chickpea": {"season": "Rabi (Oct-Feb)", "water": "Low", "soil": "Sandy Loam"},
            "Kidneybeans": {"season": "Kharif (Jun-Sep)", "water": "Medium", "soil": "Loam"},
            "Pigeonpeas": {"season": "Kharif (Jun-Oct)", "water": "Low", "soil": "Sandy Loam"},
            "Mothbeans": {"season": "Kharif (Jul-Sep)", "water": "Low", "soil": "Sandy"},
            "Mungbean": {"season": "Kharif & Zaid", "water": "Low-Medium", "soil": "Loam"},
            "Blackgram": {"season": "Kharif (Jun-Sep)", "water": "Medium", "soil": "Loam"},
            "Lentil": {"season": "Rabi (Oct-Feb)", "water": "Low", "soil": "Loam"},
            "Pomegranate": {"season": "Perennial", "water": "Low", "soil": "Well-drained Loam"},
            "Banana": {"season": "Perennial", "water": "High", "soil": "Rich Loam"},
            "Mango": {"season": "Perennial (Kharif fruit)", "water": "Medium", "soil": "Alluvial"},
            "Grapes": {"season": "Perennial (winter harvest)", "water": "Medium", "soil": "Sandy Loam"},
            "Watermelon": {"season": "Zaid (Feb-Jun)", "water": "High", "soil": "Sandy Loam"},
            "Muskmelon": {"season": "Zaid (Feb-May)", "water": "Medium", "soil": "Sandy Loam"},
            "Apple": {"season": "Rabi (harvest Oct-Jan)", "water": "Medium", "soil": "Loam"},
            "Orange": {"season": "Perennial", "water": "Medium", "soil": "Sandy Loam"},
            "Papaya": {"season": "Perennial", "water": "Medium-High", "soil": "Loam"},
            "Coconut": {"season": "Perennial", "water": "High", "soil": "Sandy Loam"},
            "Cotton": {"season": "Kharif (Apr-Dec)", "water": "Medium", "soil": "Black Cotton"},
            "Jute": {"season": "Kharif (Mar-Jun)", "water": "High", "soil": "Alluvial"},
            "Coffee": {"season": "Perennial", "water": "Medium-High", "soil": "Rich Loam"},
        }

        info = crop_info.get(recommended_crop, {"season": "Varies", "water": "Medium", "soil": "Loam"})

        return {
            "success": True,
            "recommended_crop": recommended_crop,
            "confidence": confidence,
            "confidence_pct": f"{confidence*100:.1f}%",
            "top_3_crops": top3_crops,
            "crop_info": info,
            "inputs": {
                "nitrogen": nitrogen, "phosphorus": phosphorus, "potassium": potassium,
                "temperature": temperature, "humidity": humidity, "ph": ph, "rainfall": rainfall
            }
        }

    except Exception as e:
        logger.error(f"Crop prediction error: {e}")
        return {"success": False, "error": str(e)}


# ── Fertilizer Recommendation ─────────────────────────────────────────────────
def load_fertilizer_model():
    global _fertilizer_model, _fertilizer_encoder, _fertilizer_scaler
    if _fertilizer_model is None:
        if not FERTILIZER_MODEL_PATH.exists():
            logger.warning("Fertilizer model not found. Please run ml_training/train_models.py first.")
            return None, None, None
        _fertilizer_model = joblib.load(FERTILIZER_MODEL_PATH)
        _fertilizer_encoder = joblib.load(FERTILIZER_ENCODER_PATH)
        _fertilizer_scaler = joblib.load(FERTILIZER_SCALER_PATH)
    return _fertilizer_model, _fertilizer_encoder, _fertilizer_scaler


SOIL_TYPE_MAP = {
    "sandy": 0, "loamy": 1, "black": 2, "red": 3,
    "clayey": 4, "alluvial": 5, "laterite": 6
}

CROP_TYPE_MAP = {
    "rice": 0, "wheat": 1, "maize": 2, "sugarcane": 3, "cotton": 4,
    "pulses": 5, "vegetables": 6, "fruits": 7, "oilseeds": 8, "barley": 9
}


def predict_fertilizer(soil_type: str, crop_type: str, nitrogen: float,
                       phosphorus: float, potassium: float,
                       temperature: float, humidity: float, moisture: float) -> dict:
    """
    Predict the best fertilizer based on soil and crop conditions.
    Uses Gradient Boosting Classifier.
    """
    model, encoder, scaler = load_fertilizer_model()

    if model is None:
        return {
            "success": False,
            "error": "Fertilizer model not loaded. Please run ml_training/train_models.py"
        }

    try:
        soil_encoded = SOIL_TYPE_MAP.get(soil_type.lower(), 1)
        crop_encoded = CROP_TYPE_MAP.get(crop_type.lower(), 0)

        features = np.array([[temperature, humidity, moisture, soil_encoded,
                               crop_encoded, nitrogen, potassium, phosphorus]])
        features_scaled = scaler.transform(features)

        pred_encoded = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = float(max(probabilities))

        fertilizer_name = encoder.inverse_transform([pred_encoded])[0]

        # Fertilizer advice dictionary
        fertilizer_info = {
            "Urea": {
                "npk": "46-0-0", "type": "Nitrogen fertilizer",
                "dosage": "100-150 kg/acre", "when": "At sowing and top dressing",
                "tip": "Split application recommended. Avoid before heavy rain."
            },
            "DAP": {
                "npk": "18-46-0", "type": "Phosphatic fertilizer",
                "dosage": "50-100 kg/acre", "when": "At sowing (basal dose)",
                "tip": "Best applied as basal dose before sowing."
            },
            "MOP": {
                "npk": "0-0-60", "type": "Potassic fertilizer",
                "dosage": "25-50 kg/acre", "when": "At sowing or splitting",
                "tip": "Good for root development and disease resistance."
            },
            "10-26-26": {
                "npk": "10-26-26", "type": "Complex fertilizer",
                "dosage": "50-75 kg/acre", "when": "At sowing",
                "tip": "Balanced NPK for good vegetative and root growth."
            },
            "17-17-17": {
                "npk": "17-17-17", "type": "Balanced complex fertilizer",
                "dosage": "50-100 kg/acre", "when": "At sowing and vegetative stage",
                "tip": "Well-balanced formula, good for most crops."
            },
            "20-20": {
                "npk": "20-20-0", "type": "NP fertilizer",
                "dosage": "75-100 kg/acre", "when": "At sowing",
                "tip": "Good for leguminous crops deficient in P."
            },
            "28-28": {
                "npk": "28-28-0", "type": "High NP fertilizer",
                "dosage": "50-75 kg/acre", "when": "At sowing",
                "tip": "Concentrated formula, reduces application costs."
            },
            "14-35-14": {
                "npk": "14-35-14", "type": "High phosphorus complex",
                "dosage": "50-75 kg/acre", "when": "Basal dose at sowing",
                "tip": "Ideal for phosphorus-deficient soils."
            },
        }

        info = fertilizer_info.get(fertilizer_name, {
            "npk": "Varies", "type": "Mixed fertilizer",
            "dosage": "As recommended", "when": "At sowing",
            "tip": "Consult local agricultural officer for specific dosage."
        })

        # NPK status interpretation
        npk_status = {
            "nitrogen": "Low" if nitrogen < 50 else "Medium" if nitrogen < 100 else "High",
            "phosphorus": "Low" if phosphorus < 25 else "Medium" if phosphorus < 50 else "High",
            "potassium": "Low" if potassium < 50 else "Medium" if potassium < 100 else "High",
        }

        return {
            "success": True,
            "recommended_fertilizer": fertilizer_name,
            "confidence": confidence,
            "confidence_pct": f"{confidence*100:.1f}%",
            "fertilizer_info": info,
            "npk_status": npk_status,
            "inputs": {
                "soil_type": soil_type, "crop_type": crop_type,
                "nitrogen": nitrogen, "phosphorus": phosphorus, "potassium": potassium,
                "temperature": temperature, "humidity": humidity, "moisture": moisture
            }
        }

    except Exception as e:
        logger.error(f"Fertilizer prediction error: {e}")
        return {"success": False, "error": str(e)}


# ── Plant Disease Detection ───────────────────────────────────────────────────
def load_disease_model():
    """Load plant disease detection model from HuggingFace."""
    global _disease_pipeline
    if _disease_pipeline is None:
        try:
            from transformers import pipeline
            logger.info("Loading plant disease detection model from HuggingFace...")
            _disease_pipeline = pipeline(
                "image-classification",
                model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
                top_k=3
            )
            logger.info("Plant disease model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load disease model: {e}")
            return None
    return _disease_pipeline


def detect_disease(image_bytes: bytes) -> dict:
    """
    Detect plant disease from an uploaded leaf image.
    Uses a pre-trained MobileNetV2 model from HuggingFace.
    """
    pipe = load_disease_model()

    if pipe is None:
        return {
            "success": False,
            "error": "Disease detection model could not be loaded. Check your internet connection."
        }

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Run inference
        results = pipe(image)

        if not results:
            return {"success": False, "error": "Could not analyze the image."}

        top_result = results[0]
        label = top_result["label"]
        confidence = top_result["score"]

        # Parse label (format: "Crop___Disease_Name" or "Crop___healthy")
        parts = label.replace("___", " - ").replace("_", " ").split(" - ")
        crop_name = parts[0].strip().title() if len(parts) > 1 else "Plant"
        disease_name = parts[1].strip().title() if len(parts) > 1 else label.replace("_", " ").title()

        is_healthy = "healthy" in label.lower()

        all_results = [
            {
                "label": r["label"].replace("___", " - ").replace("_", " ").title(),
                "confidence": r["score"],
                "confidence_pct": f"{r['score']*100:.1f}%"
            }
            for r in results
        ]

        return {
            "success": True,
            "crop_name": crop_name,
            "disease_name": disease_name,
            "is_healthy": is_healthy,
            "confidence": confidence,
            "confidence_pct": f"{confidence*100:.1f}%",
            "top_results": all_results,
            "raw_label": label
        }

    except Exception as e:
        logger.error(f"Disease detection error: {e}")
        return {"success": False, "error": f"Image analysis failed: {str(e)}"}
