"""
KisanAI - Disease Detection Router
CNN-based plant disease detection from leaf images
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional
from backend.services.ml_service import detect_disease
from backend.services.groq_service import generate_disease_advice

router = APIRouter(prefix="/api/disease", tags=["Disease Detection"])

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post("/detect")
async def detect_plant_disease(
    file: UploadFile = File(..., description="Leaf image (JPG/PNG, max 10MB)"),
    crop_name: Optional[str] = Form(default="", description="Name of the crop (optional)")
):
    """
    Detect plant diseases from a leaf image.
    Uses MobileNetV2 CNN model trained on PlantVillage dataset (38 disease classes).
    Provides diagnosis + AI-generated treatment advice via Groq LLM.
    """
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: JPG, PNG, WebP"
        )

    # Read and validate size
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

    if len(image_bytes) < 1000:
        raise HTTPException(status_code=400, detail="File appears to be corrupted or too small.")

    # Run disease detection ML model
    detection_result = detect_disease(image_bytes)

    if not detection_result.get("success"):
        raise HTTPException(status_code=500, detail=detection_result.get("error", "Detection failed"))

    # Generate LLM-based treatment advice
    disease_name = detection_result.get("disease_name", "Unknown")
    confidence = detection_result.get("confidence", 0.0)
    detected_crop = detection_result.get("crop_name", crop_name or "")
    is_healthy = detection_result.get("is_healthy", False)

    if is_healthy:
        ai_advice = (
            f"Great news! Your {detected_crop or 'plant'} leaf appears **healthy** "
            f"(confidence: {confidence*100:.1f}%). Continue your current care routine. "
            f"Regularly inspect leaves for any early signs of stress or disease, "
            f"and maintain proper irrigation and nutrition schedules."
        )
    else:
        ai_advice = generate_disease_advice(
            disease_name=disease_name,
            confidence=confidence,
            crop_name=detected_crop
        )

    return {
        "success": True,
        "detection": detection_result,
        "ai_treatment_advice": ai_advice,
        "filename": file.filename,
        "file_size_kb": round(len(image_bytes) / 1024, 1),
    }


@router.get("/supported-crops")
async def get_supported_crops():
    """List all crops supported by the disease detection model."""
    crops = [
        "Apple", "Blueberry", "Cherry", "Corn (Maize)",
        "Grape", "Orange", "Peach", "Bell Pepper",
        "Potato", "Raspberry", "Soybean", "Squash",
        "Strawberry", "Tomato"
    ]
    return {
        "success": True,
        "supported_crops": crops,
        "total_classes": 38,
        "model": "MobileNetV2 — PlantVillage Dataset",
        "note": "Upload a clear, well-lit photo of the plant leaf for best results"
    }
