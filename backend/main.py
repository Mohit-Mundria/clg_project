"""
KisanAI - FastAPI Main Application
Production-ready agricultural AI assistant backend
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import get_settings
from backend.routers import chat, crop, fertilizer, disease, weather, market

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("kisanai")

settings = get_settings()

# ── Directory Paths ───────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
MODEL_DIR = BASE_DIR / "backend" / "models"


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("=" * 60)
    logger.info("🌾 KisanAI - Starting Agricultural AI Assistant")
    logger.info(f"   Version    : {settings.app_version}")
    logger.info(f"   Environment: {settings.app_env}")
    logger.info(f"   Host       : {settings.app_host}:{settings.app_port}")
    logger.info("=" * 60)

    # Ensure model directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Check if ML models exist, if not prompt training
    crop_model = MODEL_DIR / "crop_model.pkl"
    fertilizer_model = MODEL_DIR / "fertilizer_model.pkl"

    if not crop_model.exists() or not fertilizer_model.exists():
        logger.warning("⚠️  ML models not found!")
        logger.warning("   Run: python ml_training/train_models.py")
        logger.warning("   Crop & Fertilizer features will be unavailable until models are trained.")
    else:
        logger.info("✅ ML models found and ready.")

    if settings.groq_api_key:
        logger.info("✅ Groq API key configured.")
    else:
        logger.error("❌ GROQ_API_KEY is missing! Chat will not work.")

    if settings.openweather_api_key:
        logger.info("✅ OpenWeatherMap API key configured.")
    else:
        logger.warning("⚠️  OPENWEATHER_API_KEY not set — using mock weather data.")

    logger.info("🚀 KisanAI is ready at http://localhost:8000")
    yield

    logger.info("👋 KisanAI shutting down...")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="KisanAI",
    description="AI-powered Agricultural Advisory Assistant for Indian Farmers",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routers ───────────────────────────────────────────────────────────────
app.include_router(chat.router)
app.include_router(crop.router)
app.include_router(fertilizer.router)
app.include_router(disease.router)
app.include_router(weather.router)
app.include_router(market.router)


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for AWS load balancer / monitoring."""
    model_dir = MODEL_DIR
    return {
        "status": "healthy",
        "service": "KisanAI",
        "version": settings.app_version,
        "environment": settings.app_env,
        "models": {
            "crop_model": (model_dir / "crop_model.pkl").exists(),
            "fertilizer_model": (model_dir / "fertilizer_model.pkl").exists(),
        },
        "api_keys": {
            "groq": bool(settings.groq_api_key),
            "openweather": bool(settings.openweather_api_key),
        }
    }


# ── Static Files & Frontend ───────────────────────────────────────────────────
static_dir = FRONTEND_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the main landing page."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "KisanAI API is running. Frontend not found."})


@app.get("/chat", include_in_schema=False)
async def serve_chat():
    """Serve the chat interface."""
    chat_file = FRONTEND_DIR / "chat.html"
    if chat_file.exists():
        return FileResponse(str(chat_file))
    return JSONResponse({"error": "Chat page not found"}, status_code=404)


@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    """Serve the farmer dashboard."""
    dashboard_file = FRONTEND_DIR / "dashboard.html"
    if dashboard_file.exists():
        return FileResponse(str(dashboard_file))
    return JSONResponse({"error": "Dashboard page not found"}, status_code=404)


# ── Exception Handlers ────────────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "Endpoint not found", "path": request.url.path}, status_code=404)
    # Try serving index for SPA routing
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"error": "Not found"}, status_code=404)


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        {"error": "Internal server error. Please try again.", "detail": str(exc)},
        status_code=500
    )


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=not settings.is_production,
        log_level="info",
        workers=1,
    )
