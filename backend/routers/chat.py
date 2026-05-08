"""
KisanAI - Chat Router
Handles AI chat with Groq LLM for agricultural advice
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from backend.services.groq_service import get_groq_response

router = APIRouter(prefix="/api/chat", tags=["Chat"])


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    topic: Optional[str] = None  # crop, soil, pest, fertilizer, weather, market
    language: str = "english"


class ChatResponse(BaseModel):
    success: bool
    message: str
    usage: dict = {}
    model: str = ""


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — sends user message to Groq LLM
    and returns agricultural advice.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Convert history to plain dicts
    history = [{"role": m.role, "content": m.content} for m in request.history]

    result = get_groq_response(
        user_message=request.message,
        conversation_history=history,
        topic=request.topic,
        language=request.language,
    )

    return ChatResponse(**result)


# Quick-action shortcut topics
QUICK_TOPICS = {
    "crop": "What crops should I grow this season? How do I improve my yield?",
    "soil": "How do I improve my soil health? What amendments should I add?",
    "pest": "How do I identify and control common pests and diseases in my crops?",
    "weather": "How does weather affect my farming? What precautions should I take?",
    "fertilizer": "What fertilizers should I use? How much and when should I apply?",
    "market": "What are the best prices for my crops? How can I get better market rates?",
    "water": "How can I optimize irrigation and water usage for my crops?",
    "organic": "How can I transition to organic farming? What are bio-fertilizers?",
    "government": "What government schemes are available for farmers like PM-KISAN, crop insurance?",
}


class QuickActionRequest(BaseModel):
    topic: str
    language: str = "english"
    history: list[ChatMessage] = []


@router.post("/quick-action")
async def quick_action(request: QuickActionRequest):
    """Handle quick-action button clicks from the UI."""
    if request.topic not in QUICK_TOPICS:
        raise HTTPException(status_code=400, detail=f"Unknown topic: {request.topic}")

    message = QUICK_TOPICS[request.topic]
    history = [{"role": m.role, "content": m.content} for m in request.history]

    result = get_groq_response(
        user_message=message,
        conversation_history=history,
        topic=request.topic,
        language=request.language,
    )

    return ChatResponse(**result)
