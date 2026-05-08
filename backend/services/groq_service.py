"""
KisanAI - Groq LLM Service
Handles all interactions with Groq API for agricultural advice
"""
from groq import Groq
from typing import Optional
from backend.config import get_settings

settings = get_settings()

# Agricultural AI System Prompt
SYSTEM_PROMPT = """You are KisanAI, an expert agricultural advisor and AI assistant designed specifically for Indian farmers. You are knowledgeable about:

1. **Crop Management**: Sowing seasons, crop rotation, intercropping, yield optimization for Indian crops (rice, wheat, cotton, sugarcane, pulses, vegetables, fruits)
2. **Soil Health**: Soil types (alluvial, black, red, laterite), soil testing, pH management, organic matter improvement
3. **Pest & Disease Control**: Identification and treatment of common pests and diseases in Indian agriculture, both organic and chemical solutions
4. **Fertilizer Usage**: NPK fertilizers, micronutrients, bio-fertilizers, timing and dosage recommendations
5. **Weather & Climate**: Weather-based farming advisory, drought/flood management, climate-smart agriculture
6. **Water Management**: Irrigation techniques (drip, sprinkler, flood), water conservation
7. **Government Schemes**: PM-KISAN, Kisan Credit Card, crop insurance (PMFBY), MSP, e-NAM
8. **Market Information**: Mandi prices, best time to sell, storage advice
9. **Organic Farming**: Natural fertilizers, composting, biodynamic farming

**Guidelines:**
- Always respond in a friendly, empathetic manner appropriate for farmers
- Provide practical, actionable advice that can be implemented immediately
- Use simple language; avoid overly technical jargon
- When asked in Hindi, respond in Hindi (Devanagari script)
- Always consider the Indian farming context (monsoon season, local crops, soil types)
- For serious issues, recommend consulting local agricultural extension officers (KVK - Krishi Vigyan Kendra)
- Be encouraging and supportive, as farming is challenging work

You are KisanAI — the trusted digital companion for every Indian farmer. 🌾"""


TOPIC_PROMPTS = {
    "crop": """Focus specifically on crop management advice. Consider: crop variety selection, sowing time, spacing, irrigation schedule, harvesting tips. For Indian context, mention Kharif, Rabi, and Zaid seasons.""",
    "soil": """Focus on soil health analysis and improvement. Consider: soil type, pH correction, organic matter addition, micro-nutrient deficiencies, green manuring, vermicomposting.""",
    "pest": """Focus on pest and disease management. Consider: Integrated Pest Management (IPM), organic alternatives, chemical pesticides (with safety warnings), spray timing, preventive measures.""",
    "fertilizer": """Focus on fertilizer recommendations. Consider: soil test reports, NPK ratios, biofertilizers (Rhizobium, Azotobacter, PSB), slow-release fertilizers, fertigation techniques.""",
    "weather": """Focus on weather-based farming advice. Consider: current weather impact on crops, precautions for upcoming weather, irrigation adjustments, post-rain/drought recovery.""",
    "market": """Focus on market and selling advice. Consider: best selling time, mandi prices, government MSP rates, storage methods, value addition, FPO benefits.""",
}


def build_messages(
    user_message: str,
    conversation_history: list[dict],
    topic: Optional[str] = None,
) -> list[dict]:
    """Build the messages array for Groq API call."""
    system_content = SYSTEM_PROMPT
    if topic and topic in TOPIC_PROMPTS:
        system_content += f"\n\n**Current Focus Area:**\n{TOPIC_PROMPTS[topic]}"

    messages = [{"role": "system", "content": system_content}]

    # Add conversation history (last 10 messages)
    if conversation_history:
        messages.extend(conversation_history[-10:])

    # Add current message
    messages.append({"role": "user", "content": user_message})
    return messages


def get_groq_response(
    user_message: str,
    conversation_history: list[dict] = None,
    topic: Optional[str] = None,
    language: str = "english",
) -> dict:
    """
    Get response from Groq LLM API.
    Returns dict with response text and token usage.
    """
    client = Groq(api_key=settings.groq_api_key)

    if conversation_history is None:
        conversation_history = []

    # Add language instruction if Hindi
    if language == "hindi":
        user_message = f"[कृपया हिंदी में उत्तर दें]\n{user_message}"

    messages = build_messages(user_message, conversation_history, topic)

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            top_p=0.9,
        )

        assistant_message = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return {
            "success": True,
            "message": assistant_message,
            "usage": usage,
            "model": settings.groq_model,
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Sorry, I encountered an error: {str(e)}. Please try again.",
            "usage": {},
            "model": settings.groq_model,
        }


def generate_disease_advice(disease_name: str, confidence: float, crop_name: str = "") -> str:
    """Generate treatment advice for a detected plant disease using Groq."""
    client = Groq(api_key=settings.groq_api_key)

    prompt = f"""A plant disease has been detected with AI image analysis:

Disease Detected: {disease_name}
Confidence: {confidence:.1%}
Crop: {crop_name if crop_name else "Unknown"}

Please provide:
1. Brief description of this disease
2. Symptoms to look for
3. Immediate treatment steps (both organic and chemical options)
4. Prevention measures for the future
5. When to consult an agricultural expert

Keep the response practical and suitable for Indian farmers."""

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=800,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Disease: {disease_name}. Please consult your local agricultural officer (KVK) for treatment advice."


def generate_soil_advice(soil_data: dict) -> str:
    """Generate soil health advice based on soil test data."""
    client = Groq(api_key=settings.groq_api_key)

    prompt = f"""Based on the soil test results below, provide comprehensive soil health advice:

Soil Data:
- Nitrogen (N): {soil_data.get('nitrogen', 'N/A')} kg/ha
- Phosphorus (P): {soil_data.get('phosphorus', 'N/A')} kg/ha
- Potassium (K): {soil_data.get('potassium', 'N/A')} kg/ha
- pH Level: {soil_data.get('ph', 'N/A')}
- Organic Matter: {soil_data.get('organic_matter', 'N/A')}%
- Soil Type: {soil_data.get('soil_type', 'Unknown')}
- Rainfall: {soil_data.get('rainfall', 'N/A')} mm/year
- Temperature: {soil_data.get('temperature', 'N/A')}°C

Please provide:
1. Overall soil health assessment (score out of 10)
2. Key deficiencies or imbalances
3. Recommended amendments with quantities
4. Best crops to grow in this soil
5. Long-term soil improvement plan"""

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=1000,
        )
        return response.choices[0].message.content
    except Exception as e:
        return "Unable to generate soil advice. Please consult your local KVK."
