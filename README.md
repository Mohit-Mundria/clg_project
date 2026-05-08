Sabse phele iss repository ko clone kro apne laptop pe.
donwload python as all the things are written using python

open the folder in the VS code in which you clone the repository.


open the terminal(CMD) and run this command: "pip install -r requirements.txt"
 made a ".env"  file in your project folder,  and keep all api secret in that, without this the project will not work. The structure of .env file:
 # ===========================
# KisanAI Environment Variables
# Copy this file to .env and fill in your keys
# ===========================

# Groq API Key (Required)
# Get free key at: https://console.groq.com
GROQ_API_KEY='YOUR GROQ API KEY'

# Groq Model (default: llama3-70b-8192)
GROQ_MODEL=llama-3.3-70b-versatile

# WeatherAPI Key (Optional - for weather features)
# Get free key at: https://www.weatherapi.com/my/
WEATHER_API_KEY=YOUR WEATHER API KEY HERE

# App Settings
APP_HOST=0.0.0.0
APP_PORT=8000
APP_ENV=development

# Default Location (for weather)
DEFAULT_CITY=New Delhi
DEFAULT_STATE=Delhi
DEFAULT_COUNTRY=India

# Secret Key (for sessions - generate a random string)
SECRET_KEY=kisanai-secret-key-change-in-production-2024

# CORS Origins (comma separated)
CORS_ORIGINS=http://localhost:8000,http://localhost:3000







then run this command: ".\start.bat"
this will give a url of localhost copy that in browser.



in the first run it will take some time os keep paitent.
