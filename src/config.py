import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# CORS origins
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
