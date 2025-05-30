import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database Configuration
    NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
    
    # OpenAI Configuration (or other LLM provider)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # App Configuration
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"