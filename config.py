"""
Environment Configuration for Multilingual Voice Chatbot
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
    
class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    TESTING = False
    ENV = "development"
    GEMINI_MODEL = "gemini-1.5-flash"
    LOG_LEVEL = "DEBUG"
    
class TestingConfig(Config):
    """Testing environment configuration"""
    DEBUG = True
    TESTING = True
    ENV = "testing"
    GEMINI_MODEL = "gemini-1.5-flash"
    LOG_LEVEL = "INFO"
    
class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False
    ENV = "production"
    GEMINI_MODEL = "gemini-2.5-pro"
    LOG_LEVEL = "WARNING"

config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
