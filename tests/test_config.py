"""
Tests for configuration
"""
import pytest
from config import DevelopmentConfig, TestingConfig, ProductionConfig

def test_development_config():
    """Test development configuration"""
    config = DevelopmentConfig()
    assert config.DEBUG == True
    assert config.TESTING == False
    assert config.ENV == "development"
    assert config.GEMINI_MODEL == "gemini-1.5-flash"
    assert config.LOG_LEVEL == "DEBUG"

def test_testing_config():
    """Test testing configuration"""
    config = TestingConfig()
    assert config.DEBUG == True
    assert config.TESTING == True
    assert config.ENV == "testing"
    assert config.GEMINI_MODEL == "gemini-1.5-flash"
    assert config.LOG_LEVEL == "INFO"

def test_production_config():
    """Test production configuration"""
    config = ProductionConfig()
    assert config.DEBUG == False
    assert config.TESTING == False
    assert config.ENV == "production"
    assert config.GEMINI_MODEL == "gemini-2.5-pro"
    assert config.LOG_LEVEL == "WARNING"

def test_base_config():
    """Test base configuration has required attributes"""
    from config import Config
    config = Config()
    assert hasattr(config, 'GEMINI_API_KEY')
    assert hasattr(config, 'GOOGLE_CSE_ID')
    assert hasattr(config, 'SECRET_KEY')
    assert config.SECRET_KEY == "dev-secret-key"  # default value
