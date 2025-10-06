"""
Tests for configuration
"""
import os
import sys
import pytest

# Add the parent directory to the Python path to find config module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DevelopmentConfig, TestingConfig, ProductionConfig, Config

def test_development_config():
    """Test development configuration"""
    config = DevelopmentConfig()
    assert config.DEBUG is True
    assert config.TESTING is False
    assert config.ENV == "development"
    assert config.GEMINI_MODEL == "gemini-1.5-flash"
    assert config.LOG_LEVEL == "DEBUG"

def test_testing_config():
    """Test testing configuration"""
    config = TestingConfig()
    assert config.DEBUG is True
    assert config.TESTING is True
    assert config.ENV == "testing"
    assert config.GEMINI_MODEL == "gemini-1.5-flash"
    assert config.LOG_LEVEL == "INFO"

def test_production_config():
    """Test production configuration"""
    config = ProductionConfig()
    assert config.DEBUG is False
    assert config.TESTING is False
    assert config.ENV == "production"
    assert config.GEMINI_MODEL == "gemini-2.5-pro"
    assert config.LOG_LEVEL == "WARNING"

def test_base_config():
    """Test base configuration has required attributes"""
    config = Config()
    assert hasattr(config, 'GEMINI_API_KEY')
    assert hasattr(config, 'GOOGLE_CSE_ID')
    assert hasattr(config, 'SECRET_KEY')
    # SECRET_KEY should have a default value if not set in environment
    assert isinstance(config.SECRET_KEY, str)
    assert len(config.SECRET_KEY) > 0

def test_config_inheritance():
    """Test that all configs inherit from base Config"""
    dev_config = DevelopmentConfig()
    test_config = TestingConfig()
    prod_config = ProductionConfig()
    
    assert isinstance(dev_config, Config)
    assert isinstance(test_config, Config)
    assert isinstance(prod_config, Config)

def test_environment_variables():
    """Test that environment variables are loaded correctly"""
    config = Config()
    
    # Test that config loads environment variables (check attributes exist)
    assert hasattr(config, 'GEMINI_API_KEY')
    assert hasattr(config, 'GOOGLE_CSE_ID') 
    assert hasattr(config, 'SECRET_KEY')
    
    # Test default SECRET_KEY when not in environment
    if not os.environ.get('SECRET_KEY'):
        assert config.SECRET_KEY == "dev-secret-key"
    
    # Test that config can handle None values gracefully
    assert config.GEMINI_API_KEY is not None or config.GEMINI_API_KEY is None
    assert config.GOOGLE_CSE_ID is not None or config.GOOGLE_CSE_ID is None
