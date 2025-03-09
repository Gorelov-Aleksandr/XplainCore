"""
Configuration module for the XAI service.

This module provides settings for the application,
using environment variables and a .env file.
"""
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application configuration settings.
    Values can be overridden by environment variables.
    """
    # Basic service configuration
    service_name: str = os.getenv("SERVICE_NAME", "xai-service")
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # Cache settings
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    cache_ttl: int = int(os.getenv("CACHE_TTL", "300"))  # seconds
    
    # Observability settings
    jaeger_host: str = os.getenv("JAEGER_HOST", "jaeger")
    jaeger_port: int = int(os.getenv("JAEGER_PORT", "6831"))
    
    # Authentication settings
    auth_enabled: bool = os.getenv("AUTH_ENABLED", "true").lower() == "true"
    dev_api_key: str = os.getenv("DEV_API_KEY", "XAI-dev-key-2023")
    
    # Yandex OAuth settings
    yandex_oauth_url: str = os.getenv("YANDEX_OAUTH_URL", "https://oauth.yandex.ru/authorize")
    yandex_token_url: str = os.getenv("YANDEX_TOKEN_URL", "https://oauth.yandex.ru/token")
    yandex_client_id: str = os.getenv("YANDEX_CLIENT_ID", "")
    yandex_client_secret: str = os.getenv("YANDEX_CLIENT_SECRET", "")
    
    # Monitoring settings
    sentry_dsn: str = os.getenv("SENTRY_DSN", "")
    sentry_environment: str = os.getenv("SENTRY_ENVIRONMENT", "production")
    
    # Rate limiting settings
    rate_limit: int = int(os.getenv("RATE_LIMIT", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    
    # Model settings
    model_path: str = os.getenv("MODEL_PATH", "./models/xai_model.pkl")
    model_version: str = os.getenv("MODEL_VERSION", "1.0.0")
    
    # Explanation settings
    max_features: int = int(os.getenv("MAX_FEATURES", "10"))
    plot_height: int = int(os.getenv("PLOT_HEIGHT", "400"))
    plot_width: int = int(os.getenv("PLOT_WIDTH", "600"))
    
    # Database settings
    database_url: str = os.getenv("DATABASE_URL", "")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Create settings instance
settings = Settings()