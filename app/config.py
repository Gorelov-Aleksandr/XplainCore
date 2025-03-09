import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Application configuration settings.
    Values can be overridden by environment variables.
    """
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    jaeger_host: str = os.getenv("JAEGER_HOST", "jaeger")
    jaeger_port: int = int(os.getenv("JAEGER_PORT", "6831"))
    
    # Yandex OAuth settings
    yandex_oauth_url: str = os.getenv("YANDEX_OAUTH_URL", "https://oauth.yandex.ru/authorize")
    yandex_token_url: str = os.getenv("YANDEX_TOKEN_URL", "https://oauth.yandex.ru/token")
    yandex_client_id: str = os.getenv("YANDEX_CLIENT_ID", "")
    yandex_client_secret: str = os.getenv("YANDEX_CLIENT_SECRET", "")
    
    # Sentry settings
    sentry_dsn: str = os.getenv("SENTRY_DSN", "")
    sentry_environment: str = os.getenv("SENTRY_ENVIRONMENT", "production")
    
    # Rate limiting
    rate_limit: int = int(os.getenv("RATE_LIMIT", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds
    
    # Cache TTL
    cache_ttl: int = int(os.getenv("CACHE_TTL", "300"))  # seconds

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
