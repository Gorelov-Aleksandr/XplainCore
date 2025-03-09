import requests
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import OAuth2AuthorizationCodeBearer, APIKeyHeader
from typing import Optional, Union, Dict, Any
from loguru import logger

from .config import settings

# Setup OAuth2 scheme for Yandex
if settings.auth_enabled and settings.yandex_client_id and settings.yandex_client_secret:
    # Use OAuth2 if Yandex credentials are provided
    oauth2_scheme = OAuth2AuthorizationCodeBearer(
        authorizationUrl=settings.yandex_oauth_url,
        tokenUrl=settings.yandex_token_url,
        scopes={}
    )
    logger.info("Yandex OAuth2 authentication configured")
else:
    # If no Yandex credentials, use API key auth for simplicity
    oauth2_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
    logger.info("Using API key authentication fallback")

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    x_api_key: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    Get the current user's information.
    
    In production: Uses Yandex OAuth token.
    In development: Accepts X-API-Key header for testing.
    
    Args:
        token: OAuth2 token or API key
        x_api_key: Optional API key header for easier testing
        
    Returns:
        dict: User information
        
    Raises:
        HTTPException: If authentication fails
    """
    # Allow disabling auth for development
    if not settings.auth_enabled:
        logger.warning("Authentication disabled! Using test user.")
        return {
            "id": "test-user-id",
            "login": "test-user",
            "name": "Test User",
            "is_test_user": True
        }
    
    # Check for API key first (for development/testing)
    api_key = token or x_api_key
    if api_key and api_key == settings.dev_api_key:
        logger.info("Using development API key authentication")
        return {
            "id": "dev-user-id",
            "login": "dev-user",
            "name": "Development User",
            "is_dev_user": True
        }
    
    # Fall back to Yandex OAuth if API key not provided or invalid
    if settings.yandex_client_id and settings.yandex_client_secret and token:
        try:
            user_info_endpoint = "https://login.yandex.ru/info"
            headers = {"Authorization": f"OAuth {token}"}
            
            response = requests.get(user_info_endpoint, headers=headers)
            response.raise_for_status()
            
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Yandex authentication error: {str(e)}")
    
    # If we got here, authentication failed
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
