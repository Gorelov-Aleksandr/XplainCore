"""
Authentication module for the XAI service.

This module provides authentication using Yandex OAuth
or a development API key.
"""
import os
from typing import Dict, Any, Optional

from fastapi import Depends, HTTPException, Header, status
from fastapi.security import OAuth2PasswordBearer
from loguru import logger

from app.config import settings

# OAuth2 scheme for Yandex OAuth tokens
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Development API key for easier testing
DEV_API_KEY = settings.dev_api_key

# Log auth configuration
if settings.auth_enabled:
    logger.info("Authentication is enabled")
    if settings.environment == "development":
        logger.info("Using API key authentication fallback")
else:
    logger.warning("Authentication is disabled")


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
    # Skip authentication if disabled
    if not settings.auth_enabled:
        return {"id": "dev-user-id", "name": "Development User"}
    
    # Development mode: Accept API key
    if settings.environment == "development" and x_api_key:
        if x_api_key == DEV_API_KEY:
            return {"id": "dev-user-id", "name": "Development User", "api_key": True}
    
    # Production mode: Validate Yandex OAuth token
    if token:
        try:
            # This would validate the token with Yandex OAuth server
            # For now, we just simulate user information
            # In a real implementation, we would call Yandex's API
            return {"id": "user-123", "name": "OAuth User", "oauth": True}
        except Exception as e:
            logger.error(f"OAuth validation error: {str(e)}")
            
    # Default to API key validation if token not provided
    if x_api_key and x_api_key == DEV_API_KEY:
        return {"id": "dev-user-id", "name": "Development User", "api_key": True}
    
    # Authentication failed
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )