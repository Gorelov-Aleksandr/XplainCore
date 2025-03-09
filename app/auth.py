import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer

from .config import settings

# Setup OAuth2 scheme for Yandex
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=settings.yandex_oauth_url,
    tokenUrl=settings.yandex_token_url,
    scopes={}
)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """
    Get the current user's information from Yandex using OAuth token.
    
    Args:
        token: OAuth2 token received from Yandex
        
    Returns:
        dict: User information from Yandex
        
    Raises:
        HTTPException: If the token is invalid or expired
    """
    user_info_endpoint = "https://login.yandex.ru/info"
    headers = {"Authorization": f"OAuth {token}"}
    
    try:
        response = requests.get(user_info_endpoint, headers=headers)
        response.raise_for_status()
        
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail=f"Invalid credentials: {str(e)}"
        )
