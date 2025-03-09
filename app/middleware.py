import time
import uuid
from typing import Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from loguru import logger
import sentry_sdk

from .config import settings
from .monitoring import REQUEST_COUNTER, RESPONSE_TIME, ACTIVE_REQUESTS

# Storage for rate limiting
rate_limit_cache: Dict[str, Dict[str, Any]] = {}

async def rate_limiting_middleware(request: Request, call_next):
    """
    Middleware for rate limiting requests based on client IP.
    Limits requests to settings.rate_limit per settings.rate_limit_window seconds.
    
    Args:
        request: FastAPI request object
        call_next: Next middleware in chain
        
    Returns:
        Response: FastAPI response
    """
    client_ip = request.client.host
    current_time = time.time()
    window = settings.rate_limit_window
    
    if client_ip not in rate_limit_cache:
        rate_limit_cache[client_ip] = {"count": 1, "start_time": current_time}
    else:
        elapsed = current_time - rate_limit_cache[client_ip]["start_time"]
        if elapsed > window:
            rate_limit_cache[client_ip] = {"count": 1, "start_time": current_time}
        else:
            rate_limit_cache[client_ip]["count"] += 1
            if rate_limit_cache[client_ip]["count"] > settings.rate_limit:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded. Try again later."}
                )
                
    response = await call_next(request)
    return response

async def observability_middleware(request: Request, call_next):
    """
    Middleware for observability:
    - Adds request_id to each request
    - Logs request and response
    - Records Prometheus metrics
    - Captures exceptions in Sentry
    
    Args:
        request: FastAPI request object
        call_next: Next middleware in chain
        
    Returns:
        Response: FastAPI response
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    logger.bind(request_id=request_id)
    
    start_time = time.time()
    ACTIVE_REQUESTS.labels(endpoint=request.url.path).inc()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        REQUEST_COUNTER.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        RESPONSE_TIME.labels(request.url.path).observe(duration)
        
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        sentry_sdk.capture_exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal Server Error", "request_id": request_id}
        )
    finally:
        ACTIVE_REQUESTS.labels(endpoint=request.url.path).dec()
