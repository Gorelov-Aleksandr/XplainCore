"""
Middleware module for the XAI service.

This module provides middleware for the FastAPI application:
- Rate limiting
- Metrics collection
- Logging
- Request ID generation
"""
import time
import uuid
import json
from typing import Dict

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
import sentry_sdk

from app.config import settings


# Simple in-memory rate limiting store
# In production, this should use Redis or another distributed cache
rate_limit_store: Dict[str, Dict] = {}


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
    # Skip rate limiting for health check endpoints
    if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
        return await call_next(request)
    
    # Get client IP (in production, handle X-Forwarded-For)
    client_ip = request.client.host if request.client else "unknown"
    
    # Get current time
    current_time = int(time.time())
    
    # Clean up expired rate limit entries
    for ip in list(rate_limit_store.keys()):
        if current_time - rate_limit_store[ip]["timestamp"] > settings.rate_limit_window:
            del rate_limit_store[ip]
    
    # Check if client is rate limited
    if client_ip in rate_limit_store:
        if rate_limit_store[client_ip]["count"] >= settings.rate_limit:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return Response(
                content=json.dumps({"detail": "Rate limit exceeded"}),
                status_code=429,
                media_type="application/json"
            )
        
        # Increment count
        rate_limit_store[client_ip]["count"] += 1
    else:
        # New client
        rate_limit_store[client_ip] = {
            "timestamp": current_time,
            "count": 1
        }
    
    # Continue with the request
    return await call_next(request)


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
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Start timing
    start_time = time.time()
    
    # Add context to logger
    logger.bind(request_id=request_id)
    
    # Log the request
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Request: {request.method} {request.url.path} from {client_ip}")
    
    try:
        # Process the request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        # Calculate processing time
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log the response
        logger.info(f"Response: {response.status_code} in {process_time:.4f}s")
        
        return response
    except Exception as e:
        # Capture exception in Sentry
        if settings.sentry_dsn:
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("request_id", request_id)
                scope.set_context("request", {
                    "method": request.method,
                    "path": request.url.path,
                    "client_ip": client_ip
                })
                sentry_sdk.capture_exception(e)
        
        # Log the error
        logger.exception(f"Error processing request: {str(e)}")
        
        # Re-raise the exception
        raise