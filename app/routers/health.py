"""
Health check endpoints for the XAI service.

This module provides endpoints for health checks and metrics.
"""
from fastapi import APIRouter, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from loguru import logger

router = APIRouter(
    tags=["health"],
    responses={404: {"description": "Not found"}},
)


@router.get("/health")
async def health():
    """
    Health check endpoint for the service.
    Used for monitoring and load balancers.
    
    Returns:
        dict: Status information
    """
    return {
        "status": "ok",
        "service": "xai-service",
        "version": "1.0.0"
    }


@router.get("/metrics")
async def metrics():
    """
    Endpoint for Prometheus metrics.
    
    Returns:
        Response: Prometheus metrics in text format
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )