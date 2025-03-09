from fastapi import APIRouter, Response
from prometheus_client import generate_latest

router = APIRouter(tags=["monitoring"])

@router.get("/health", summary="Service health check")
async def health():
    """
    Health check endpoint for the service.
    Used for monitoring and load balancers.
    
    Returns:
        dict: Status information
    """
    return {"status": "OK", "message": "Service is healthy"}

@router.get("/metrics", summary="Prometheus metrics")
async def metrics():
    """
    Endpoint for Prometheus metrics.
    
    Returns:
        Response: Prometheus metrics in text format
    """
    return Response(content=generate_latest(), media_type="text/plain")
