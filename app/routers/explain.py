from fastapi import APIRouter, Request, BackgroundTasks, HTTPException, Depends
from pydantic import ValidationError
import sentry_sdk
from loguru import logger
import json
import hashlib
import time
from functools import wraps

from ..models import InputData, ExplanationResponse
from ..auth import get_current_user
from ..config import settings
from ..monitoring import tracer

router = APIRouter(tags=["xai"])

# Simple in-memory cache
explanation_cache = {}

def cache_decorator(ttl=300):
    """
    Custom cache decorator that works without external dependencies.
    
    Args:
        ttl: Cache time-to-live in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Try to create a cache key from the first argument (input_data)
            try:
                if len(args) > 0 and hasattr(args[0], 'json'):
                    data_str = args[0].json()
                    cache_key = f"explain:{hashlib.md5(data_str.encode()).hexdigest()}"
                else:
                    # Fallback cache key if no data object
                    cache_key = f"explain:default:{time.time()}"
                
                # Check if result in cache and not expired
                if cache_key in explanation_cache:
                    cached_item = explanation_cache[cache_key]
                    if time.time() - cached_item["timestamp"] < ttl:
                        logger.info(f"Cache hit for key: {cache_key}")
                        return cached_item["data"]
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                explanation_cache[cache_key] = {
                    "data": result,
                    "timestamp": time.time()
                }
                logger.info(f"Cached result for key: {cache_key}")
                return result
            except Exception as e:
                logger.warning(f"Error in cache: {str(e)}")
                # Skip caching and just execute the function
                return await func(*args, **kwargs)
        return wrapper
    return decorator

async def log_explanation(data: dict, request_id: str):
    """
    Background task to log explanations.
    
    Args:
        data: Explanation data to log
        request_id: Unique request ID
    """
    logger.info(f"[{request_id}] Explanation generated: {data}")

@router.post(
    "/explain",
    response_model=ExplanationResponse,
    summary="Generate model explanation"
)
@cache_decorator(ttl=settings.cache_ttl)
async def explain(
    data: InputData,
    background_tasks: BackgroundTasks,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Generates an explanation for the input data.
    This endpoint requires authentication with Yandex OAuth or API key.
    
    Args:
        data: Input data for the model
        background_tasks: FastAPI background tasks
        request: FastAPI request object
        current_user: Current authenticated user
        
    Returns:
        ExplanationResponse: Explanation of the model's decision
        
    Raises:
        HTTPException: If there's a validation error or other exception
    """
    request_id = getattr(request.state, "request_id", str(time.time()))
    
    try:
        # Start tracing span
        with tracer.start_as_current_span("explain_endpoint") as span:
            # Add user info and input data to the span for better tracing
            span.set_attribute("user.id", current_user.get("id", "unknown"))
            span.set_attribute("input.income", data.income)
            span.set_attribute("input.loan_amount", data.loan_amount)
            
            # Calculate a decision based on the input data
            # This is where the actual model would be called
            loan_to_income_ratio = data.loan_amount / data.income
            credit_factor = data.credit_history / 10.0
            
            # Simple decision logic (in a real system, this would be a trained model)
            decision = "APPROVED" if (loan_to_income_ratio < 0.3 or credit_factor > 0.7) else "DENIED"
            
            # Generate explanation
            explanation = {
                "feature_importance": {
                    "income": 0.4,
                    "loan_amount": 0.3,
                    "credit_history": 0.3
                },
                "factors": {
                    "loan_to_income_ratio": round(loan_to_income_ratio, 2),
                    "credit_factor": round(credit_factor, 2)
                },
                "decision": decision,
                "timestamp": time.time()
            }
            
            # Log the explanation in the background
            background_tasks.add_task(log_explanation, explanation, request_id)
            
            return ExplanationResponse(
                request_id=request_id,
                explanation=explanation,
                metadata={
                    "version": settings.model_version,
                    "model_type": "rule_based_model",
                    "user_id": str(current_user.get("id", "unknown"))
                }
            )
    except ValidationError as e:
        logger.error(f"Validation error: {e.errors()}")
        raise HTTPException(
            status_code=422,
            detail={"errors": e.errors()}
        )
    except Exception as e:
        logger.exception(f"Unhandled exception in /explain endpoint: {str(e)}")
        try:
            sentry_sdk.capture_exception(e)
        except:
            logger.warning("Failed to send exception to Sentry")
        
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error"
        )
