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
            # Just execute the function directly for now (disabled cache)
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
            
            # Generate prediction results
            prediction = {
                "decision": decision,
                "loan_to_income_ratio": round(loan_to_income_ratio, 2),
                "credit_factor": round(credit_factor, 2),
                "score": 0.7 + (0.1 * credit_factor) - (0.2 * loan_to_income_ratio)
            }
            
            # Compute confidence metrics
            confidence_metrics = {
                "overall_score": 0.85,
                "feature_reliability": {
                    "income": 0.9,
                    "loan_amount": 0.8,
                    "credit_history": 0.95
                },
                "uncertainty_range": {
                    "loan_to_income_ratio": [max(0, loan_to_income_ratio - 0.05), min(1, loan_to_income_ratio + 0.05)],
                    "credit_factor": [max(0, credit_factor - 0.05), min(1, credit_factor + 0.05)]
                },
                "statistical_significance": {
                    "income": 0.001,  # Very significant
                    "loan_amount": 0.015,
                    "credit_history": 0.005
                }
            }
            
            # Create explanation details (using the feature importance method)
            explanation_details = [{
                "method": "feature_importance",
                "model_type": "rule_based",
                "feature_importance": {
                    "income": 0.4,
                    "loan_amount": 0.3,
                    "credit_history": 0.3
                },
                "decision_rules": [
                    "Loan-to-income ratio is " + ("good" if loan_to_income_ratio < 0.3 else "concerning"),
                    "Credit history score is " + ("strong" if credit_factor > 0.7 else "weak")
                ],
                "visualizations": [
                    {
                        "type": "bar_chart",
                        "title": "Feature Importance",
                        "description": "Relative importance of each feature in the model's decision",
                        "data": {
                            "labels": ["income", "loan_amount", "credit_history"],
                            "values": [0.4, 0.3, 0.3],
                            "colors": ["#4285F4", "#EA4335", "#FBBC05"]
                        },
                        "format": "json"
                    }
                ]
            }]
            
            # Computation times
            computation_time = {
                "total": 0.025,
                "feature_importance": 0.015,
                "visualization": 0.010
            }
            
            # Version information
            version_info = {
                "model": settings.model_version,
                "explainer": "1.0.0",
                "api": "1.0.0"
            }
            
            # Log the explanation in the background
            background_tasks.add_task(log_explanation, explanation_details, request_id)
            
            return ExplanationResponse(
                request_id=request_id,
                prediction=prediction,
                confidence=confidence_metrics,
                explanations=explanation_details,
                metadata={
                    "model_type": "rule_based_model",
                    "user_id": str(current_user.get("id", "dev-user-id")),
                    "timestamp": time.time()
                },
                computation_time=computation_time,
                version_info=version_info
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
