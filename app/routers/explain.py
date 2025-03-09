from fastapi import APIRouter, Request, BackgroundTasks, HTTPException, Depends
from pydantic import ValidationError
import sentry_sdk
from loguru import logger
import json
import hashlib
import time
import uuid
import asyncio
from functools import wraps
from typing import Dict, List, Any, Optional

from ..models import InputData, ExplanationResponse, ExplanationMethod, ExplanationDetails
from ..auth import get_current_user
from ..config import settings
from ..monitoring import tracer
from ..explainers import FeatureImportanceExplainer, ShapleyExplainer, CounterfactualExplainer

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

async def get_explanation_for_method(method: ExplanationMethod, data: InputData, **kwargs) -> tuple:
    """
    Get explanation for a specific method by instantiating and using the appropriate explainer.
    
    Args:
        method: The explanation method to use
        data: Input data
        **kwargs: Additional parameters
        
    Returns:
        ExplanationDetails: The explanation details
    """
    if method == ExplanationMethod.FEATURE_IMPORTANCE:
        explainer = FeatureImportanceExplainer()
    elif method == ExplanationMethod.SHAPLEY:
        explainer = ShapleyExplainer()
    elif method == ExplanationMethod.COUNTERFACTUAL:
        explainer = CounterfactualExplainer()
    else:
        # For now, default to feature importance for unsupported methods
        logger.warning(f"Unsupported explanation method: {method}. Using feature importance instead.")
        explainer = FeatureImportanceExplainer()
    
    # Validate explainer configuration
    if not await explainer.validate():
        logger.error(f"Explainer validation failed for method: {method}")
        raise HTTPException(status_code=500, detail=f"Explainer validation failed for method: {method}")
    
    # Generate explanation
    explanation = await explainer.explain(data, **kwargs)
    
    # Return computation times for metrics
    computation_times = explainer.get_computation_times()
    
    return explanation, computation_times

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
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    start_time = time.time()
    
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
            
            # Run appropriate explainers based on requested methods
            explanations = []
            computation_times = {}
            
            # If no methods specified, default to feature importance
            methods = data.explanation_methods or [ExplanationMethod.FEATURE_IMPORTANCE]
            
            # Generate explanations for each requested method
            for method in methods:
                try:
                    logger.info(f"Generating explanation using method: {method}")
                    explanation, method_times = await get_explanation_for_method(method, data)
                    explanations.append(explanation)
                    
                    # Add method-specific computation times
                    for key, value in method_times.items():
                        computation_times[f"{method.value}_{key}"] = value
                        
                except Exception as e:
                    logger.error(f"Error generating explanation for method {method}: {str(e)}")
                    # Continue with other methods even if one fails
            
            # Calculate total time
            end_time = time.time()
            computation_times["total"] = end_time - start_time
            
            # Use the confidence metrics from the first explainer
            # In a production system, we might want to combine confidence metrics from multiple explainers
            first_explainer = None
            if methods[0] == ExplanationMethod.FEATURE_IMPORTANCE:
                first_explainer = FeatureImportanceExplainer()
            elif methods[0] == ExplanationMethod.SHAPLEY:
                first_explainer = ShapleyExplainer()
            elif methods[0] == ExplanationMethod.COUNTERFACTUAL:
                first_explainer = CounterfactualExplainer()
            
            confidence_metrics = await first_explainer.get_confidence(data) if first_explainer else {
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
                    "income": 0.001,
                    "loan_amount": 0.015,
                    "credit_history": 0.005
                }
            }
            
            # Version information
            version_info = {
                "model": settings.model_version,
                "explainer": "1.1.0",  # Updated version with multiple explainers
                "api": "1.0.0"
            }
            
            # Create the response
            response = ExplanationResponse(
                request_id=request_id,
                prediction=prediction,
                confidence=confidence_metrics,
                explanations=explanations,
                metadata={
                    "model_type": "rule_based_model",
                    "user_id": str(current_user.get("id", "dev-user-id")),
                    "timestamp": time.time(),
                    "explanation_methods": [method.value for method in methods]
                },
                computation_time=computation_times,
                version_info=version_info
            )
            
            # Log the explanation in the background
            background_tasks.add_task(log_explanation, [e.model_dump() for e in explanations], request_id)
            
            return response
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
