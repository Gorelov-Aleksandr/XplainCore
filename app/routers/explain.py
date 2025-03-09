from fastapi import APIRouter, Request, BackgroundTasks, HTTPException, Depends
from pydantic import ValidationError
import sentry_sdk
from loguru import logger
from aiocache import cached

from ..models import InputData, ExplanationResponse
from ..auth import get_current_user
from ..cache import cache, custom_cache_key_builder
from ..config import settings
from ..monitoring import tracer

router = APIRouter(tags=["xai"])

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
@cached(ttl=settings.cache_ttl, key_builder=custom_cache_key_builder, cache=cache)
async def explain(
    data: InputData,
    background_tasks: BackgroundTasks,
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """
    Generates an explanation for the input data.
    This endpoint requires authentication with Yandex OAuth.
    
    Args:
        data: Input data for the model
        background_tasks: FastAPI background tasks
        request: FastAPI request object
        current_user: Current authenticated user (from OAuth)
        
    Returns:
        ExplanationResponse: Explanation of the model's decision
        
    Raises:
        HTTPException: If there's a validation error or other exception
    """
    with tracer.start_as_current_span("explain_endpoint") as span:
        try:
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
                "decision": decision
            }
            
            # Log the explanation in the background
            background_tasks.add_task(log_explanation, explanation, request.state.request_id)
            
            return ExplanationResponse(
                request_id=request.state.request_id,
                explanation=explanation,
                metadata={
                    "version": "1.0",
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
            logger.exception("Unhandled exception in /explain endpoint")
            sentry_sdk.capture_exception(e)
            raise HTTPException(
                status_code=500,
                detail="Internal Server Error"
            )
