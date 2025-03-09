"""
Endpoints for retrieving explanation history.

This module provides API endpoints for retrieving past explanations.
"""
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from loguru import logger

from app.database import get_db
from app.auth import get_current_user
from app.repository import ExplanationRepository
from app.models.schema import ExplanationResponse

router = APIRouter(
    prefix="/history",
    tags=["history"],
    responses={404: {"description": "Not found"}},
)


@router.get("/explanations")
async def get_recent_explanations(
    limit: int = Query(10, ge=1, le=100),
    db = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieve recent explanations for the current user.
    
    Args:
        limit: Maximum number of explanations to return (1-100)
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        List of recent explanations
    """
    try:
        user_id = current_user.get("user_id")
        explanations = await ExplanationRepository.get_recent_explanations(db, limit, user_id)
        return {"explanations": explanations}
    except Exception as e:
        logger.error(f"Error retrieving recent explanations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving explanations"
        )


@router.get("/explanations/{request_id}")
async def get_explanation_by_id(
    request_id: str,
    db = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieve a specific explanation by request ID.
    
    Args:
        request_id: Unique request identifier
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        The explanation data
    """
    try:
        explanation = await ExplanationRepository.get_explanation_by_request_id(db, request_id)
        
        if not explanation:
            raise HTTPException(
                status_code=404,
                detail=f"Explanation with request_id {request_id} not found"
            )
        
        # Check if the explanation belongs to the current user
        user_id = current_user.get("user_id")
        if explanation.get("user_id") and explanation.get("user_id") != user_id:
            raise HTTPException(
                status_code=403,
                detail="You don't have permission to access this explanation"
            )
        
        return explanation
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving explanation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving explanation"
        )