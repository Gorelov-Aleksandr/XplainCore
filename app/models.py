from typing import Any, Dict
from pydantic import BaseModel, Field, validator

class InputData(BaseModel):
    """
    Input data model for XAI service.
    Validates the input data before processing.
    """
    income: float = Field(..., gt=0, example=50000.0, description="User's income")
    loan_amount: float = Field(..., gt=0, example=20000.0, description="Requested loan amount")
    credit_history: int = Field(..., ge=0, le=10, example=7, description="Credit history score (0-10)")

    @validator('loan_amount')
    def validate_loan_amount(cls, v, values):
        """
        Validates that loan amount doesn't exceed 50% of income.
        """
        if 'income' in values and v > values['income'] * 0.5:
            raise ValueError("Loan amount exceeds 50% of income")
        return v

class ExplanationResponse(BaseModel):
    """
    Response model for explanation endpoint.
    """
    request_id: str = Field(..., description="Unique request identifier")
    explanation: Dict[str, Any] = Field(..., description="Model explanation details")
    metadata: Dict[str, str] = Field(..., description="Additional metadata about the response")
