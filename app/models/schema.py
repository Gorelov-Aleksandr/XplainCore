"""
Pydantic schema models for XAI service.

This module contains the data models for request validation and responses
using Pydantic models.
"""
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, model_validator

from app.models import ModelType, ExplanationMethod, VisualizationType


class InputData(BaseModel):
    """
    Input data model for XAI service.
    Validates the input data before processing.
    """
    # Core financial attributes
    income: float = Field(..., gt=0, example=50000.0, description="User's income")
    loan_amount: float = Field(..., gt=0, example=20000.0, description="Requested loan amount")
    credit_history: int = Field(..., ge=0, le=10, example=7, description="Credit history score (0-10)")
    
    # Optional extended attributes for more detailed analysis
    employment_years: Optional[float] = Field(None, ge=0, example=5.5, description="Years of employment")
    debt_to_income_ratio: Optional[float] = Field(None, ge=0, lt=1, example=0.3, description="Current debt to income ratio")
    age: Optional[int] = Field(None, ge=18, example=35, description="Applicant age")
    previous_defaults: Optional[int] = Field(None, ge=0, example=0, description="Number of previous defaults")
    education_level: Optional[str] = Field(None, example="bachelor", description="Education level")
    dependents: Optional[int] = Field(None, ge=0, example=2, description="Number of dependents")
    
    # Explainability configuration
    explanation_methods: Optional[List[ExplanationMethod]] = Field(
        default=[ExplanationMethod.FEATURE_IMPORTANCE], 
        description="Methods to use for explaining the model's decision"
    )
    visualization_types: Optional[List[VisualizationType]] = Field(
        default=[VisualizationType.BAR_CHART], 
        description="Types of visualizations to generate"
    )
    comparison_reference: Optional[str] = Field(
        None, 
        description="Reference point for explanations (e.g., 'average', 'optimal')"
    )
    max_features_to_show: Optional[int] = Field(
        5, ge=1, le=20, 
        description="Maximum number of features to show in the explanation"
    )
    
    @validator('loan_amount')
    def validate_loan_amount(cls, v, values):
        """
        Validates that loan amount doesn't exceed 50% of income.
        """
        if 'income' in values and v > values['income'] * 0.5:
            raise ValueError("Loan amount exceeds 50% of income")
        return v
    
    @model_validator(mode='after')
    def validate_explanation_config(self):
        """
        Validates that the explanation configuration is valid
        """
        methods = self.explanation_methods or [ExplanationMethod.FEATURE_IMPORTANCE]
        vis_types = self.visualization_types or [VisualizationType.BAR_CHART]
        
        # Check if visualization types are compatible with explanation methods
        for vis_type in vis_types:
            if vis_type == VisualizationType.WATERFALL and ExplanationMethod.SHAPLEY not in methods:
                methods.append(ExplanationMethod.SHAPLEY)
            
            if vis_type == VisualizationType.ATTENTION_MAP and ExplanationMethod.ATTENTION_VISUALIZATION not in methods:
                methods.append(ExplanationMethod.ATTENTION_VISUALIZATION)
        
        self.explanation_methods = methods
        return self


class VisualizationData(BaseModel):
    """
    Structure for visualization data
    """
    type: VisualizationType
    title: str
    description: str
    data: Dict[str, Any]
    format: str = "json"  # Could be json, svg, base64, etc.


class Confidence(BaseModel):
    """
    Model confidence metrics
    """
    overall_score: float = Field(..., ge=0, le=1, description="Overall confidence score")
    feature_reliability: Dict[str, float] = Field(..., description="Reliability score per feature")
    uncertainty_range: Optional[Dict[str, List[float]]] = Field(None, description="Uncertainty ranges")
    statistical_significance: Optional[Dict[str, float]] = Field(None, description="Statistical significance per feature")


class ExplanationDetails(BaseModel):
    """
    Detailed structure for model explanations
    """
    method: ExplanationMethod
    model_type: ModelType
    feature_importance: Optional[Dict[str, float]] = None
    counterfactuals: Optional[List[Dict[str, Any]]] = None
    feature_interactions: Optional[Dict[str, Dict[str, float]]] = None
    decision_rules: Optional[List[str]] = None
    example_cases: Optional[List[Dict[str, Any]]] = None
    local_explanation: Optional[Dict[str, Any]] = None
    global_explanation: Optional[Dict[str, Any]] = None
    bias_metrics: Optional[Dict[str, float]] = None
    robustness_metrics: Optional[Dict[str, Any]] = None
    visualizations: List[VisualizationData] = Field(default_factory=list)
    

class ExplanationResponse(BaseModel):
    """
    Response model for explanation endpoint with enhanced metadata and multiple explanation methods.
    """
    request_id: str = Field(..., description="Unique request identifier")
    prediction: Dict[str, Any] = Field(..., description="Model prediction results")
    confidence: Confidence = Field(..., description="Confidence metrics for the prediction")
    explanations: List[ExplanationDetails] = Field(..., description="List of explanations using different methods")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata about the response")
    regulatory_compliance: Optional[Dict[str, Any]] = Field(None, description="Compliance information (e.g., GDPR)")
    computation_time: Dict[str, float] = Field(..., description="Computation time for different parts of the explanation")
    version_info: Dict[str, str] = Field(..., description="Version information for the model and explainer")