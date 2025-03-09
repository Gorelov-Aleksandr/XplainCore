"""
Feature importance explainer implementation.
"""
import time
import json
import random
import logging
from typing import Dict, Any, List, Optional

# Using standard logging instead of loguru for better compatibility
logger = logging.getLogger(__name__)

from app.models import ModelType, ExplanationMethod, VisualizationType
from app.models.schema import InputData, ExplanationDetails, VisualizationData
from .base import BaseExplainer

class FeatureImportanceExplainer(BaseExplainer):
    """
    Implementation of feature importance explanation method.
    This provides interpretable insights by ranking features based on their importance
    to the model's prediction.
    """
    def __init__(self, model_type: ModelType = ModelType.RULE_BASED):
        """
        Initialize the feature importance explainer.
        
        Args:
            model_type: Type of model being explained
        """
        super().__init__(model_type=model_type, method=ExplanationMethod.FEATURE_IMPORTANCE)
    
    def _compute_feature_importance(self, data: InputData) -> Dict[str, float]:
        """
        Compute feature importance scores.
        
        In a real implementation, this would use a trained model to determine feature importance.
        For the demo, we use a rule-based approach to simulate the importance scores.
        
        Args:
            data: Input data
            
        Returns:
            Dict: Feature importance scores
        """
        # This decorator would be used here in a real implementation
        # We're avoiding it for now to simplify the demo
        start_time = time.time()
        # For loan decisions, calculate importance based on specific domain knowledge
        loan_to_income_ratio = data.loan_amount / data.income
        
        # Basic feature importance calculation
        # In a real model, this would use trained coefficients, feature permutation, etc.
        importance = {
            "income": 0.35,
            "loan_amount": 0.25,
            "credit_history": 0.40,
        }
        
        # Adjust based on additional factors if provided
        if data.employment_years is not None:
            importance["employment_years"] = 0.15
            # Normalize to ensure sum is 1.0
            total = sum(importance.values())
            importance = {k: v/total for k, v in importance.items()}
        
        if data.debt_to_income_ratio is not None:
            importance["debt_to_income_ratio"] = 0.20
            # Normalize to ensure sum is 1.0
            total = sum(importance.values())
            importance = {k: v/total for k, v in importance.items()}
            
        # Further adjust based on loan-to-income ratio
        if loan_to_income_ratio > 0.3:
            # When loan amount is high relative to income, it becomes more important
            importance["loan_amount"] *= 1.2
            importance["income"] *= 1.2
            # Normalize again
            total = sum(importance.values())
            importance = {k: v/total for k, v in importance.items()}
            
        return importance
    
    def _generate_decision_rules(self, data: InputData) -> List[str]:
        """
        Generate human-readable decision rules from the model.
        
        Args:
            data: Input data
            
        Returns:
            List[str]: List of decision rules
        """
        # Using direct timing instead of decorator for simplicity
        start_time = time.time()
        loan_to_income_ratio = data.loan_amount / data.income
        credit_factor = data.credit_history / 10.0
        
        rules = []
        
        # Add basic decision rules
        rules.append(f"Loan-to-income ratio is {loan_to_income_ratio:.2f} (threshold: 0.3)")
        rules.append(f"Credit factor is {credit_factor:.2f} (threshold: 0.7)")
        
        # Add conditional rules
        if loan_to_income_ratio < 0.3:
            rules.append("Loan amount is within acceptable range relative to income")
        else:
            rules.append("Loan amount is high relative to income")
        
        if credit_factor > 0.7:
            rules.append("Credit history is strong enough to approve")
        else:
            rules.append("Credit history is below the preferred threshold")
        
        # Add additional rules based on optional fields
        if data.employment_years is not None:
            if data.employment_years > 3:
                rules.append(f"Employment history of {data.employment_years:.1f} years indicates stability")
            else:
                rules.append(f"Employment history of {data.employment_years:.1f} years is below threshold (3 years)")
        
        if data.debt_to_income_ratio is not None:
            if data.debt_to_income_ratio < 0.4:
                rules.append(f"Current debt-to-income ratio ({data.debt_to_income_ratio:.2f}) is within acceptable range")
            else:
                rules.append(f"Current debt-to-income ratio ({data.debt_to_income_ratio:.2f}) is higher than preferred")
        
        return rules
    
    def _create_visualizations(self, 
                               data: InputData, 
                               feature_importance: Dict[str, float], 
                               max_features: int = 5) -> List[VisualizationData]:
        """
        Create visualizations for feature importance.
        
        Args:
            data: Input data
            feature_importance: Feature importance scores
            max_features: Maximum number of features to include
            
        Returns:
            List[VisualizationData]: List of visualization data
        """
        visualizations = []
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:max_features]
        
        # Bar chart visualization
        bar_chart_data = {
            "labels": [item[0] for item in sorted_features],
            "values": [item[1] for item in sorted_features],
            "colors": ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF6D01"][:len(sorted_features)],
            "orientation": "h",  # horizontal
        }
        
        bar_chart = VisualizationData(
            type=VisualizationType.BAR_CHART,
            title="Feature Importance",
            description="Relative importance of each feature in the model's decision",
            data=bar_chart_data,
            format="json"
        )
        visualizations.append(bar_chart)
        
        # Add a radar chart if we have enough features
        if len(sorted_features) >= 3:
            radar_data = {
                "labels": [item[0] for item in sorted_features],
                "values": [item[1] for item in sorted_features],
                "max_value": 1.0,
                "fill": True,
                "fill_color": "rgba(66, 133, 244, 0.2)",
                "line_color": "#4285F4"
            }
            
            radar_chart = VisualizationData(
                type=VisualizationType.RADAR_CHART,
                title="Feature Importance Distribution",
                description="Distribution of importance across different features",
                data=radar_data,
                format="json"
            )
            visualizations.append(radar_chart)
        
        return visualizations
    
    async def get_confidence(self, data: InputData, **kwargs) -> Dict[str, Any]:
        """
        Calculate confidence metrics for the explanation.
        
        Args:
            data: The input data
            **kwargs: Additional parameters
            
        Returns:
            Dict: Confidence metrics
        """
        # Calculate overall confidence
        loan_to_income_ratio = data.loan_amount / data.income
        credit_factor = data.credit_history / 10.0
        
        # For confidence, we're looking at how strongly the features support the decision
        feature_importance = self._compute_feature_importance(data)
        
        # Calculate feature reliability - how confident we are in each feature's importance
        feature_reliability = {}
        for feature, importance in feature_importance.items():
            # Higher importance features typically have higher reliability
            # Add a small amount of variation for realism
            reliability = min(0.95, importance * 1.5 + random.uniform(0.05, 0.15))
            feature_reliability[feature] = round(reliability, 2)
        
        # Overall confidence is weighted average of feature reliabilities
        overall_score = sum(imp * feature_reliability[feat] for feat, imp in feature_importance.items())
        
        # Generate uncertainty range for key metrics
        uncertainty_range = {
            "loan_to_income_ratio": [max(0, loan_to_income_ratio - 0.05), min(1, loan_to_income_ratio + 0.05)],
            "credit_factor": [max(0, credit_factor - 0.05), min(1, credit_factor + 0.05)]
        }
        
        # Statistical significance (p-values) - in real system these would be computed
        statistical_significance = {
            "income": 0.001,  # Very significant
            "loan_amount": 0.015,
            "credit_history": 0.005
        }
        
        return {
            "overall_score": round(overall_score, 2),
            "feature_reliability": feature_reliability,
            "uncertainty_range": uncertainty_range,
            "statistical_significance": statistical_significance
        }
    
    async def validate(self) -> bool:
        """
        Validate that the explainer is properly configured.
        
        Returns:
            bool: True if valid, False otherwise
        """
        # Simple validation for this explainer
        return True
    
    async def explain(self, data: InputData, **kwargs) -> ExplanationDetails:
        """
        Generate a feature importance explanation for the input data.
        
        Args:
            data: The input data to explain
            **kwargs: Additional parameters
            
        Returns:
            ExplanationDetails: The explanation details
        """
        start_total = time.time()
        
        # Compute feature importance
        feature_importance = self._compute_feature_importance(data)
        
        # Generate decision rules
        decision_rules = self._generate_decision_rules(data)
        
        # Create visualizations
        max_features = kwargs.get('max_features', data.max_features_to_show)
        visualizations = self._create_visualizations(data, feature_importance, max_features)
        
        end_total = time.time()
        self.computation_times["total_explanation_time"] = end_total - start_total
        
        # Create explanation details
        explanation = ExplanationDetails(
            method=self.method,
            model_type=self.model_type,
            feature_importance=feature_importance,
            decision_rules=decision_rules,
            visualizations=visualizations
        )
        
        return explanation