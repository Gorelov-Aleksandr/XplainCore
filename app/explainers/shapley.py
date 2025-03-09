"""
Shapley values explainer implementation.

Shapley values are a concept from cooperative game theory used in XAI to 
allocate the contribution of each feature to the prediction in a fair way.
"""
import time
import random
import json
import math
import logging
from typing import Dict, Any, List, Optional, Tuple

# Using standard logging instead of loguru for better compatibility
logger = logging.getLogger(__name__)

from ..models import ModelType, ExplanationMethod, InputData, ExplanationDetails, VisualizationType, VisualizationData
from .base import BaseExplainer

class ShapleyExplainer(BaseExplainer):
    """
    Implementation of Shapley values explanation method.
    This provides a mathematically fair approach to calculate feature attribution
    by examining all possible combinations of features.
    """
    def __init__(self, model_type: ModelType = ModelType.TABULAR_ML):
        """
        Initialize the Shapley values explainer.
        
        Args:
            model_type: Type of model being explained
        """
        super().__init__(model_type=model_type, method=ExplanationMethod.SHAPLEY)
    
    def _compute_shapley_values(self, data: InputData) -> Dict[str, float]:
        """
        Compute Shapley values for the features.
        
        For a real implementation, this would involve a lot of model evaluations
        across all possible feature combinations. For the demo, we use a simplified
        approach that approximates Shapley values.
        
        Args:
            data: Input data
            
        Returns:
            Dict: Shapley values for each feature
        """
        # Core features
        features = {
            "income": data.income,
            "loan_amount": data.loan_amount,
            "credit_history": data.credit_history
        }
        
        # Add optional features if provided
        if data.employment_years is not None:
            features["employment_years"] = data.employment_years
        if data.debt_to_income_ratio is not None:
            features["debt_to_income_ratio"] = data.debt_to_income_ratio
        if data.age is not None:
            features["age"] = data.age
        if data.previous_defaults is not None:
            features["previous_defaults"] = data.previous_defaults
        
        # Base prediction (simple rule-based for demo)
        base_prediction = self._make_prediction(features)
        
        # Calculate Shapley values by leaving out each feature
        shapley_values = {}
        for feature_name in features.keys():
            # Create a version without this feature (set to average/reference value)
            features_without = features.copy()
            if feature_name == "income":
                features_without[feature_name] = 50000  # Average income
            elif feature_name == "loan_amount":
                features_without[feature_name] = 15000  # Average loan
            elif feature_name == "credit_history":
                features_without[feature_name] = 5  # Average credit score
            elif feature_name == "employment_years":
                features_without[feature_name] = 3  # Average employment years
            elif feature_name == "debt_to_income_ratio":
                features_without[feature_name] = 0.3  # Average DTI
            elif feature_name == "age":
                features_without[feature_name] = 35  # Average age
            elif feature_name == "previous_defaults":
                features_without[feature_name] = 0  # Average defaults
            
            # Calculate prediction without this feature
            prediction_without = self._make_prediction(features_without)
            
            # Shapley value is impact on the prediction
            shapley_values[feature_name] = base_prediction - prediction_without
        
        # Normalize Shapley values
        total = sum(abs(v) for v in shapley_values.values())
        if total > 0:
            shapley_values = {k: v/total for k, v in shapley_values.items()}
        
        return shapley_values
    
    def _make_prediction(self, features: Dict[str, Any]) -> float:
        """
        Make a prediction based on the features.
        This is a simplified model for demo purposes.
        
        Args:
            features: Feature values
            
        Returns:
            float: Prediction score (0 to 1)
        """
        # Calculate loan-to-income ratio
        loan_to_income_ratio = features.get("loan_amount", 0) / features.get("income", 1)
        credit_factor = features.get("credit_history", 0) / 10.0
        
        # Base score from main features
        score = 0.7 - loan_to_income_ratio + credit_factor
        
        # Adjust based on optional features
        if "employment_years" in features:
            employment_factor = min(1.0, features["employment_years"] / 10.0)
            score += employment_factor * 0.1
            
        if "debt_to_income_ratio" in features:
            score -= features["debt_to_income_ratio"] * 0.2
            
        if "previous_defaults" in features:
            score -= features["previous_defaults"] * 0.15
            
        # Ensure score is between 0 and 1
        return max(0, min(1, score))
    
    def _compute_feature_interactions(self, data: InputData) -> Dict[str, Dict[str, float]]:
        """
        Compute interactions between features based on Shapley interaction index.
        
        Args:
            data: Input data
            
        Returns:
            Dict: Feature interactions
        """
        # This is a simplified version for demo purposes
        # Real Shapley interaction indices require evaluating all feature subsets
        
        # Core features
        features = ["income", "loan_amount", "credit_history"]
        
        # Add optional features if provided
        if data.employment_years is not None:
            features.append("employment_years")
        if data.debt_to_income_ratio is not None:
            features.append("debt_to_income_ratio")
        
        # Calculate interaction strengths
        interactions = {}
        for i, feat1 in enumerate(features):
            interactions[feat1] = {}
            for j, feat2 in enumerate(features):
                if i != j:
                    # Some hypothetical interactions between features
                    if (feat1 == "income" and feat2 == "loan_amount"):
                        # Strong interaction - they work together as loan-to-income ratio
                        interactions[feat1][feat2] = 0.8
                    elif (feat1 == "credit_history" and feat2 == "previous_defaults"):
                        # Strong interaction - both related to credit risk
                        interactions[feat1][feat2] = 0.7
                    elif (feat1 == "employment_years" and feat2 == "income"):
                        # Moderate interaction
                        interactions[feat1][feat2] = 0.5
                    else:
                        # Weak random interaction
                        interactions[feat1][feat2] = round(random.uniform(0.1, 0.3), 2)
        
        return interactions
    
    @BaseExplainer.measure_time("create_visualizations")
    def _create_visualizations(self, 
                              data: InputData, 
                              shapley_values: Dict[str, float],
                              feature_interactions: Dict[str, Dict[str, float]],
                              max_features: int = 5) -> List[VisualizationData]:
        """
        Create visualizations for Shapley values.
        
        Args:
            data: Input data
            shapley_values: Shapley values
            feature_interactions: Feature interactions
            max_features: Maximum number of features to include
            
        Returns:
            List[VisualizationData]: List of visualization data
        """
        visualizations = []
        
        # Sort features by absolute Shapley value
        sorted_features = sorted(shapley_values.items(), key=lambda x: abs(x[1]), reverse=True)[:max_features]
        
        # Waterfall plot - shows how each feature contributes to moving from the base value
        waterfall_data = {
            "labels": [item[0] for item in sorted_features],
            "values": [item[1] for item in sorted_features],
            "base_value": 0.5,  # Base prediction (average outcome)
            "colors": {
                "positive": "#34A853",  # Green for positive impact
                "negative": "#EA4335",  # Red for negative impact
                "total": "#4285F4"      # Blue for total
            }
        }
        
        waterfall_plot = VisualizationData(
            type=VisualizationType.WATERFALL,
            title="Feature Contributions (Shapley Values)",
            description="How each feature moves the prediction from the baseline",
            data=waterfall_data,
            format="json"
        )
        visualizations.append(waterfall_plot)
        
        # Force plot - shows the push and pull of each feature on the prediction
        force_data = {
            "features": [item[0] for item in sorted_features],
            "values": [item[1] for item in sorted_features],
            "base_value": 0.5,
            "prediction": sum(item[1] for item in sorted_features) + 0.5,
            "feature_values": {
                "income": data.income,
                "loan_amount": data.loan_amount,
                "credit_history": data.credit_history
            }
        }
        
        # Add optional features if they were included
        if data.employment_years is not None:
            force_data["feature_values"]["employment_years"] = data.employment_years
        if data.debt_to_income_ratio is not None:
            force_data["feature_values"]["debt_to_income_ratio"] = data.debt_to_income_ratio
        
        force_plot = VisualizationData(
            type=VisualizationType.FORCE_PLOT,
            title="Force Plot (SHAP)",
            description="How each feature pushes the prediction higher or lower",
            data=force_data,
            format="json"
        )
        visualizations.append(force_plot)
        
        # Heatmap for feature interactions
        if feature_interactions and len(feature_interactions) > 1:
            # Convert interactions to a matrix format
            features_for_heatmap = [f[0] for f in sorted_features[:5]]  # Top 5 features
            interaction_matrix = []
            
            for feat1 in features_for_heatmap:
                row = []
                for feat2 in features_for_heatmap:
                    if feat1 == feat2:
                        row.append(1.0)  # Perfect interaction with self
                    else:
                        row.append(feature_interactions.get(feat1, {}).get(feat2, 0))
                interaction_matrix.append(row)
            
            heatmap_data = {
                "matrix": interaction_matrix,
                "x_labels": features_for_heatmap,
                "y_labels": features_for_heatmap,
                "colorscale": "Blues"
            }
            
            heatmap = VisualizationData(
                type=VisualizationType.HEATMAP,
                title="Feature Interactions",
                description="Strength of interactions between features",
                data=heatmap_data,
                format="json"
            )
            visualizations.append(heatmap)
        
        return visualizations
    
    @BaseExplainer.measure_time("generate_confidence_metrics")
    async def get_confidence(self, data: InputData, **kwargs) -> Dict[str, Any]:
        """
        Calculate confidence metrics for the explanation.
        
        Args:
            data: The input data
            **kwargs: Additional parameters
            
        Returns:
            Dict: Confidence metrics
        """
        # For Shapley values, confidence relates to sampling accuracy
        # In real implementations, this would be based on convergence metrics
        
        # Calculate feature reliability - how confident we are in each Shapley value
        shapley_values = self._compute_shapley_values(data)
        
        feature_reliability = {}
        for feature, shapley in shapley_values.items():
            # Reliability can be lower than feature importance since Shapley approximation has variance
            reliability = 0.85 + random.uniform(-0.1, 0.1)  # High reliability with some variation
            feature_reliability[feature] = round(reliability, 2)
        
        # Overall confidence is weighted average of feature reliabilities
        # For Shapley values, all features have equal weight in the reliability calculation
        overall_score = sum(feature_reliability.values()) / len(feature_reliability)
        
        # Higher confidence in shapley values for more important features
        sorted_features = sorted(shapley_values.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Generate uncertainty ranges based on statistical properties
        uncertainty_range = {}
        for feature, value in sorted_features:
            # More important features may have narrower uncertainty ranges
            width = 0.2 * (1 - abs(value))  # Larger values have smaller uncertainty
            uncertainty_range[feature] = [value - width/2, value + width/2]
        
        # Statistical significance (p-values) - in real system these would be computed
        statistical_significance = {}
        for feature, value in sorted_features:
            # More important features typically have higher statistical significance
            p_value = max(0.001, 0.05 * (1 - abs(value)))
            statistical_significance[feature] = round(p_value, 3)
        
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
        # For Shapley explainer, just check that required dependencies are available
        try:
            # In a real implementation, we would check for required packages
            # For this demo, we'll just return True
            return True
        except Exception as e:
            logger.error(f"Shapley explainer validation failed: {str(e)}")
            return False
    
    async def explain(self, data: InputData, **kwargs) -> ExplanationDetails:
        """
        Generate a Shapley values explanation for the input data.
        
        Args:
            data: The input data to explain
            **kwargs: Additional parameters
            
        Returns:
            ExplanationDetails: The explanation details
        """
        start_total = time.time()
        
        # Compute Shapley values
        shapley_values = self._compute_shapley_values(data)
        
        # Compute feature interactions
        feature_interactions = self._compute_feature_interactions(data)
        
        # Create visualizations
        max_features = kwargs.get('max_features', data.max_features_to_show)
        visualizations = self._create_visualizations(
            data, 
            shapley_values, 
            feature_interactions,
            max_features
        )
        
        end_total = time.time()
        self.computation_times["total_explanation_time"] = end_total - start_total
        
        # Create explanation details
        explanation = ExplanationDetails(
            method=self.method,
            model_type=self.model_type,
            feature_importance=shapley_values,
            feature_interactions=feature_interactions,
            visualizations=visualizations
        )
        
        return explanation