"""
LIME explainer implementation.

LIME (Local Interpretable Model-agnostic Explanations) approximates the model
locally around the prediction by fitting a simple, interpretable model to 
perturbed versions of the input data.
"""
import time
import random
import json
import math
import logging
from typing import Dict, Any, List, Optional, Tuple

# Using standard logging instead of loguru for better compatibility
logger = logging.getLogger(__name__)

from app.models import ModelType, ExplanationMethod, VisualizationType
from app.models.schema import InputData, ExplanationDetails, VisualizationData
from .base import BaseExplainer


class LimeExplainer(BaseExplainer):
    """
    Implementation of LIME explanation method.
    This provides local model-agnostic explanations by perturbing the input
    and learning a simple interpretable model around the prediction.
    """
    def __init__(self, model_type: ModelType = ModelType.TABULAR_ML):
        """
        Initialize the LIME explainer.
        
        Args:
            model_type: Type of model being explained
        """
        super().__init__(model_type=model_type, method=ExplanationMethod.LIME)
        self.num_samples = 1000  # Number of perturbations to generate
        logger.info("Initialized LIME explainer")
    
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
    
    def _generate_perturbations(self, data: InputData) -> List[Dict[str, Any]]:
        """
        Generate perturbed versions of the input data for LIME.
        
        Args:
            data: Input data
            
        Returns:
            List[Dict]: List of perturbed feature sets
        """
        # Extract features from input data
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
        
        # Define perturbation ranges for each feature (reasonable values)
        ranges = {
            "income": (data.income * 0.5, data.income * 1.5),
            "loan_amount": (data.loan_amount * 0.5, data.loan_amount * 1.5),
            "credit_history": (max(0, data.credit_history - 3), min(10, data.credit_history + 3))
        }
        
        # Add ranges for optional features
        if data.employment_years is not None:
            ranges["employment_years"] = (max(0, data.employment_years - 2), data.employment_years + 2)
        if data.debt_to_income_ratio is not None:
            ranges["debt_to_income_ratio"] = (max(0, data.debt_to_income_ratio - 0.2), min(1, data.debt_to_income_ratio + 0.2))
        if data.age is not None:
            ranges["age"] = (max(18, data.age - 10), data.age + 10)
        if data.previous_defaults is not None:
            ranges["previous_defaults"] = (max(0, data.previous_defaults - 1), data.previous_defaults + 1)
        
        # Generate perturbations
        perturbations = []
        for _ in range(self.num_samples):
            perturbed = {}
            for feature, (min_val, max_val) in ranges.items():
                # Generate perturbed value
                if feature == "credit_history" or feature == "previous_defaults" or feature == "age":
                    # Integer features
                    perturbed[feature] = random.randint(int(min_val), int(max_val))
                else:
                    # Continuous features
                    perturbed[feature] = random.uniform(min_val, max_val)
            
            perturbations.append(perturbed)
        
        return perturbations
    
    def _compute_lime_explanations(self, data: InputData) -> Dict[str, float]:
        """
        Compute LIME feature importance scores by fitting a linear model to
        perturbed inputs around the instance.
        
        Args:
            data: Input data
            
        Returns:
            Dict: LIME feature importance scores
        """
        # Get perturbations
        perturbations = self._generate_perturbations(data)
        
        # Get predictions for each perturbation
        predictions = [self._make_prediction(p) for p in perturbations]
        
        # Original prediction
        original_features = {
            "income": data.income,
            "loan_amount": data.loan_amount,
            "credit_history": data.credit_history
        }
        
        # Add optional features if provided
        if data.employment_years is not None:
            original_features["employment_years"] = data.employment_years
        if data.debt_to_income_ratio is not None:
            original_features["debt_to_income_ratio"] = data.debt_to_income_ratio
        if data.age is not None:
            original_features["age"] = data.age
        if data.previous_defaults is not None:
            original_features["previous_defaults"] = data.previous_defaults
        
        original_prediction = self._make_prediction(original_features)
        
        # Calculate distances from original instance
        def calculate_distance(perturbed):
            # Normalize and calculate Euclidean distance
            sum_squared = 0
            for feature in original_features:
                # Normalize by feature range to make distances comparable
                if feature == "income":
                    norm_orig = original_features[feature] / 100000
                    norm_pert = perturbed[feature] / 100000
                elif feature == "loan_amount":
                    norm_orig = original_features[feature] / 50000
                    norm_pert = perturbed[feature] / 50000
                elif feature == "credit_history":
                    norm_orig = original_features[feature] / 10
                    norm_pert = perturbed[feature] / 10
                elif feature == "employment_years":
                    norm_orig = original_features[feature] / 10
                    norm_pert = perturbed[feature] / 10
                elif feature == "debt_to_income_ratio":
                    norm_orig = original_features[feature]
                    norm_pert = perturbed[feature]
                elif feature == "age":
                    norm_orig = original_features[feature] / 100
                    norm_pert = perturbed[feature] / 100
                elif feature == "previous_defaults":
                    norm_orig = original_features[feature] / 5
                    norm_pert = perturbed[feature] / 5
                else:
                    # Default normalization
                    norm_orig = original_features[feature]
                    norm_pert = perturbed[feature]
                
                sum_squared += (norm_orig - norm_pert) ** 2
            
            return math.sqrt(sum_squared)
        
        distances = [calculate_distance(p) for p in perturbations]
        
        # Calculate weights (kernel) - points closer to the original instance get higher weights
        max_distance = max(distances) if distances else 1
        weights = [math.exp(-d / max_distance) for d in distances]
        
        # Fit a weighted linear model
        # Since we don't have sklearn, we'll approximate the coefficients using a simplified approach
        # For a real implementation, we would use sklearn.linear_model.Ridge with sample_weight=weights
        
        # Find feature importance for each feature
        importance = {}
        for feature in original_features:
            # Calculate correlation between each feature and the predictions, weighted by distance
            feature_values = [p[feature] for p in perturbations]
            
            # Normalize feature values for more accurate calculations
            if feature == "income":
                feature_values = [v / 100000 for v in feature_values]
            elif feature == "loan_amount":
                feature_values = [v / 50000 for v in feature_values]
            
            # Calculate weighted correlation
            weighted_mean_x = sum(x * w for x, w in zip(feature_values, weights)) / sum(weights)
            weighted_mean_y = sum(y * w for y, w in zip(predictions, weights)) / sum(weights)
            
            weighted_cov = sum(w * (x - weighted_mean_x) * (y - weighted_mean_y) 
                              for x, y, w in zip(feature_values, predictions, weights)) / sum(weights)
            
            weighted_var_x = sum(w * (x - weighted_mean_x) ** 2 
                               for x, w in zip(feature_values, weights)) / sum(weights)
            
            weighted_var_y = sum(w * (y - weighted_mean_y) ** 2 
                               for y, w in zip(predictions, weights)) / sum(weights)
            
            # To avoid division by zero
            if weighted_var_x > 0 and weighted_var_y > 0:
                weighted_corr = weighted_cov / (math.sqrt(weighted_var_x) * math.sqrt(weighted_var_y))
            else:
                weighted_corr = 0
            
            # Coefficient is proportional to correlation and feature variability
            coefficient = weighted_corr * math.sqrt(weighted_var_x / weighted_var_y) if weighted_var_y > 0 else 0
            
            # Calculate coefficient magnitude (absolute value)
            importance[feature] = abs(coefficient)
        
        # Normalize importance values to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def _create_visualizations(self, 
                              data: InputData, 
                              lime_importance: Dict[str, float],
                              max_features: int = 5) -> List[VisualizationData]:
        """
        Create visualizations for LIME explanations.
        
        Args:
            data: Input data
            lime_importance: LIME feature importance scores
            max_features: Maximum number of features to include
            
        Returns:
            List[VisualizationData]: List of visualization data
        """
        visualizations = []
        
        # Sort features by importance
        sorted_features = sorted(lime_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:max_features]
        
        # Bar chart visualization
        bar_chart_data = {
            "labels": [item[0] for item in sorted_features],
            "values": [item[1] for item in sorted_features],
            "colors": ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#FF6D01"][:len(sorted_features)],
            "title": "LIME Feature Importance",
            "subtitle": "Local feature importance around this prediction"
        }
        
        bar_chart = VisualizationData(
            type=VisualizationType.BAR_CHART,
            title="LIME Feature Importance",
            description="Local explanation of feature importance around this prediction",
            data=bar_chart_data,
            format="json"
        )
        visualizations.append(bar_chart)
        
        # Decision plot visualization - shows how features contribute to the prediction
        # Extract original features for display
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
        
        # Format feature values for display
        display_features = {}
        for feature, value in features.items():
            if feature in ["income", "loan_amount"]:
                display_features[feature] = f"${value:,.2f}"
            elif feature == "credit_history":
                display_features[feature] = f"{value}/10"
            elif feature == "employment_years":
                display_features[feature] = f"{value} years"
            elif feature == "debt_to_income_ratio":
                display_features[feature] = f"{value:.1%}"
            else:
                display_features[feature] = str(value)
        
        # Create a decision plot data structure
        decision_plot_data = {
            "features": [item[0] for item in sorted_features],
            "importance": [item[1] for item in sorted_features],
            "feature_values": display_features,
            "base_value": 0.5,  # Base prediction value
            "prediction": self._make_prediction(features)
        }
        
        decision_plot = VisualizationData(
            type=VisualizationType.DECISION_PLOT,
            title="LIME Decision Plot",
            description="How each feature contributes to the model prediction",
            data=decision_plot_data,
            format="json"
        )
        visualizations.append(decision_plot)
        
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
        # Confidence for LIME is based on the quality of the linear fit
        # In a real implementation, we would use the model's R-squared value
        
        # Calculate feature importance
        importance = self._compute_lime_explanations(data)
        
        # For this demo, we'll synthesize confidence metrics
        # In reality, these would come from the LIME model's fit quality
        
        # Feature reliability is related to the stability of the LIME model
        feature_reliability = {}
        for feature, imp in importance.items():
            # Higher importance generally means more reliable explanation
            # Add some randomness to simulate real-world variance
            reliability = 0.6 + (imp * 0.3) + random.uniform(-0.05, 0.05)
            feature_reliability[feature] = round(min(0.95, reliability), 2)
        
        # Overall confidence is a combination of reliability factors
        overall_score = 0.7  # LIME generally has moderate confidence
        
        # Higher score for models with clear feature importances
        importance_distribution = list(importance.values())
        # If one feature is much more important than others, confidence is higher
        importance_max = max(importance_distribution) if importance_distribution else 0
        if importance_max > 0.4:  # One feature dominates
            overall_score += 0.1
        
        # Generate uncertainty ranges for key features
        uncertainty_range = {}
        for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]:
            # More important features have narrower uncertainty ranges
            width = 0.3 * (1 - imp)  # Larger importance values have smaller uncertainty
            uncertainty_range[feature] = [-width/2, width/2]  # Centered around zero as these are coefficients
        
        # Statistical significance (p-values) - in real system these would be computed
        statistical_significance = {}
        for feature, imp in importance.items():
            # More important features typically have higher statistical significance
            p_value = max(0.001, 0.1 * (1 - imp))
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
        # For LIME explainer, just check that it's properly initialized
        try:
            # Ensure we have the minimum requirements 
            if self.num_samples < 100:
                logger.warning("LIME explainer has too few samples, performance may be poor")
                return False
                
            return True
        except Exception as e:
            logger.error(f"LIME explainer validation failed: {str(e)}")
            return False
    
    async def explain(self, data: InputData, **kwargs) -> ExplanationDetails:
        """
        Generate a LIME explanation for the input data.
        
        Args:
            data: The input data to explain
            **kwargs: Additional parameters
            
        Returns:
            ExplanationDetails: The explanation details
        """
        start_total = time.time()
        
        # Compute LIME explanations
        lime_importance = self._compute_lime_explanations(data)
        
        # Create visualizations
        max_features = kwargs.get('max_features', data.max_features_to_show)
        visualizations = self._create_visualizations(
            data, 
            lime_importance,
            max_features
        )
        
        end_total = time.time()
        self.computation_times["total_explanation_time"] = end_total - start_total
        
        # Create explanation details
        explanation = ExplanationDetails(
            method=self.method,
            model_type=self.model_type,
            feature_importance=lime_importance,
            # LIME doesn't create these outputs
            feature_interactions=None,
            counterfactuals=None,
            decision_rules=None,
            
            # Local explanation data specific to LIME
            local_explanation={
                "num_samples": self.num_samples,
                "kernel_width": "adaptive",  # In a real implementation, this would be configurable
                "feature_selection": "auto"  # In a real implementation, this would be configurable
            },
            
            visualizations=visualizations
        )
        
        return explanation