"""
Counterfactual explainer implementation.

Counterfactual explanations show the minimal changes needed in the input 
to achieve a different outcome, helping users understand "what if" scenarios.
"""
import time
import random
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

# Using standard logging instead of loguru for better compatibility
logger = logging.getLogger(__name__)

from ..models import ModelType, ExplanationMethod, InputData, ExplanationDetails, VisualizationType, VisualizationData
from .base import BaseExplainer

class CounterfactualExplainer(BaseExplainer):
    """
    Implementation of counterfactual explanation method.
    This provides "what if" scenarios by showing minimal changes needed to flip the model's decision.
    """
    def __init__(self, model_type: ModelType = ModelType.TABULAR_ML):
        """
        Initialize the counterfactual explainer.
        
        Args:
            model_type: Type of model being explained
        """
        super().__init__(model_type=model_type, method=ExplanationMethod.COUNTERFACTUAL)
    
    def _make_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction based on the features.
        This is a simplified model for demo purposes.
        
        Args:
            features: Feature values
            
        Returns:
            Dict: Prediction results
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
            
        # Determine decision
        decision = "APPROVED" if score >= 0.6 else "DENIED"
        
        # Ensure score is between 0 and 1
        score = max(0, min(1, score))
        
        return {
            "decision": decision,
            "score": score,
            "loan_to_income_ratio": loan_to_income_ratio,
            "credit_factor": credit_factor
        }
    
    def _generate_counterfactuals(self, data: InputData) -> List[Dict[str, Any]]:
        """
        Generate counterfactual examples by finding minimal changes to flip the decision.
        
        Args:
            data: Input data
            
        Returns:
            List[Dict]: List of counterfactual examples
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
        
        # Get the current prediction
        original_prediction = self._make_prediction(features)
        original_decision = original_prediction["decision"]
        target_decision = "APPROVED" if original_decision == "DENIED" else "DENIED"
        
        counterfactuals = []
        
        # Generate counterfactuals for key features
        if original_decision == "DENIED":
            # Try to find ways to get APPROVED
            
            # 1. Increase income
            if features["income"] < 100000:
                cf_features = features.copy()
                income_increase_factor = 1.3  # 30% income increase
                cf_features["income"] = features["income"] * income_increase_factor
                cf_prediction = self._make_prediction(cf_features)
                
                if cf_prediction["decision"] == "APPROVED":
                    counterfactuals.append({
                        "features": cf_features,
                        "prediction": cf_prediction,
                        "changes": [{
                            "feature": "income",
                            "original": features["income"],
                            "counterfactual": cf_features["income"],
                            "change_percent": round((income_increase_factor - 1) * 100, 1)
                        }],
                        "explanation": f"Increasing income to ${cf_features['income']:,.2f} would lead to approval"
                    })
            
            # 2. Decrease loan amount
            if features["loan_amount"] > 5000:
                cf_features = features.copy()
                loan_decrease_factor = 0.7  # 30% loan decrease
                cf_features["loan_amount"] = features["loan_amount"] * loan_decrease_factor
                cf_prediction = self._make_prediction(cf_features)
                
                if cf_prediction["decision"] == "APPROVED":
                    counterfactuals.append({
                        "features": cf_features,
                        "prediction": cf_prediction,
                        "changes": [{
                            "feature": "loan_amount",
                            "original": features["loan_amount"],
                            "counterfactual": cf_features["loan_amount"],
                            "change_percent": round((loan_decrease_factor - 1) * 100, 1)
                        }],
                        "explanation": f"Reducing loan amount to ${cf_features['loan_amount']:,.2f} would lead to approval"
                    })
            
            # 3. Improve credit history
            if features["credit_history"] < 9:
                cf_features = features.copy()
                cf_features["credit_history"] = min(10, features["credit_history"] + 2)
                cf_prediction = self._make_prediction(cf_features)
                
                if cf_prediction["decision"] == "APPROVED":
                    counterfactuals.append({
                        "features": cf_features,
                        "prediction": cf_prediction,
                        "changes": [{
                            "feature": "credit_history",
                            "original": features["credit_history"],
                            "counterfactual": cf_features["credit_history"],
                            "change_percent": None  # Not applicable for credit score
                        }],
                        "explanation": f"Improving credit history to {cf_features['credit_history']} would lead to approval"
                    })
            
            # 4. Combined: Moderate improvement in multiple factors
            cf_features = features.copy()
            changes = []
            
            # Improve income by 10%
            cf_features["income"] = features["income"] * 1.1
            changes.append({
                "feature": "income",
                "original": features["income"],
                "counterfactual": cf_features["income"],
                "change_percent": 10.0
            })
            
            # Reduce loan by 10%
            cf_features["loan_amount"] = features["loan_amount"] * 0.9
            changes.append({
                "feature": "loan_amount",
                "original": features["loan_amount"],
                "counterfactual": cf_features["loan_amount"],
                "change_percent": -10.0
            })
            
            # Improve credit slightly
            if features["credit_history"] < 10:
                cf_features["credit_history"] = min(10, features["credit_history"] + 1)
                changes.append({
                    "feature": "credit_history",
                    "original": features["credit_history"],
                    "counterfactual": cf_features["credit_history"],
                    "change_percent": None
                })
            
            cf_prediction = self._make_prediction(cf_features)
            if cf_prediction["decision"] == "APPROVED":
                counterfactuals.append({
                    "features": cf_features,
                    "prediction": cf_prediction,
                    "changes": changes,
                    "explanation": "Multiple small improvements would lead to approval"
                })
        else:
            # Original decision is APPROVED, find ways to get DENIED
            
            # 1. Decrease income
            if features["income"] > 30000:
                cf_features = features.copy()
                income_decrease_factor = 0.7  # 30% income decrease
                cf_features["income"] = features["income"] * income_decrease_factor
                cf_prediction = self._make_prediction(cf_features)
                
                if cf_prediction["decision"] == "DENIED":
                    counterfactuals.append({
                        "features": cf_features,
                        "prediction": cf_prediction,
                        "changes": [{
                            "feature": "income",
                            "original": features["income"],
                            "counterfactual": cf_features["income"],
                            "change_percent": round((income_decrease_factor - 1) * 100, 1)
                        }],
                        "explanation": f"Decreasing income to ${cf_features['income']:,.2f} would lead to denial"
                    })
            
            # 2. Increase loan amount
            cf_features = features.copy()
            loan_increase_factor = 1.4  # 40% loan increase
            cf_features["loan_amount"] = features["loan_amount"] * loan_increase_factor
            cf_prediction = self._make_prediction(cf_features)
            
            if cf_prediction["decision"] == "DENIED":
                counterfactuals.append({
                    "features": cf_features,
                    "prediction": cf_prediction,
                    "changes": [{
                        "feature": "loan_amount",
                        "original": features["loan_amount"],
                        "counterfactual": cf_features["loan_amount"],
                        "change_percent": round((loan_increase_factor - 1) * 100, 1)
                    }],
                    "explanation": f"Increasing loan amount to ${cf_features['loan_amount']:,.2f} would lead to denial"
                })
            
            # 3. Worsen credit history
            if features["credit_history"] > 2:
                cf_features = features.copy()
                cf_features["credit_history"] = max(0, features["credit_history"] - 3)
                cf_prediction = self._make_prediction(cf_features)
                
                if cf_prediction["decision"] == "DENIED":
                    counterfactuals.append({
                        "features": cf_features,
                        "prediction": cf_prediction,
                        "changes": [{
                            "feature": "credit_history",
                            "original": features["credit_history"],
                            "counterfactual": cf_features["credit_history"],
                            "change_percent": None
                        }],
                        "explanation": f"Decreasing credit history to {cf_features['credit_history']} would lead to denial"
                    })
        
        # Sort counterfactuals by complexity (fewer changes first)
        counterfactuals.sort(key=lambda cf: len(cf["changes"]))
        
        return counterfactuals[:3]  # Return top 3 counterfactuals
    
    def _create_visualizations(self, 
                              data: InputData, 
                              counterfactuals: List[Dict[str, Any]]) -> List[VisualizationData]:
        """
        Create visualizations for counterfactual explanations.
        
        Args:
            data: Input data
            counterfactuals: List of counterfactual examples
            
        Returns:
            List[VisualizationData]: List of visualization data
        """
        visualizations = []
        
        if not counterfactuals:
            return visualizations
        
        # Bar chart comparing original vs counterfactual values
        for i, cf in enumerate(counterfactuals):
            changes = cf["changes"]
            feature_names = [change["feature"] for change in changes]
            
            original_values = [change["original"] for change in changes]
            cf_values = [change["counterfactual"] for change in changes]
            
            # Normalize values for display
            max_vals = [max(o, c) for o, c in zip(original_values, cf_values)]
            original_norm = [o/m if m > 0 else 0 for o, m in zip(original_values, max_vals)]
            cf_norm = [c/m if m > 0 else 0 for c, m in zip(cf_values, max_vals)]
            
            contrastive_data = {
                "title": f"Counterfactual #{i+1}: {cf['explanation']}",
                "features": feature_names,
                "original": {
                    "values": original_values,
                    "normalized": original_norm,
                    "decision": cf["prediction"]["decision"] == "APPROVED"  # True if approved
                },
                "counterfactual": {
                    "values": cf_values,
                    "normalized": cf_norm,
                    "decision": cf["prediction"]["decision"] != "APPROVED"  # Opposite of original
                },
                "colors": {
                    "original": "#4285F4",  # Blue
                    "counterfactual": "#EA4335"  # Red
                }
            }
            
            vis = VisualizationData(
                type=VisualizationType.CONTRASTIVE,
                title=f"Counterfactual Example #{i+1}",
                description=cf["explanation"],
                data=contrastive_data,
                format="json"
            )
            visualizations.append(vis)
        
        # Add a radar chart comparing all features between original and best counterfactual
        if counterfactuals:
            best_cf = counterfactuals[0]  # First counterfactual (simplest change)
            
            # Extract features from original and counterfactual
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
            
            # Extract the same features from counterfactual
            cf_features = best_cf["features"]
            
            # Select features to include in radar chart
            radar_features = ["income", "loan_amount", "credit_history"]
            if "employment_years" in features and "employment_years" in cf_features:
                radar_features.append("employment_years")
            if "debt_to_income_ratio" in features and "debt_to_income_ratio" in cf_features:
                radar_features.append("debt_to_income_ratio")
            
            # Normalize values for radar chart (0-1 scale)
            # Use realistic max values for normalization
            max_values = {
                "income": 150000,
                "loan_amount": 100000,
                "credit_history": 10,
                "employment_years": 20,
                "debt_to_income_ratio": 1.0
            }
            
            original_radar = [min(1.0, features.get(f, 0) / max_values.get(f, 1)) for f in radar_features]
            cf_radar = [min(1.0, cf_features.get(f, 0) / max_values.get(f, 1)) for f in radar_features]
            
            radar_data = {
                "features": radar_features,
                "original": {
                    "values": original_radar,
                    "label": "Current Application"
                },
                "counterfactual": {
                    "values": cf_radar,
                    "label": "Alternative Scenario"
                }
            }
            
            radar_chart = VisualizationData(
                type=VisualizationType.RADAR_CHART,
                title="Feature Comparison: Current vs Alternative",
                description="Comparison between your current application and an alternative scenario with different outcome",
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
        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(data)
        
        # Confidence depends on how easy it is to find counterfactuals
        # If we find multiple counterfactuals with small changes, confidence is lower
        # because the decision boundary is close
        
        # Calculate overall score based on counterfactual distances
        if not counterfactuals:
            overall_score = 0.95  # High confidence when no counterfactuals found
        else:
            # Calculate average percentage change across all counterfactuals
            total_change_pct = 0
            count = 0
            
            for cf in counterfactuals:
                for change in cf["changes"]:
                    if change["change_percent"] is not None:
                        total_change_pct += abs(change["change_percent"])
                        count += 1
            
            avg_change_pct = total_change_pct / count if count > 0 else 30
            
            # Higher changes mean more confidence in the original decision
            # Map from 0-50% change range to 0.5-0.9 confidence
            overall_score = min(0.9, max(0.5, 0.5 + avg_change_pct / 125))
        
        # Feature reliability varies based on their importance in counterfactuals
        feature_reliability = {
            "income": 0.8,
            "loan_amount": 0.85,
            "credit_history": 0.9
        }
        
        # Update reliability based on counterfactual changes
        if counterfactuals:
            feature_counts = {}
            
            for cf in counterfactuals:
                for change in cf["changes"]:
                    feature = change["feature"]
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            # Features that appear more often in counterfactuals have lower reliability
            for feature, count in feature_counts.items():
                if feature in feature_reliability:
                    # Reduce reliability for frequently changed features
                    feature_reliability[feature] = max(0.7, feature_reliability[feature] - count * 0.05)
        
        # Calculate uncertainty ranges
        # First, get feature values
        features = {
            "income": data.income,
            "loan_amount": data.loan_amount,
            "credit_history": data.credit_history
        }
        
        # For each feature, define uncertainty range based on stability
        uncertainty_range = {}
        for feature, value in features.items():
            if feature == "income":
                # Income has moderate uncertainty
                uncertainty_range[feature] = [value * 0.95, value * 1.05]
            elif feature == "loan_amount":
                # Loan amount has low uncertainty (fixed value)
                uncertainty_range[feature] = [value * 0.99, value * 1.01]
            elif feature == "credit_history":
                # Credit history has low uncertainty (discrete value)
                uncertainty_range[feature] = [max(0, value - 0.5), min(10, value + 0.5)]
        
        # Statistical significance based on feature importance in counterfactuals
        statistical_significance = {
            "income": 0.01,
            "loan_amount": 0.01,
            "credit_history": 0.01
        }
        
        if counterfactuals:
            # Features that change more in counterfactuals have higher significance
            for cf in counterfactuals:
                for change in cf["changes"]:
                    feature = change["feature"]
                    if feature in statistical_significance:
                        # Lower p-value (higher significance) for important features
                        statistical_significance[feature] = max(0.001, statistical_significance[feature] * 0.8)
        
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
        # For counterfactual explainer, just check that required dependencies are available
        try:
            # In a real implementation, we would check for required packages
            # For this demo, we'll just return True
            return True
        except Exception as e:
            logger.error(f"Counterfactual explainer validation failed: {str(e)}")
            return False
    
    async def explain(self, data: InputData, **kwargs) -> ExplanationDetails:
        """
        Generate a counterfactual explanation for the input data.
        
        Args:
            data: The input data to explain
            **kwargs: Additional parameters
            
        Returns:
            ExplanationDetails: The explanation details
        """
        start_total = time.time()
        
        # Generate counterfactuals
        counterfactuals = self._generate_counterfactuals(data)
        
        # Create visualizations
        visualizations = self._create_visualizations(data, counterfactuals)
        
        end_total = time.time()
        self.computation_times["total_explanation_time"] = end_total - start_total
        
        # Create explanation details
        explanation = ExplanationDetails(
            method=self.method,
            model_type=self.model_type,
            counterfactuals=counterfactuals,
            visualizations=visualizations
        )
        
        return explanation