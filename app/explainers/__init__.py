"""
Explainers module for XAI service.
This module contains implementations of various model explanation methods.
"""

from .base import BaseExplainer
from .feature_importance import FeatureImportanceExplainer
from .shapley import ShapleyExplainer
from .counterfactual import CounterfactualExplainer
from .lime_explainer import LimeExplainer

# Import future explainers once implemented
# from .uncertainty import UncertaintyExplainer
# from .fairness import FairnessExplainer
# from .visualizer import ExplanationVisualizer

__all__ = [
    'BaseExplainer',
    'FeatureImportanceExplainer',
    'ShapleyExplainer',
    'CounterfactualExplainer',
    'LimeExplainer',
]