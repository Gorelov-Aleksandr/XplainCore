"""
Models module for the XAI service.
This module contains the data models for the API.
"""
from enum import Enum

# Define enums here for use in storage and other modules
class ModelType(str, Enum):
    """Supported model types for explainability"""
    RULE_BASED = "rule_based"
    TABULAR_ML = "tabular_ml"
    DEEP_LEARNING = "deep_learning"
    TREE_BASED = "tree_based"
    LANGUAGE_MODEL = "language_model"
    MULTIMODAL = "multimodal"

class ExplanationMethod(str, Enum):
    """Available explanation methods"""
    FEATURE_IMPORTANCE = "feature_importance"
    SHAPLEY = "shapley_values"
    LIME = "lime"
    COUNTERFACTUAL = "counterfactual"
    ANCHORS = "anchors"
    GRADIENT_BASED = "gradient_based"
    ATTENTION_VISUALIZATION = "attention_visualization"
    DECISION_TREE = "decision_tree_approximation"
    EXAMPLES = "example_based"
    FAIRNESS_METRICS = "fairness_metrics"
    UNCERTAINTY = "uncertainty_quantification"

class VisualizationType(str, Enum):
    """Available visualization types"""
    BAR_CHART = "bar_chart"
    WATERFALL = "waterfall_plot"
    HEATMAP = "heatmap"
    FORCE_PLOT = "force_plot"
    DECISION_PLOT = "decision_plot"
    PDPBOX = "partial_dependence_plot"
    ATTENTION_MAP = "attention_map"
    TREE_VISUAL = "tree_visualization"
    CONTRASTIVE = "contrastive_explanation"
    RADAR_CHART = "radar_chart"
    TEXT_HIGHLIGHT = "text_highlight"
    CONFIDENCE_INTERVAL = "confidence_interval"