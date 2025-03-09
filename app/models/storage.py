"""
Database models for storing explanations.

This module defines SQLAlchemy ORM models for storing XAI explanations,
methods, features, and visualizations.
"""
import json
import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, 
    DateTime, Text, JSON, ForeignKey, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from app.models.base import Base
from app.models import ModelType, ExplanationMethod as ExplanationMethodEnum, VisualizationType as VisualizationTypeEnum


class Explanation(Base):
    """
    Model for storing explanations.
    
    Each record represents a single explanation request with its results.
    """
    __tablename__ = "explanations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(36), nullable=False, index=True)
    user_id = Column(String(255), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Input data (stored as JSON)
    input_data = Column(JSON, nullable=False)
    
    # Prediction results
    prediction_result = Column(JSON, nullable=False)
    prediction_score = Column(Float, nullable=True)
    prediction_decision = Column(String(255), nullable=True)
    
    # Confidence metrics
    confidence_score = Column(Float, nullable=True)
    uncertainty_data = Column(JSON, nullable=True)
    
    # Computation time
    total_computation_time = Column(Float, nullable=True)
    
    # Model information
    model_type = Column(String(50), nullable=True)
    model_version = Column(String(50), nullable=True)
    
    # Relationships
    methods = relationship("ExplanationMethod", back_populates="explanation", cascade="all, delete-orphan")
    features = relationship("Feature", back_populates="explanation", cascade="all, delete-orphan")
    visualizations = relationship("Visualization", back_populates="explanation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Explanation(request_id='{self.request_id}', timestamp='{self.timestamp}')>"


class ExplanationMethod(Base):
    """
    Model for storing explanation methods.
    
    Each record represents a method used to generate an explanation.
    """
    __tablename__ = "explanation_methods"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    explanation_id = Column(UUID(as_uuid=True), ForeignKey("explanations.id"), nullable=False)
    
    # Method information
    method_name = Column(SQLEnum(ExplanationMethodEnum, name="explanation_method_enum"), nullable=False)
    model_type = Column(SQLEnum(ModelType, name="model_type_enum"), nullable=True)
    computation_time = Column(Float, nullable=True)
    
    # Method-specific data
    method_data = Column(JSON, nullable=True)
    
    # Decision rules (for rule-based explainers)
    decision_rules = Column(JSON, nullable=True)
    
    # Feature interactions (for Shapley explainer)
    feature_interactions = Column(JSON, nullable=True)
    
    # Counterfactuals (for counterfactual explainer)
    counterfactuals = Column(JSON, nullable=True)
    
    # Relationships
    explanation = relationship("Explanation", back_populates="methods")
    visualizations = relationship("Visualization", back_populates="method")
    
    def __repr__(self):
        return f"<ExplanationMethod(method='{self.method_name}')>"


class Feature(Base):
    """
    Model for storing feature importance.
    
    Each record represents the importance of a feature in an explanation.
    """
    __tablename__ = "features"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    explanation_id = Column(UUID(as_uuid=True), ForeignKey("explanations.id"), nullable=False)
    method_id = Column(UUID(as_uuid=True), ForeignKey("explanation_methods.id"), nullable=True)
    
    # Feature information
    feature_name = Column(String(255), nullable=False)
    feature_value = Column(Float, nullable=True)
    feature_importance = Column(Float, nullable=True)
    feature_type = Column(String(50), nullable=True)
    
    # Additional feature metrics
    reliability_score = Column(Float, nullable=True)
    statistical_significance = Column(Float, nullable=True)
    
    # Relationships
    explanation = relationship("Explanation", back_populates="features")
    
    def __repr__(self):
        return f"<Feature(name='{self.feature_name}', importance={self.feature_importance})>"


class Visualization(Base):
    """
    Model for storing visualizations.
    
    Each record represents a visualization generated for an explanation.
    """
    __tablename__ = "visualizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    explanation_id = Column(UUID(as_uuid=True), ForeignKey("explanations.id"), nullable=False)
    method_id = Column(UUID(as_uuid=True), ForeignKey("explanation_methods.id"), nullable=True)
    
    # Visualization information
    type = Column(SQLEnum(VisualizationTypeEnum, name="visualization_type_enum"), nullable=False)
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    
    # Visualization data
    data = Column(JSON, nullable=False)
    format = Column(String(50), default="json")
    
    # Relationships
    explanation = relationship("Explanation", back_populates="visualizations")
    method = relationship("ExplanationMethod", back_populates="visualizations")
    
    def __repr__(self):
        return f"<Visualization(type='{self.type}', title='{self.title}')>"