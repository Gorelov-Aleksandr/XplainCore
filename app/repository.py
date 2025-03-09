"""
Repository module for database operations.

This module provides functions for saving and retrieving explanations
from the database.
"""
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from sqlalchemy.orm import Session
from loguru import logger

from app.models.storage import Explanation, ExplanationMethod, Feature, Visualization
from app.models import ModelType, ExplanationMethod as ExplanationMethodEnum, VisualizationType as VisualizationTypeEnum


class ExplanationRepository:
    """
    Repository for explanation database operations.
    
    Provides methods for saving and retrieving explanations.
    """
    
    @staticmethod
    async def save_explanation(
        db: Session,
        request_id: str,
        input_data: Dict[str, Any],
        prediction: Dict[str, Any],
        confidence: Dict[str, Any],
        explanations: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        computation_time: Dict[str, float],
        version_info: Dict[str, str],
        user_id: Optional[str] = None
    ) -> Explanation:
        """
        Save an explanation to the database.
        
        Args:
            db: Database session
            request_id: Unique request identifier
            input_data: Input data for the explanation
            prediction: Prediction results
            confidence: Confidence metrics
            explanations: List of explanations using different methods
            metadata: Additional metadata
            computation_time: Computation time for different parts
            version_info: Version information
            user_id: Optional user identifier
            
        Returns:
            Explanation: The saved explanation
        """
        if db is None:
            logger.warning("Cannot save explanation: No database connection")
            return None
        
        try:
            # Create the explanation record
            explanation = Explanation(
                request_id=request_id,
                user_id=user_id,
                input_data=input_data,
                prediction_result=prediction,
                prediction_score=prediction.get("score"),
                prediction_decision=prediction.get("decision"),
                confidence_score=confidence.get("overall_score"),
                uncertainty_data=confidence.get("uncertainty_range"),
                total_computation_time=computation_time.get("total"),
                model_type=metadata.get("model_type"),
                model_version=version_info.get("model")
            )
            
            # Add to session
            db.add(explanation)
            db.flush()  # Flush to get the ID without committing
            
            # Add explanations
            for expl in explanations:
                method_name = expl.get("method")
                model_type = expl.get("model_type")
                
                # Create explanation method record
                method = ExplanationMethod(
                    explanation_id=explanation.id,
                    method_name=method_name,
                    model_type=model_type,
                    computation_time=computation_time.get(f"{method_name}_total_explanation_time"),
                    method_data={},
                    decision_rules=expl.get("decision_rules"),
                    feature_interactions=expl.get("feature_interactions"),
                    counterfactuals=expl.get("counterfactuals")
                )
                
                db.add(method)
                db.flush()
                
                # Add feature importance
                if expl.get("feature_importance"):
                    for name, importance in expl.get("feature_importance", {}).items():
                        feature = Feature(
                            explanation_id=explanation.id,
                            method_id=method.id,
                            feature_name=name,
                            feature_importance=importance,
                            # Optional fields from confidence
                            reliability_score=confidence.get("feature_reliability", {}).get(name),
                            statistical_significance=confidence.get("statistical_significance", {}).get(name)
                        )
                        db.add(feature)
                
                # Add visualizations
                for viz in expl.get("visualizations", []):
                    visualization = Visualization(
                        explanation_id=explanation.id,
                        method_id=method.id,
                        type=viz.get("type"),
                        title=viz.get("title"),
                        description=viz.get("description"),
                        data=viz.get("data"),
                        format=viz.get("format", "json")
                    )
                    db.add(visualization)
            
            # Commit the transaction
            db.commit()
            logger.info(f"Saved explanation {request_id} to database")
            
            return explanation
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving explanation to database: {str(e)}")
            return None
    
    @staticmethod
    async def get_explanation_by_request_id(db: Session, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve an explanation by request ID.
        
        Args:
            db: Database session
            request_id: Unique request identifier
            
        Returns:
            Dict: The explanation data, or None if not found
        """
        if db is None:
            logger.warning("Cannot retrieve explanation: No database connection")
            return None
        
        try:
            # Query the explanation
            explanation = db.query(Explanation).filter(Explanation.request_id == request_id).first()
            
            if not explanation:
                logger.warning(f"Explanation with request_id {request_id} not found")
                return None
            
            # Convert to dictionary
            return ExplanationRepository._explanation_to_dict(db, explanation)
        
        except Exception as e:
            logger.error(f"Error retrieving explanation: {str(e)}")
            return None
    
    @staticmethod
    async def get_recent_explanations(
        db: Session, 
        limit: int = 10, 
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent explanations.
        
        Args:
            db: Database session
            limit: Maximum number of explanations to return
            user_id: Optional user identifier to filter by
            
        Returns:
            List[Dict]: List of recent explanations
        """
        if db is None:
            logger.warning("Cannot retrieve explanations: No database connection")
            return []
        
        try:
            # Create base query
            query = db.query(Explanation).order_by(Explanation.timestamp.desc())
            
            # Add user filter if provided
            if user_id:
                query = query.filter(Explanation.user_id == user_id)
            
            # Get results with limit
            explanations = query.limit(limit).all()
            
            # Convert to dictionaries
            return [
                {
                    "request_id": expl.request_id,
                    "timestamp": expl.timestamp.isoformat(),
                    "prediction": expl.prediction_result,
                    "model_type": expl.model_type,
                    "model_version": expl.model_version
                }
                for expl in explanations
            ]
        
        except Exception as e:
            logger.error(f"Error retrieving recent explanations: {str(e)}")
            return []
    
    @staticmethod
    def _explanation_to_dict(db: Session, explanation: Explanation) -> Dict[str, Any]:
        """
        Convert an Explanation ORM object to a dictionary.
        
        Args:
            db: Database session
            explanation: Explanation ORM object
            
        Returns:
            Dict: Dictionary representation of the explanation
        """
        # Query related data
        methods = db.query(ExplanationMethod).filter(
            ExplanationMethod.explanation_id == explanation.id
        ).all()
        
        # Convert methods to dictionaries
        method_dicts = []
        for method in methods:
            # Get features for this method
            features = db.query(Feature).filter(
                Feature.explanation_id == explanation.id,
                Feature.method_id == method.id
            ).all()
            
            # Get visualizations for this method
            visualizations = db.query(Visualization).filter(
                Visualization.explanation_id == explanation.id,
                Visualization.method_id == method.id
            ).all()
            
            # Create feature importance dictionary
            feature_importance = {
                feature.feature_name: feature.feature_importance
                for feature in features
                if feature.feature_importance is not None
            }
            
            # Create visualization dictionaries
            viz_dicts = [
                {
                    "type": viz.type.value,
                    "title": viz.title,
                    "description": viz.description,
                    "data": viz.data,
                    "format": viz.format
                }
                for viz in visualizations
            ]
            
            # Create method dictionary
            method_dict = {
                "method": method.method_name.value,
                "model_type": method.model_type.value if method.model_type else None,
                "feature_importance": feature_importance,
                "visualizations": viz_dicts
            }
            
            # Add optional fields if they exist
            if method.decision_rules:
                method_dict["decision_rules"] = method.decision_rules
            
            if method.feature_interactions:
                method_dict["feature_interactions"] = method.feature_interactions
                
            if method.counterfactuals:
                method_dict["counterfactuals"] = method.counterfactuals
            
            method_dicts.append(method_dict)
        
        # Create the response
        return {
            "request_id": explanation.request_id,
            "timestamp": explanation.timestamp.isoformat(),
            "user_id": explanation.user_id,
            "input_data": explanation.input_data,
            "prediction": explanation.prediction_result,
            "confidence": {
                "overall_score": explanation.confidence_score,
                "uncertainty_range": explanation.uncertainty_data,
                # Feature reliability would need to be reconstructed from features
            },
            "explanations": method_dicts,
            "metadata": {
                "model_type": explanation.model_type,
                "model_version": explanation.model_version,
                "total_computation_time": explanation.total_computation_time
            }
        }