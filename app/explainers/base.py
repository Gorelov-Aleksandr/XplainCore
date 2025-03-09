"""
Base explainer class that all explainers should inherit from.
"""
import time
import functools
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
import logging

# Using standard logging instead of loguru for better compatibility
logger = logging.getLogger(__name__)

from app.models import ModelType, ExplanationMethod
from app.models.schema import InputData, ExplanationDetails

class BaseExplainer(ABC):
    """
    Abstract base class for all explainers.
    Provides common functionality and interface that all explainers must implement.
    """
    def __init__(self, model_type: ModelType, method: ExplanationMethod):
        """
        Initialize the explainer.
        
        Args:
            model_type: Type of model being explained
            method: Explanation method used
        """
        self.model_type = model_type
        self.method = method
        self.computation_times: Dict[str, float] = {}
        logger.info(f"Initialized {self.__class__.__name__} explainer")
    
    def measure_time(self, func_name: str) -> Callable:
        """
        Decorator to measure computation time of a function.
        
        Args:
            func_name: Name of the function being measured
            
        Returns:
            Callable: Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                self.computation_times[func_name] = end_time - start_time
                logger.debug(f"{func_name} took {end_time - start_time:.4f} seconds")
                return result
            return wrapper
        return decorator
    
    @abstractmethod
    async def explain(self, data: InputData, **kwargs) -> ExplanationDetails:
        """
        Generate an explanation for the given input data.
        
        Args:
            data: The input data to explain
            **kwargs: Additional parameters specific to the explainer
            
        Returns:
            ExplanationDetails: The explanation details
        """
        pass
    
    @abstractmethod
    async def get_confidence(self, data: InputData, **kwargs) -> Dict[str, Any]:
        """
        Calculate confidence metrics for the explanation.
        
        Args:
            data: The input data
            **kwargs: Additional parameters specific to the explainer
            
        Returns:
            Dict: Confidence metrics
        """
        pass
    
    def get_computation_times(self) -> Dict[str, float]:
        """
        Get the computation times for different parts of the explanation.
        
        Returns:
            Dict: Computation times in seconds
        """
        return self.computation_times
    
    @abstractmethod
    async def validate(self) -> bool:
        """
        Validate that the explainer is properly configured.
        
        Returns:
            bool: True if valid, False otherwise
        """
        pass