"""
Base SQLAlchemy module for the XAI service.

This module defines the Base class for all SQLAlchemy models
to avoid circular imports between database.py and models.
"""
import os
from sqlalchemy.ext.declarative import declarative_base

# Create declarative base for all models
Base = declarative_base()