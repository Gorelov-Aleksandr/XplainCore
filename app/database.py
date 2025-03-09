"""
Database module for the XAI service.

This module provides database connection and session management
using SQLAlchemy ORM. It establishes connections to PostgreSQL
using environment variables.
"""
import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from loguru import logger

from app.config import settings


# Create database engine
if settings.database_url:
    try:
        # Configure engine with connection pooling
        engine = create_engine(
            settings.database_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=300,  # Recycle connections after 5 minutes
            pool_pre_ping=True,  # Check connection before using
            poolclass=QueuePool
        )
        logger.info("Database engine created successfully")
    except Exception as e:
        logger.error(f"Failed to create database engine: {str(e)}")
        engine = None
else:
    logger.warning("DATABASE_URL not set. Database features will be disabled.")
    engine = None

# Create session factory
if engine:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database session factory created")
else:
    SessionLocal = None
    logger.warning("Database session factory not created")


def get_db() -> Generator:
    """
    Context manager for database sessions.
    
    Provides a transactional scope around a series of operations.
    Automatically handles session creation, committing, and cleanup.
    
    Yields:
        SQLAlchemy session
    """
    if SessionLocal is None:
        logger.warning("Database session requested but database is not configured")
        yield None
        return
    
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()
        logger.debug("Database session closed")


def init_db() -> None:
    """
    Initialize the database.
    
    Creates all tables defined in ORM models.
    """
    from app.models.base import Base
    
    if engine is None:
        logger.warning("Cannot initialize database: No engine")
        return
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")