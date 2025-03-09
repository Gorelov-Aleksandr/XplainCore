import uvicorn
from fastapi import FastAPI, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging
    logger.basicConfig(level=logging.INFO)

from app.config import settings
from app.monitoring import setup_monitoring
from app.middleware import rate_limiting_middleware, observability_middleware
from app.routers import explain, health, history
from app.database import init_db

# Configure logger
logger.info("Starting XAI Service...")

# Initialize monitoring with error handling
try:
    setup_monitoring()
    logger.info("Monitoring setup completed")
except Exception as e:
    logger.error(f"Error during monitoring setup: {str(e)}")
    logger.info("Continuing without full monitoring setup")

# Create FastAPI application
app = FastAPI(
    title="XAI Service",
    description="Explainable AI service with Yandex OAuth integration",
    version="1.0.0",
    docs_url=None,  # Disable default docs to use custom documentation
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middlewares
app.middleware("http")(rate_limiting_middleware)
app.middleware("http")(observability_middleware)

# Initialize database
try:
    init_db()
    logger.info("Database initialized successfully")
except Exception as e:
    logger.error(f"Error initializing database: {str(e)}")
    logger.warning("Continuing without database initialization")

# Include routers
app.include_router(health.router)  # Health check endpoint
app.include_router(explain.router)  # XAI explanations endpoint
app.include_router(history.router)  # History endpoints

# Custom Swagger UI endpoint with better styling
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
    )

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint to verify the service is running.
    """
    return {
        "service": "XAI Service",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    logger.info(f"Starting server at 0.0.0.0:5000")
    uvicorn.run(app, host="0.0.0.0", port=5000)
