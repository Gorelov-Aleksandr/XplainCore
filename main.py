import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
try:
    from loguru import logger
except ImportError:
    import logging as logger

# Define simple root endpoint
def create_app():
    # Create FastAPI application
    app = FastAPI(
        title="XAI Service",
        description="Explainable AI service with Yandex OAuth integration",
        version="1.0.0",
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
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
            "status": "operational"
        }
    
    # Health check endpoint
    @app.get("/health", tags=["monitoring"])
    async def health():
        """
        Health check endpoint for the service.
        Used for monitoring and load balancers.
        """
        return {"status": "OK", "message": "Service is healthy"}
    
    return app

app = create_app()

if __name__ == "__main__":
    print("Starting XAI Service...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
