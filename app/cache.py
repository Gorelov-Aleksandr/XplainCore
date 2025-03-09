import hashlib
from aiocache import Cache, caches
from aiocache.serializers import JsonSerializer
from loguru import logger

from .config import settings

# Initialize in-memory cache instead of Redis for simplicity
# This can be changed to Redis when needed
try:
    # Configure caches
    caches.set_config({
        'default': {
            'cache': "aiocache.SimpleMemoryCache",
            'serializer': {
                'class': "aiocache.serializers.JsonSerializer"
            }
        }
    })
    # Get the default cache
    cache = caches.get('default')
    logger.info("Cache initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize cache: {str(e)}")
    # Fallback to simple memory cache
    cache = Cache(Cache.MEMORY)
    logger.info("Using simple memory cache as fallback")

def custom_cache_key_builder(func, *args, **kwargs):
    """
    Builds a custom cache key based on the input data.
    Uses MD5 hash of the input data JSON.
    
    Args:
        func: The function being cached
        *args, **kwargs: Arguments to the function
        
    Returns:
        str: Cache key for the function call
    """
    if args:
        try:
            input_data = args[0]
            data_str = input_data.json()
            hashed = hashlib.md5(data_str.encode()).hexdigest()
            return f"explain:{hashed}"
        except Exception as e:
            logger.warning(f"Error creating cache key: {str(e)}")
    return "explain:default"
