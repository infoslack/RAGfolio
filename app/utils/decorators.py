import functools
import logging
from typing import TypeVar, Callable
from groq import GroqError
from fastapi import HTTPException

logger = logging.getLogger(__name__)

T = TypeVar("T")


def handle_errors(operation: str = "Operation"):
    """
    Decorator to handle errors in async functions with consistent logging

    Args:
        operation: Description of the operation being performed
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Re-raise HTTP exceptions as they already have proper status codes
                raise
            except GroqError as e:
                # Handle Groq specific errors
                logger.error(f"{operation} failed - Groq error: {str(e)}")
                raise Exception(f"{operation} failed: LLM service error")
            except FileNotFoundError as e:
                # Handle file not found errors (configs, prompts)
                logger.error(f"{operation} failed - File not found: {str(e)}")
                raise Exception(f"{operation} failed: Configuration error")
            except ValueError as e:
                # Handle validation errors
                logger.error(f"{operation} failed - Validation error: {str(e)}")
                raise ValueError(str(e))
            except Exception as e:
                # Handle any other unexpected errors
                logger.error(f"{operation} failed - Unexpected error: {str(e)}")
                raise Exception(f"{operation} failed: {str(e)}")

        return wrapper

    return decorator


def handle_analyzer_errors(analyzer_name: str):
    """
    Specialized decorator for analyzer methods

    Args:
        analyzer_name: Name of the analyzer (e.g., "Fundamental", "Momentum")
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(self, ticker: str, *args, **kwargs) -> T:
            try:
                logger.info(f"Starting {analyzer_name} analysis for {ticker}")
                result = await func(self, ticker, *args, **kwargs)
                logger.info(f"Completed {analyzer_name} analysis for {ticker}")
                return result
            except Exception as e:
                logger.error(f"{analyzer_name} analysis failed for {ticker}: {str(e)}")
                raise Exception(f"{analyzer_name} analysis failed: {str(e)}")

        return wrapper

    return decorator


def handle_service_errors(service_name: str):
    """
    Decorator for service methods with specific error handling

    Args:
        service_name: Name of the service
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{service_name} error: {str(e)}")
                # Don't wrap the exception again if it's already wrapped
                if str(e).startswith(f"{service_name} error:"):
                    raise
                raise Exception(f"{service_name} error: {str(e)}")

        return wrapper

    return decorator
