"""
Error Handling Middleware

Global exception handlers for the FastAPI application.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
import traceback


def setup_exception_handlers(app: FastAPI):
    """
    Setup global exception handlers for the FastAPI application.
    """

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors"""
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "success": False,
                "error": "Validation Error",
                "detail": exc.errors(),
                "body": exc.body
            }
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle ValueError exceptions"""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": "Invalid Value",
                "detail": str(exc)
            }
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions"""
        # Log the full traceback for debugging
        print(f"❌ Unhandled exception: {type(exc).__name__}")
        print(traceback.format_exc())

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": "Internal Server Error",
                "detail": str(exc),
                "type": type(exc).__name__
            }
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        """Handle 404 Not Found"""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "success": False,
                "error": "Not Found",
                "detail": f"Path {request.url.path} not found"
            }
        )
