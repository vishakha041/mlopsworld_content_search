"""
Main entry point compatibility shim

This tiny shim allows running `uvicorn main:app` from the `backend/` folder
by importing the real application from `app.main` and exposing it as `app`.
"""
from app.main import app

if __name__ == "__main__":
    import uvicorn
    from app.config import settings

    uvicorn.run(app, host=settings.HOST, port=settings.PORT, reload=True)
