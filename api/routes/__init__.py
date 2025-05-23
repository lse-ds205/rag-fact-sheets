from fastapi import FastAPI
from .chatapi import router as auth_router
from .scrape import router as scrape_router

def register_routes(app: FastAPI):
    app.include_router(auth_router)
    app.include_router(scrape_router)