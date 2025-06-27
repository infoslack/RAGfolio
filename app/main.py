import logging
from fastapi import FastAPI
from app.config.settings import Settings
from app.routers.search import router as search_router
from app.routers.llm import router as llm_router
from app.routers.agent import router as agent_router
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def get_settings():
    return Settings()


def create_application():
    # Initialize settings
    settings = get_settings()

    # Create FastAPI app
    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],  # Frontend local
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add routers
    app.include_router(search_router)
    app.include_router(llm_router)
    app.include_router(agent_router)

    @app.get("/")
    async def root():
        return {"message": "Welcome to RAG API"}

    @app.get("/health")
    async def health_check():
        logging.info("Health endpoint was called")
        return {"status": "Ok"}

    return app


app = create_application()
