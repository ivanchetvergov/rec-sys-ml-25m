import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.movies import router as movies_router
from app.routers.auth import router as auth_router
from app.services.popularity_service import get_popularity_service
from app.services.recommender_service import get_recommender_service
from app.services.similarity_service import get_similarity_service
from app.database import run_migrations

load_dotenv()  # loads backend/.env if present

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s – %(message)s")

app = FastAPI(title="RecSys API", version="0.1.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # local dev
        "http://localhost",        # nginx (port 80)
        "http://localhost:80",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(movies_router, prefix="/api")
app.include_router(auth_router, prefix="/api")


@app.on_event("startup")
async def _warm_up() -> None:
    """Pre-load all heavy services so the first HTTP request is fast."""
    logger = logging.getLogger("startup")

    logger.info("Running database migrations...")
    run_migrations()
    logger.info("Migrations done. Starting service warmup...")

    logger.info("Warming up PopularityService...")
    pop = get_popularity_service()
    pop._ensure_movies_loaded()
    logger.info(f"PopularityService ready — {pop.total_count():,} movies")

    logger.info("Warming up RecommenderService (ALS + CatBoost)...")
    rec = get_recommender_service()
    rec._ensure_loaded()
    status = "two_stage ready" if rec.model_available else "model not found — popularity fallback"
    logger.info(f"RecommenderService ready — {status}")

    logger.info("Warming up SimilarityService …")
    sim = get_similarity_service()
    sim._ensure_loaded()
    logger.info(f"SimilarityService ready — available={sim.available}")


@app.get("/api/health")
def health():
    return {"status": "ok"}
