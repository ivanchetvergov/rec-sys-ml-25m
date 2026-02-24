import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.movies import router as movies_router

load_dotenv()  # loads backend/.env if present

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s – %(message)s")

app = FastAPI(title="RecSys API", version="0.1.0", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(movies_router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
