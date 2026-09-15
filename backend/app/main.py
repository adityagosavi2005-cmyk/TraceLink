from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.cases import router as cases_router

from app.routers.auth import router as auth_router

app = FastAPI(
    title="TraceLink API",
    description="AI-assisted missing-person search and case management platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(cases_router)


@app.get("/")
def root():
    return {
        "message": "TraceLink API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }