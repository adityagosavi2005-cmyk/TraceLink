from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.cases import router as cases_router
from app.routers.case_photos import router as case_photos_router
from app.routers.sightings import router as sightings_router
from app.routers.sightings import index_router as sightings_index_router
from app.routers.sighting_photos import router as sighting_photos_router
from app.routers.face_detections import case_router as face_case_router
from app.routers.face_detections import sighting_router as face_sighting_router
from app.routers.enhancements import case_router as enhancement_case_router
from app.routers.enhancements import (
    sighting_router as enhancement_sighting_router,
)
from app.routers.similarity import router as similarity_router
from app.routers.organizations import router as organizations_router

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
app.include_router(case_photos_router)
app.include_router(sightings_router)
app.include_router(sightings_index_router)
app.include_router(sighting_photos_router)
app.include_router(face_case_router)
app.include_router(face_sighting_router)
app.include_router(enhancement_case_router)
app.include_router(enhancement_sighting_router)
app.include_router(similarity_router)
app.include_router(organizations_router)


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