# TraceLink

AI-assisted missing-person case management and investigation support — built to assist human investigators, not replace them.

## Overview

When someone goes missing, information fragments fast: case files in one place, reported sightings in another, photographs scattered across inboxes and drives. TraceLink pulls that together into one managed workflow. Authorized users maintain cases, store evidence photos privately, and get computational help — preprocessing, face detection, face representation, similarity retrieval — at each step. Every AI output is candidate evidence for a person to review. The system never decides who someone is.

## What it does

- Case management with organizations, memberships, and role-based access
- Private evidence storage for case photos and sighting photos (S3-compatible; MinIO for local dev)
- Deterministic image preprocessing with SHA-pinned provenance
- Face detection (YuNet), with re-detection and full run history
- Face representation (OpenCV SFace, 128-dimensional embeddings in pgvector)
- Similarity retrieval across authorized evidence: case → sighting and sighting → case
- Cosine similarity with threshold + Top-K, filtered by authorization before ranking
- JWT authentication with Argon2 password hashing

## Architecture

```text
Case / Sighting
→ Private Evidence
→ Preprocessing
→ Face Detection
→ Face Representation
→ Similarity Retrieval
→ Human Review
```

- **Case / Sighting** — missing-person cases and reported sightings of them, scoped to organizations.
- **Private Evidence** — original photos are preserved in private object storage and are never overwritten by processing.
- **Preprocessing** — deterministic normalization (EXIF, color, size, JPEG quality), SHA-recorded so every later stage can verify it is working from current bytes.
- **Face Detection** — YuNet finds faces and landmarks; each attempt is a recorded run, so stale results are never mistaken for current ones.
- **Face Representation** — SFace converts each detected face into a 128-D embedding, versioned by representation, model, and model hash.
- **Similarity Retrieval** — pgvector cosine distance ranks faces from the opposite evidence type; only current, compatible, in-scope embeddings participate.
- **Human Review** — ranked candidates with scores go to an investigator or reviewer. Nothing is auto-confirmed.

## Tech Stack

| Layer    | Technology                                            |
|----------|-------------------------------------------------------|
| Backend  | Python, FastAPI, SQLAlchemy 2.x, Alembic, Pydantic    |
| Database | PostgreSQL 18 + pgvector                              |
| Auth     | JWT, Argon2                                           |
| Storage  | boto3, S3-compatible object storage (MinIO locally)   |
| Frontend | React, Vite, React Router, Axios, React Context       |
| Vision   | Pillow, OpenCV, YuNet (detection), SFace (representation) |

## Current Status

- [x] Phase 0 — Foundation, auth, organizations, cases, authorization
- [x] Phase 1 — Private case-photo evidence storage
- [x] Phase 2 — Sightings and sighting-photo evidence
- [x] Phase 3 — Deterministic image preprocessing
- [x] Phase 4 — Face detection (YuNet)
- [x] Phase 5 — Face representation (SFace, 128-D embeddings)
- [x] Phase 6 — Similarity retrieval (pgvector cosine distance)

## Design principles

- **Originals are sacred.** Private evidence is write-once; processing produces derived images, never overwrites.
- **Provenance is checked, not assumed.** Every stage re-verifies SHA hashes and version identity before trusting prior output.
- **Authorization comes first.** Candidates outside your scope are excluded before scoring, and retrieval never grants edit rights.
- **Similarity is not identity.** A score is a retrieval ranking over pixels — confirmation is always a human decision.

## Future direction

Not implemented yet, but on the roadmap: image enhancement and restoration for poor-quality evidence, richer visual and contextual analysis, fuller investigation workflows, and generated posters and reports.

## Local setup

You need PostgreSQL (with pgvector) and an S3-compatible store (MinIO works locally). Then:

```bash
cd backend
cp .env.example .env        # fill in real values; never commit .env
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

```bash
cd frontend
npm install
npm run dev
```

See `backend/.env.example` for every setting, including storage, detector, and representation options.

## Note about model files

The YuNet and SFace ONNX weights are intentionally excluded from version control (`backend/app/assets/*.onnx` is gitignored). Obtain them from the OpenCV Zoo releases, place them under `backend/app/assets/`, or point the documented `*_MODEL_PATH` variables at them. See `backend/app/assets/README.md` for exact filenames and details.

## Project note

TraceLink is an academic, non-commercial project exploring how AI can responsibly support — and never supplant — human judgment in missing-person investigations.
