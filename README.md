# TraceLink

AI-assisted missing-person case management and investigation support — built to assist human investigators, not replace them.

## Overview

When someone goes missing, information fragments fast: case files in one place, reported sightings in another, photographs scattered across inboxes and drives. TraceLink pulls that together into one managed workflow. Authorized users maintain cases, store evidence photos privately, and get computational help — preprocessing, face detection, face representation, similarity retrieval — plus two explicit opt-in paths, whole-image enhancement (Real-ESRGAN) and face-level restoration (GFPGAN) — at each step. Every AI output is candidate evidence for a person to review. The system never decides who someone is.

## What it does

- Case management with organizations, memberships, and role-based access
- Private evidence storage for case photos and sighting photos (S3-compatible; MinIO for local dev)
- Deterministic image preprocessing with SHA-pinned provenance
- Face detection (YuNet), with re-detection and full run history
- Face representation (OpenCV SFace, 128-dimensional embeddings in pgvector)
- Similarity retrieval across authorized evidence: case → sighting and sighting → case
- Cosine similarity with threshold + Top-K, filtered by authorization before ranking
- JWT authentication with Argon2 password hashing
- Optional whole-image enhancement (Real-ESRGAN): explicit per-photo runs that never overwrite originals
- Optional face-level restoration (GFPGAN): an investigator explicitly selects a detected face; each restoration is a recorded run whose restored artifact stays separate from originals and derived images, and restored faces can earn SFace embeddings and join similarity retrieval

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

Optional paths (explicit, never automatic):

```text
Preprocessed Image
→ optional explicit Real-ESRGAN Enhancement (Phase 7)
→ Enhanced Artifact
→ Face Detection
→ Face Representation
→ Similarity Retrieval
```

```text
Phase 3 Derived Image
→ Face Detection
→ explicit face selection (Phase 8)
→ Face Input Preparation
→ GFPGAN Face Restoration
→ Restored Face Artifact
→ SFace Face Representation
→ Similarity Retrieval
→ Human Review
```

Real-ESRGAN and GFPGAN are separate optional capabilities: neither runs automatically, and GFPGAN never chains off an enhanced image — restoration consumes the Phase 3 derived image only.

- **Case / Sighting** — missing-person cases and reported sightings of them, scoped to organizations.
- **Private Evidence** — original photos are preserved in private object storage and are never overwritten by processing.
- **Preprocessing** — deterministic normalization (EXIF, color, size, JPEG quality), SHA-recorded so every later stage can verify it is working from current bytes.
- **Face Detection** — YuNet finds faces and landmarks; each attempt is a recorded run, so stale results are never mistaken for current ones.
- **Face Representation** — SFace converts each detected face into a 128-D embedding, versioned by representation, model, and model hash.
- **Similarity Retrieval** — pgvector cosine distance ranks faces from the opposite evidence type; only current, compatible, in-scope embeddings participate.
- **Human Review** — ranked candidates with scores go to an investigator or reviewer. Nothing is auto-confirmed.
- **Enhancement & Restoration (optional)** — Phase 7 upscales whole images with Real-ESRGAN; Phase 8 restores investigator-selected faces with GFPGAN. Both are explicit recorded runs whose artifacts stay separate from originals and derived images.

## Face restoration (Phase 8)

An investigator or reviewer explicitly selects a detected face; restoration runs only for that face, and multiple faces restore independently. Each restoration is an explicit recorded `FaceRestorationRun`: face input preparation → GFPGAN → restored artifact, with provenance (face, preparation record, model hashes, geometry strategy, artifact SHA) and run identity on every run. Restored artifacts are stored separately — originals and Phase 3 derived images are never overwritten. Because GFPGAN realigns faces internally, restored-frame geometry is recorded as versioned canonical geometry for SFace. Each restored embedding is associated with its exact restoration run, similarity searches can explicitly use a selected restored run, and a restored query never silently falls back to the normal embedding. Restored matches carry a synthesized-detail warning: they are investigation-support signals for human review, not identity evidence.

## Tech Stack

| Layer    | Technology                                            |
|----------|-------------------------------------------------------|
| Backend  | Python, FastAPI, SQLAlchemy 2.x, Alembic, Pydantic    |
| Database | PostgreSQL 18 + pgvector                              |
| Auth     | JWT, Argon2                                           |
| Storage  | boto3, S3-compatible object storage (MinIO locally)   |
| Frontend | React, Vite, React Router, Axios, React Context       |
| Vision   | Pillow, OpenCV, YuNet (detection), SFace (representation), Real-ESRGAN (optional enhancement), GFPGAN (optional face-level restoration) |

## Current Status

- [x] Phase 0 — Foundation, auth, organizations, cases, authorization
- [x] Phase 1 — Private case-photo evidence storage
- [x] Phase 2 — Sightings and sighting-photo evidence
- [x] Phase 3 — Deterministic image preprocessing
- [x] Phase 4 — Face detection (YuNet)
- [x] Phase 5 — Face representation (SFace, 128-D embeddings)
- [x] Phase 6 — Similarity retrieval (pgvector cosine distance)
- [x] Phase 7 — Optional whole-image enhancement/restoration using Real-ESRGAN (x4, ADMIN/REVIEWER only, explicit per-photo runs; originals and Phase 3 derived images are never modified)
- [x] Phase 8 — Optional face-level restoration using GFPGAN (ADMIN/REVIEWER only, explicit per-face runs; restored artifacts stored separately, originals and Phase 3 derived images never modified; restored embeddings join similarity only via explicit run selection)

## Design principles

- **Originals are sacred.** Private evidence is write-once; processing produces derived images, never overwrites.
- **Provenance is checked, not assumed.** Every stage re-verifies SHA hashes and version identity before trusting prior output.
- **Authorization comes first.** Candidates outside your scope are excluded before scoring, and retrieval never grants edit rights.
- **Similarity is not identity.** A score is a retrieval ranking over pixels — confirmation is always a human decision.
- **Explicit runs, never implicit "latest".** Enhancement and restoration artifacts are always chosen by run id; nothing auto-selects the latest run.
- **Restored provenance is preserved.** Every restoration run records its face, preparation, model hashes, geometry strategy, and artifact SHA; restored embeddings point at the exact restoration run.
- **Restoration is human-selected.** Investigators pick faces; the system never auto-restores, and a restored query never silently falls back to the normal embedding.

## Future direction

Poor-quality evidence has two implemented optional paths: whole-image enhancement as Phase 7 (explicit Real-ESRGAN runs) and face-level restoration as Phase 8 (explicit per-face GFPGAN runs) — both never automatic and never overwriting evidence. Still on the roadmap: richer visual and contextual analysis, fuller investigation workflows, and generated posters and reports.

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

Phase 7 additionally needs the optional Real-ESRGAN weights (`RealESRGAN_x4plus.pth`, gitignored via `*.pth`, never in the evidence bucket) plus the torch runtime (`backend/requirements-enhancement.txt`). Without them the API still runs; enhancement attempts fail with a controlled error. See `backend/app/assets/README.md` for the verified SHA-256 and license.

Phase 8 needs the optional GFPGAN weights (`backend/app/assets/GFPGANv1.3.pth`, gitignored, never in the evidence bucket; TencentARC/GFPGAN release v1.3.0, SHA-256 `c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70`, 348,632,874 bytes — license terms were NOT verified, review the upstream repository license before production use) plus two required facexlib auxiliary weights pre-seeded at `backend/gfpgan/weights/` (`detection_Resnet50_Final.pth`, SHA-256 `6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d`; `parsing_parsenet.pth`, SHA-256 `3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2`) and the pinned torch runtime (`torch==2.9.1+cpu` / `torchvision==0.24.1+cpu` with `gfpgan==1.3.8`, `basicsr==1.4.2`, `facexlib==0.3.0`, installed via the controlled 4-step procedure in `backend/requirements-restoration.txt`). Without them the API still runs; restoration attempts fail with a controlled error.

## Project note

TraceLink is an academic, non-commercial project exploring how AI can responsibly support — and never supplant — human judgment in missing-person investigations.
