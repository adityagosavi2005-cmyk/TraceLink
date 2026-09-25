import React, { useState, useEffect, useCallback } from 'react';
import {
  photoService,
  sightingService,
  enhancementService,
  similarityService,
  restorationService,
} from '../services/api';
import { Modal } from './Modal';
import { LoadingState, EmptyState, ErrorState } from './StateBlocks';
import {
  EnhancementStatusBadge,
  RestorationStatusBadge,
  FaceStatusBadge,
  SourceBadge,
} from './StatusBadge';
import { SectionHeader } from './Workspace';

const apiMessage = (err, fallback) => {
  if (err && err.response) {
    if (err.response.status === 403) {
      return 'You do not have permission to perform this action.';
    }
    if (err.response.status === 404) {
      return (
        (err.response.data && err.response.data.detail) ||
        'The requested item was not found. It may have been deleted.'
      );
    }
    if (err.response.status === 409 || err.response.status === 422) {
      return (
        (err.response.data && err.response.data.detail) ||
        'The request conflicts with the current state.'
      );
    }
    if (err.response.data && err.response.data.detail) {
      return err.response.data.detail;
    }
  }
  return fallback;
};

const buildPhotoApi = (photoKind, caseId, sightingId, photoId) => {
  if (photoKind === 'sighting') {
    return {
      fetchFaces: () =>
        sightingService.getSightingFaces(caseId, sightingId, photoId),
      detectFaces: (runId) =>
        sightingService.detectSightingFaces(
          caseId,
          sightingId,
          photoId,
          runId
        ),
      listRuns: () =>
        enhancementService.listSightingEnhancements(caseId, sightingId, photoId),
      triggerEnhancement: () =>
        enhancementService.triggerSightingEnhancement(caseId, sightingId, photoId),
      getRun: (runId) =>
        enhancementService.getSightingEnhancement(
          caseId,
          sightingId,
          photoId,
          runId
        ),
      retryRun: (runId) =>
        enhancementService.retrySightingEnhancement(
          caseId,
          sightingId,
          photoId,
          runId
        ),
      searchSimilar: (faceId) =>
        similarityService.searchSightingFaceSimilar(
          caseId,
          sightingId,
          photoId,
          faceId
        ),
    };
  }
  return {
    fetchFaces: () => photoService.getFaces(caseId, photoId),
    detectFaces: (runId) => photoService.detectFaces(caseId, photoId, runId),
    listRuns: () => enhancementService.listCaseEnhancements(caseId, photoId),
    triggerEnhancement: () =>
      enhancementService.triggerCaseEnhancement(caseId, photoId),
    getRun: (runId) =>
      enhancementService.getCaseEnhancement(caseId, photoId, runId),
    retryRun: (runId) =>
      enhancementService.retryCaseEnhancement(caseId, photoId, runId),
    searchSimilar: (faceId) =>
      similarityService.searchCaseFaceSimilar(caseId, photoId, faceId),
  };
};

const buildRestorationApi = (photoKind, caseId, sightingId, photoId) => {
  if (photoKind === 'sighting') {
    return {
      trigger: (faceId) =>
        restorationService.triggerSightingRestoration(caseId, sightingId, photoId, faceId),
      list: (faceId) =>
        restorationService.listSightingRestorations(caseId, sightingId, photoId, faceId),
      get: (faceId, runId) =>
        restorationService.getSightingRestoration(caseId, sightingId, photoId, faceId, runId),
      retry: (faceId, runId) =>
        restorationService.retrySightingRestoration(caseId, sightingId, photoId, faceId, runId),
      embed: (faceId, runId) =>
        restorationService.embedSightingRestoredFace(caseId, sightingId, photoId, faceId, runId),
      searchSimilarRestored: (faceId, runId, options) =>
        sightingService.searchSightingFaceSimilar(caseId, sightingId, photoId, faceId, {
          ...(options || {}),
          restorationRunId: runId,
        }),
    };
  }
  return {
    trigger: (faceId) =>
      restorationService.triggerCaseRestoration(caseId, photoId, faceId),
    list: (faceId) =>
      restorationService.listCaseRestorations(caseId, photoId, faceId),
    get: (faceId, runId) =>
      restorationService.getCaseRestoration(caseId, photoId, faceId, runId),
    retry: (faceId, runId) =>
      restorationService.retryCaseRestoration(caseId, photoId, faceId, runId),
    embed: (faceId, runId) =>
      restorationService.embedCaseRestoredFace(caseId, photoId, faceId, runId),
    searchSimilarRestored: (faceId, runId, options) =>
      similarityService.searchCaseFaceSimilar(caseId, photoId, faceId, {
        ...(options || {}),
        restorationRunId: runId,
      }),
  };
};

const FaceFigure = ({ imageUrl, faces, alt, onImageError }) => {
  const boxes = (faces || []).filter(
    (face) => face.frame_width > 0 && face.frame_height > 0
  );
  return (
    <div className="ai-face-figure">
      {imageUrl ? (
        <img
          src={imageUrl}
          alt={alt}
          className="ai-face-image"
          onError={onImageError}
        />
      ) : (
        <div className="ai-face-noimage">No preview available</div>
      )}
      {boxes.map((face) => (
        <span
          key={face.id}
          className="ai-face-box"
          style={{
            left: `${(face.x_min / face.frame_width) * 100}%`,
            top: `${(face.y_min / face.frame_height) * 100}%`,
            width: `${(Math.max(0, face.x_max - face.x_min) / face.frame_width) * 100}%`,
            height: `${(Math.max(0, face.y_max - face.y_min) / face.frame_height) * 100}%`,
          }}
          title={`Face #${face.ordinal} · confidence ${Number(face.confidence).toFixed(2)}`}
        />
      ))}
    </div>
  );
};

const FaceRow = ({ face, searching, onFindSimilar }) => (
  <div className="ai-face-row">
    <div className="ai-face-meta">
      <span className="ai-face-id">Face #{face.ordinal}</span>
      <span className="ai-face-confidence">
        Confidence {Number(face.confidence).toFixed(2)}
      </span>
      <span className="ai-face-boxdims">
        Box {face.x_max - face.x_min}×{face.y_max - face.y_min} px
      </span>
    </div>
    <button
      type="button"
      className="btn btn-ai btn-sm"
      onClick={() => onFindSimilar(face)}
      disabled={searching}
      title="Search for faces similar to this detected face"
    >
      {searching ? '…' : 'Find similar'}
    </button>
  </div>
);

const formatSimilarity = (value) =>
  `${(Number(value) * 100).toFixed(1)}%`;

const CandidateRow = ({ candidate }) => {
  const context =
    candidate.photo_type === 'sighting'
      ? `Sighting #${candidate.sighting_id} · Photo #${candidate.photo_id} · Face #${candidate.face_id}`
      : `Case #${candidate.case_id} · Photo #${candidate.photo_id} · Face #${candidate.face_id}`;
  return (
    <div className="ai-candidate-row">
      <div className="ai-candidate-main">
        <span className="ai-candidate-score">
          {formatSimilarity(candidate.similarity)}
        </span>
        <span className="ai-candidate-context">{context}</span>
      </div>
      <div className="ai-candidate-sub">
        <SourceBadge
          sourceType={candidate.source_type}
          runId={candidate.enhancement_run_id}
        />
        <span className="ai-candidate-review">Needs human review</span>
        {candidate.is_restored && (
          <span
            className="ai-candidate-warning"
            title={candidate.synthesized_detail_warning || 'GFPGAN may synthesize facial detail.'}
          >
            Restored face{candidate.face_restoration_run_id ? ` · Run #${candidate.face_restoration_run_id}` : ''} · synthetic detail possible
          </span>
        )}
      </div>
    </div>
  );
};

export const PhotoAIModal = ({
  photo,
  photoKind,
  caseId,
  sightingId,
  canEnhance,
  onClose,
  onPhotosChanged,
}) => {
  const photoApi = buildPhotoApi(photoKind, caseId, sightingId, photo.id);
  const restorationApi = buildRestorationApi(photoKind, caseId, sightingId, photo.id);

  const [bannerError, setBannerError] = useState(null);

  const [facesData, setFacesData] = useState(null);
  const [facesLoading, setFacesLoading] = useState(true);
  const [facesError, setFacesError] = useState(null);

  const [runs, setRuns] = useState([]);
  const [runsLoading, setRunsLoading] = useState(true);
  const [runsError, setRunsError] = useState(null);
  const [enhancing, setEnhancing] = useState(false);
  const [retryingRunId, setRetryingRunId] = useState(null);
  const [selectedRunId, setSelectedRunId] = useState(null);
  const [artifactUrl, setArtifactUrl] = useState(null);
  const [artifactLoading, setArtifactLoading] = useState(false);

  const [enhancedResult, setEnhancedResult] = useState(null);
  const [enhancedDetecting, setEnhancedDetecting] = useState(false);

  const [searchingFaceId, setSearchingFaceId] = useState(null);
  const [selectedFaceId, setSelectedFaceId] = useState(null);
  const [restorations, setRestorations] = useState([]);
  const [restorationsLoading, setRestorationsLoading] = useState(false);
  const [restoringFaceId, setRestoringFaceId] = useState(null);
  const [retryingRestorationId, setRetryingRestorationId] = useState(null);
  const [selectedRestorationId, setSelectedRestorationId] = useState(null);
  const [restorationArtifactUrl, setRestorationArtifactUrl] = useState(null);
  const [restorationArtifactLoading, setRestorationArtifactLoading] = useState(false);
  const [embeddingResult, setEmbeddingResult] = useState(null);
  const [embeddingLoading, setEmbeddingLoading] = useState(false);
  const [similarity, setSimilarity] = useState(null);
  const [similarityError, setSimilarityError] = useState(null);

  // Query provenance for the Similarity results section. Normal
  // searches carry no restoredRunId; restored searches carry the
  // explicitly selected restoration run id (set by
  // handleFindSimilarRestored, never inferred).
  const similarityRestoredRunId =
    similarity &&
    similarity.restoredRunId !== undefined &&
    similarity.restoredRunId !== null
      ? similarity.restoredRunId
      : null;
  const similarityQueryProvenance =
    similarity && similarity.face
      ? (similarityRestoredRunId !== null
        ? `Query: Face #${similarity.face.ordinal} · Restored face · Run #${similarityRestoredRunId}`
        : `Query: Face #${similarity.face.ordinal} · Normal derived face`)
      : null;

  const similarityCandidates =
    similarity &&
    similarity.result &&
    Array.isArray(similarity.result.results)
      ? similarity.result.results
      : null;

  const fetchFaces = useCallback(async () => {
    setFacesLoading(true);
    setFacesError(null);
    try {
      const data = await photoApi.fetchFaces();
      setFacesData(data);
    } catch (err) {
      setFacesData(null);
      setFacesError(apiMessage(err, 'Failed to load face results.'));
    } finally {
      setFacesLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [photoKind, caseId, sightingId, photo.id]);

  const fetchRuns = useCallback(async () => {
    setRunsLoading(true);
    setRunsError(null);
    try {
      const data = await photoApi.listRuns();
      setRuns(Array.isArray(data) ? data : []);
    } catch (err) {
      setRuns([]);
      setRunsError(apiMessage(err, 'Failed to load enhancement runs.'));
    } finally {
      setRunsLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [photoKind, caseId, sightingId, photo.id]);

  useEffect(() => {
    fetchFaces();
    fetchRuns();
  }, [fetchFaces, fetchRuns]);

  const selectedRun =
    runs.find((run) => run.id === selectedRunId) || null;

  const handleTriggerEnhancement = async () => {
    setEnhancing(true);
    setBannerError(null);
    try {
      await photoApi.triggerEnhancement();
      await fetchRuns();
      if (onPhotosChanged) onPhotosChanged();
    } catch (err) {
      setBannerError(apiMessage(err, 'Failed to start enhancement.'));
    } finally {
      setEnhancing(false);
    }
  };

  const handleRetryRun = async (runId) => {
    setRetryingRunId(runId);
    setBannerError(null);
    try {
      await photoApi.retryRun(runId);
      await fetchRuns();
      if (onPhotosChanged) onPhotosChanged();
    } catch (err) {
      setBannerError(apiMessage(err, 'Failed to retry enhancement.'));
    } finally {
      setRetryingRunId(null);
    }
  };

  const handleViewArtifact = async (run) => {
    if (!run || run.status !== 'COMPLETE') return;
    setArtifactLoading(true);
    setBannerError(null);
    try {
      // Refetch the run detail every time so the presigned artifact
      // URL is never treated as permanent.
      const detail = await photoApi.getRun(run.id);
      setSelectedRunId(run.id);
      setEnhancedResult(null);
      setArtifactUrl(detail.view_url || null);
      if (!detail.view_url) {
        setBannerError('No viewable artifact is available for this run.');
      }
    } catch (err) {
      setArtifactUrl(null);
      setBannerError(apiMessage(err, 'Failed to load the enhanced artifact.'));
    } finally {
      setArtifactLoading(false);
    }
  };

  const handleEnhancedDetect = async () => {
    if (!selectedRun) return;
    setEnhancedDetecting(true);
    setBannerError(null);
    try {
      const result = await photoApi.detectFaces(selectedRun.id);
      setEnhancedResult(result);
      if (onPhotosChanged) onPhotosChanged();
    } catch (err) {
      if (err && err.response && err.response.status === 404) {
        setSelectedRunId(null);
        setArtifactUrl(null);
        setEnhancedResult(null);
        await fetchRuns();
      }
      setBannerError(
        apiMessage(err, 'Enhanced face detection failed.')
      );
    } finally {
      setEnhancedDetecting(false);
    }
  };

  const fetchRestorations = useCallback(async (faceId) => {
    if (!faceId) {
      setRestorations([]);
      return;
    }
    setRestorationsLoading(true);
    try {
      const data = await restorationApi.list(faceId);
      setRestorations(Array.isArray(data) ? data : []);
    } catch (err) {
      setRestorations([]);
      setBannerError(apiMessage(err, 'Failed to load face restoration runs.'));
    } finally {
      setRestorationsLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [photoKind, caseId, sightingId, photo.id]);

  const handleSelectFaceForRestoration = (faceId) => {
    const id = faceId ? Number(faceId) : null;
    setSelectedFaceId(id);
    setSelectedRestorationId(null);
    setRestorationArtifactUrl(null);
    setEmbeddingResult(null);
    if (id) fetchRestorations(id);
    else setRestorations([]);
  };

  const handleRestoreFace = async (faceId) => {
    setRestoringFaceId(faceId);
    setBannerError(null);
    try {
      await restorationApi.trigger(faceId);
      await fetchRestorations(faceId);
      if (onPhotosChanged) onPhotosChanged();
    } catch (err) {
      setBannerError(apiMessage(err, 'Failed to start face restoration.'));
    } finally {
      setRestoringFaceId(null);
    }
  };

  const handleViewRestoration = async (faceId, run) => {
    if (!run || run.status !== 'COMPLETE') return;
    setRestorationArtifactLoading(true);
    setBannerError(null);
    try {
      // Refetch the run detail every time so the presigned artifact
      // URL is never treated as permanent.
      const detail = await restorationApi.get(faceId, run.id);
      setSelectedRestorationId(run.id);
      setEmbeddingResult(null);
      setRestorationArtifactUrl(detail.view_url || null);
      if (!detail.view_url) {
        setBannerError('No viewable artifact is available for this run.');
      }
    } catch (err) {
      setRestorationArtifactUrl(null);
      setBannerError(apiMessage(err, 'Failed to load the restored artifact.'));
    } finally {
      setRestorationArtifactLoading(false);
    }
  };

  const handleRetryRestoration = async (faceId, runId) => {
    setRetryingRestorationId(runId);
    setBannerError(null);
    try {
      await restorationApi.retry(faceId, runId);
      await fetchRestorations(faceId);
      if (onPhotosChanged) onPhotosChanged();
    } catch (err) {
      setBannerError(apiMessage(err, 'Failed to retry face restoration.'));
    } finally {
      setRetryingRestorationId(null);
    }
  };

  const handleEmbedRestored = async (faceId, run) => {
    if (!run || run.status !== 'COMPLETE') return;
    setEmbeddingLoading(true);
    setBannerError(null);
    try {
      const result = await restorationApi.embed(faceId, run.id);
      setEmbeddingResult(result);
    } catch (err) {
      setEmbeddingResult(null);
      setBannerError(apiMessage(err, 'Restored-face embedding failed.'));
    } finally {
      setEmbeddingLoading(false);
    }
  };

  const handleFindSimilarRestored = async (face, runId) => {
    setSearchingFaceId(face.id);
    setSimilarityError(null);
    try {
      const result = await restorationApi.searchSimilarRestored(face.id, runId);
      setSimilarity({ face, result, restoredRunId: runId });
    } catch (err) {
      setSimilarity(null);
      setSimilarityError(apiMessage(err, 'Similarity search failed.'));
    } finally {
      setSearchingFaceId(null);
    }
  };

  const handleFindSimilar = async (face) => {
    setSearchingFaceId(face.id);
    setSimilarityError(null);
    try {
      const result = await photoApi.searchSimilar(face.id);
      setSimilarity({ face, result });
    } catch (err) {
      setSimilarity(null);
      setSimilarityError(apiMessage(err, 'Similarity search failed.'));
    } finally {
      setSearchingFaceId(null);
    }
  };

  const normalFaces = facesData && Array.isArray(facesData.faces)
    ? facesData.faces
    : [];
  const normalImageUrl =
    (photo && (photo.derived_view_url || photo.view_url)) || null;

  const enhancedFaces =
    enhancedResult && Array.isArray(enhancedResult.faces)
      ? enhancedResult.faces
      : [];

  return (
    <Modal
      title={`AI workflow — Photo #${photo.id}`}
      onClose={onClose}
      wide
    >
      {bannerError && (
        <div className="alert alert-error" role="alert">
          {bannerError}
        </div>
      )}

      <div className="evidence-section">
        <SectionHeader
          title="Detected faces"
          meta={<SourceBadge sourceType="DERIVED" runId={null} />}
        />
        {facesLoading ? (
          <LoadingState label="Loading face results…" />
        ) : facesError ? (
          <ErrorState detail={facesError} onRetry={fetchFaces} />
        ) : normalFaces.length === 0 ? (
          <EmptyState
            title={
              facesData && facesData.status === 'COMPLETE'
                ? 'Face detection complete — 0 faces detected'
                : 'No face scan yet'
            }
            detail={
              facesData && facesData.status === 'COMPLETE'
                ? 'This is a completed result, not an error. Run detection from the photo card to scan.'
                : 'Run face detection from the photo card to scan this photo.'
            }
          />
        ) : (
          <>
            <FaceFigure
              imageUrl={normalImageUrl}
              faces={normalFaces}
              alt={`Photo ${photo.id} with detected face boxes`}
            />
            {normalFaces.map((face) => (
              <FaceRow
                key={face.id}
                face={face}
                searching={searchingFaceId === face.id}
                onFindSimilar={handleFindSimilar}
              />
            ))}
          </>
        )}
      </div>

      <div className="evidence-section">
        <SectionHeader
          title="Optional enhancement"
          meta="ADMIN / REVIEWER"
        />
        <p className="ai-note">
          Enhancement is optional. Normal processing above stays
          available whether or not an enhanced version is ever created.
          Enhanced images never replace the original evidence.
        </p>
        {canEnhance && (
          <button
            type="button"
            className="btn btn-ai btn-sm"
            onClick={handleTriggerEnhancement}
            disabled={enhancing}
            title="Create a new enhancement run for this photo"
          >
            {enhancing ? 'Enhancing…' : 'Enhance this photo'}
          </button>
        )}
        {!canEnhance && (
          <p className="ai-note">
            Enhancement can be triggered by ADMIN or REVIEWER users.
          </p>
        )}
        {runsLoading ? (
          <LoadingState label="Loading enhancement runs…" />
        ) : runsError ? (
          <ErrorState detail={runsError} onRetry={fetchRuns} />
        ) : runs.length === 0 ? (
          <EmptyState
            title="No enhancement runs yet"
            detail="Create one to try an AI-upscaled version of this photo."
          />
        ) : (
          <div className="ai-run-list">
            {runs.map((run) => (
              <div key={run.id} className="ai-run-row">
                <div className="ai-run-main">
                  <span className="ai-run-id">Run #{run.id}</span>
                  <EnhancementStatusBadge status={run.status} />
                  {run.status === 'COMPLETE' &&
                    run.width > 0 &&
                    run.height > 0 && (
                      <span className="ai-run-dims">
                        {run.width}×{run.height}
                      </span>
                    )}
                  {selectedRunId === run.id && (
                    <span className="badge badge-reviewer">Selected</span>
                  )}
                </div>
                {run.status === 'FAILED' && run.error_message && (
                  <div className="ai-run-error">{run.error_message}</div>
                )}
                <div className="ai-run-actions">
                  {run.status === 'COMPLETE' && (
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => handleViewArtifact(run)}
                      disabled={artifactLoading}
                      title="View the enhanced artifact and select this run"
                    >
                      {selectedRunId === run.id ? 'Selected' : 'Use run'}
                    </button>
                  )}
                  {run.status === 'FAILED' && canEnhance && (
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => handleRetryRun(run.id)}
                      disabled={retryingRunId === run.id}
                      title="Retry this failed enhancement (creates a new run)"
                    >
                      {retryingRunId === run.id ? '…' : '↻ Retry'}
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
        {selectedRun && selectedRun.status === 'COMPLETE' && (
          <div className="ai-selected-run">
            <SectionHeader
              title={`Using enhancement Run #${selectedRun.id}`}
              meta={<SourceBadge sourceType="ENHANCED" runId={selectedRun.id} />}
            />
            {artifactLoading ? (
              <LoadingState label="Loading enhanced artifact…" />
            ) : artifactUrl ? (
              <FaceFigure
                imageUrl={artifactUrl}
                faces={enhancedFaces}
                alt={`Enhanced artifact for Run #${selectedRun.id}`}
                onImageError={() => {
                  setArtifactUrl(null);
                  setBannerError(
                    'The artifact preview expired. Select the run again to refresh it.'
                  );
                }}
              />
            ) : (
              <EmptyState
                title="Artifact preview unavailable"
                detail="The enhanced image could not be loaded."
              />
            )}
            {canEnhance && (
              <button
                type="button"
                className="btn btn-ai btn-sm"
                onClick={handleEnhancedDetect}
                disabled={enhancedDetecting}
                title="Run face detection on the selected enhanced artifact"
              >
                {enhancedDetecting
                  ? 'Detecting…'
                  : `Detect faces on Run #${selectedRun.id}`}
              </button>
            )}
            {enhancedResult && (
              <div className="ai-enhanced-result">
                <div className="ai-face-row">
                  <FaceStatusBadge status={enhancedResult.status} />
                  <span className="ai-face-meta">
                    {enhancedResult.status === 'COMPLETE'
                      ? `${enhancedResult.face_count ?? enhancedFaces.length} face(s) on enhanced source`
                      : 'Enhanced detection did not complete'}
                  </span>
                </div>
                {enhancedResult.status === 'COMPLETE' &&
                enhancedFaces.length === 0 ? (
                  <EmptyState
                    title="Face detection complete — 0 faces detected"
                    detail="This is a completed result on the enhanced source, not an error."
                  />
                ) : (
                  enhancedFaces.map((face) => (
                    <FaceRow
                      key={face.id}
                      face={face}
                      searching={searchingFaceId === face.id}
                      onFindSimilar={handleFindSimilar}
                    />
                  ))
                )}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="evidence-section">
        <SectionHeader title="Similarity results" meta="Needs human review" />
        {similarityError ? (
          <ErrorState detail={similarityError} onRetry={null} />
        ) : !similarity ? (
          <EmptyState
            title="No similarity search yet"
            detail="Use “Find similar” on any detected face above to search the opposite evidence type."
          />
        ) : !similarityCandidates ? (
          <ErrorState detail="Similarity search returned an unexpected response." onRetry={null} />
        ) : similarityCandidates.length === 0 ? (
          <EmptyState
            title="No similar candidates found"
            detail={
              similarityRestoredRunId !== null
                ? `Search using restored Face #${similarity.face.ordinal} · Restoration Run #${similarityRestoredRunId} completed with no candidates meeting the threshold.`
                : `Search on Face #${similarity.face.ordinal} completed with no candidates meeting the threshold.`
            }
          />
        ) : (
          <>
            <p className="ai-note">
              Showing faces similar to Face #{similarity.face.ordinal} from
              this photo. Scores are retrieval signals, not identity proof.
            </p>
            <p className="ai-note">{similarityQueryProvenance}</p>
            {similarityRestoredRunId !== null && (
              <p className="ai-note">
                Restored-face query: GFPGAN may synthesize facial detail. Results are
                investigation-support signals requiring human review, not identity evidence.
              </p>
            )}
            {similarityCandidates.map((candidate) => (
              <CandidateRow key={candidate.face_id} candidate={candidate} />
            ))}
          </>
        )}
      </div>
      <div className="evidence-section">
        <SectionHeader
          title="Face restoration (optional)"
          meta="ADMIN / REVIEWER"
        />
        <p className="ai-note">
          Restore one explicitly selected face with GFPGAN for
          downstream SFace / similarity investigation. Restoration
          consumes the Phase 3 derived image only and never modifies
          the original evidence.
        </p>
        <p className="ai-note">
          GFPGAN may synthesize facial detail. Restored-face results
          are investigation-support signals requiring human review,
          not identity evidence.
        </p>
        {normalFaces.length === 0 ? (
          <EmptyState
            title="No detected faces"
            detail="Detect faces above to select one for restoration."
          />
        ) : (
          <>
            <label className="ai-note" htmlFor="restore-face-select">
              Selected face
            </label>
            <select
              id="restore-face-select"
              value={selectedFaceId ?? ''}
              onChange={(e) => handleSelectFaceForRestoration(e.target.value)}
            >
              <option value="">Select a face…</option>
              {normalFaces.map((face) => (
                <option key={face.id} value={face.id}>
                  {`Face #${face.ordinal} · confidence ${Number(face.confidence).toFixed(2)}`}
                </option>
              ))}
            </select>
            {canEnhance && selectedFaceId && (
              <button
                type="button"
                className="btn btn-ai btn-sm"
                onClick={() => handleRestoreFace(selectedFaceId)}
                disabled={restoringFaceId !== null}
                title="Restore the selected face (creates a new run)"
              >
                {restoringFaceId ? 'Restoring…' : 'Restore This Face'}
              </button>
            )}
            {!canEnhance && (
              <p className="ai-note">
                Face restoration can be triggered by ADMIN or REVIEWER users.
              </p>
            )}
            {selectedFaceId && (
              restorationsLoading ? (
                <LoadingState label="Loading restoration runs…" />
              ) : restorations.length === 0 ? (
                <EmptyState
                  title="No restoration runs yet"
                  detail="Restore the selected face to create one."
                />
              ) : (
                <div className="ai-run-list">
                  {restorations.map((run) => (
                    <div key={run.id} className="ai-run-row">
                      <div className="ai-run-main">
                        <span className="ai-run-id">Run #{run.id}</span>
                        <RestorationStatusBadge status={run.status} />
                        {run.status === 'COMPLETE' &&
                          run.width > 0 &&
                          run.height > 0 && (
                            <span className="ai-run-dims">
                              {run.width}×{run.height}
                            </span>
                          )}
                        <span className="ai-run-dims">
                          {run.restored_geometry_kind === 'CANONICAL'
                            ? 'canonical geometry'
                            : 'mapped geometry'}
                        </span>
                        {selectedRestorationId === run.id && (
                          <span className="badge badge-reviewer">Selected</span>
                        )}
                      </div>
                      {run.status === 'FAILED' && run.error_message && (
                        <div className="ai-run-error">{run.error_message}</div>
                      )}
                      <div className="ai-run-actions">
                        {run.status === 'COMPLETE' && (
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            onClick={() => handleViewRestoration(selectedFaceId, run)}
                            disabled={restorationArtifactLoading}
                            title="View the restored face and select this run"
                          >
                            {selectedRestorationId === run.id ? 'Selected' : 'Use run'}
                          </button>
                        )}
                        {run.status === 'FAILED' && canEnhance && (
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            onClick={() => handleRetryRestoration(selectedFaceId, run.id)}
                            disabled={retryingRestorationId === run.id}
                            title="Retry this failed restoration (creates a new run)"
                          >
                            {retryingRestorationId === run.id ? '…' : '↻ Retry'}
                          </button>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )
            )}
            {selectedRestorationId && (
              <div className="ai-selected-run">
                <SectionHeader
                  title={`Using restoration Run #${selectedRestorationId}`}
                  meta="Restored face · explicit run"
                />
                {restorationArtifactLoading ? (
                  <LoadingState label="Loading restored artifact…" />
                ) : restorationArtifactUrl ? (
                  <div className="ai-face-figure">
                    <img
                      src={restorationArtifactUrl}
                      alt={`Restored face for Run #${selectedRestorationId}`}
                      className="ai-face-image"
                      onError={() => {
                        setRestorationArtifactUrl(null);
                        setBannerError(
                          'The artifact preview expired. Select the run again to refresh it.'
                        );
                      }}
                    />
                  </div>
                ) : (
                  <EmptyState
                    title="Artifact preview unavailable"
                    detail="The restored face could not be loaded."
                  />
                )}
                {canEnhance && (
                  <>
                    <button
                      type="button"
                      className="btn btn-ai btn-sm"
                      onClick={() => {
                        const run = restorations.find((r) => r.id === selectedRestorationId);
                        if (run) handleEmbedRestored(selectedFaceId, run);
                      }}
                      disabled={embeddingLoading}
                      title="Generate the SFace embedding for the selected restored face"
                    >
                      {embeddingLoading ? 'Embedding…' : 'Generate Embedding'}
                    </button>
                    {embeddingResult && (
                      <p className="ai-note">
                        {`Embedding #${embeddingResult.id} ready for Face #${selectedFaceId}. `}
                        <button
                          type="button"
                          className="btn btn-ai btn-sm"
                          onClick={() => {
                            const face = normalFaces.find((f) => f.id === selectedFaceId);
                            if (face) handleFindSimilarRestored(face, selectedRestorationId);
                          }}
                          disabled={searchingFaceId !== null}
                          title="Search for faces similar to this restored face"
                        >
                          {searchingFaceId ? '…' : 'Find Similar'}
                        </button>
                      </p>
                    )}
                  </>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </Modal>
  );
};

export default PhotoAIModal;
