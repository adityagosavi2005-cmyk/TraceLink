import React from 'react';

const CASE_TONES = {
  OPEN: 'badge-org',
  UNDER_REVIEW: 'badge-reviewer',
  RESOLVED: 'badge-success',
  CLOSED: 'badge-reporter',
};

const SIGHTING_TONES = {
  REPORTED: 'badge-org',
  UNDER_REVIEW: 'badge-reviewer',
  VERIFIED: 'badge-success',
  DISMISSED: 'badge-reporter',
};

const PHOTO_TONES = {
  READY: 'badge-success',
  FAILED: 'badge-admin',
  PROCESSING: 'badge-reviewer',
  UPLOADED: 'badge-org',
};

const FACE_TONES = {
  COMPLETE: 'badge-success',
  FAILED: 'badge-admin',
  PROCESSING: 'badge-reviewer',
  NOT_RUN: 'badge-reporter',
};

function toneFor(map, status, fallback = 'badge-reporter') {
  return map[status] || fallback;
}

export const CaseStatusBadge = ({ status }) => (
  <span className={`badge ${toneFor(CASE_TONES, status)}`}>{status}</span>
);

export const SightingStatusBadge = ({ status }) => (
  <span className={`badge ${toneFor(SIGHTING_TONES, status)}`}>{status}</span>
);

export const PhotoStatusBadge = ({ status }) => (
  <span className={`badge ${toneFor(PHOTO_TONES, status)}`}>
    {status === 'PROCESSING' ? 'Processing…' : status}
  </span>
);

export const RestorationStatusBadge = ({ status }) => {
  if (status === 'COMPLETE') {
    return <span className="badge badge-success">Restoration complete</span>;
  }
  if (status === 'FAILED') {
    return <span className="badge badge-admin">Restoration failed</span>;
  }
  if (status === 'PROCESSING') {
    return <span className="badge badge-reviewer">Restoring…</span>;
  }
  return <span className="badge badge-reporter">{status || 'Unknown'}</span>;
};

export const EnhancementStatusBadge = ({ status }) => {
  if (status === 'COMPLETE') {
    return <span className="badge badge-success">Enhancement complete</span>;
  }
  if (status === 'FAILED') {
    return <span className="badge badge-admin">Enhancement failed</span>;
  }
  if (status === 'PROCESSING') {
    return <span className="badge badge-reviewer">Enhancing…</span>;
  }
  return <span className="badge badge-reporter">{status || 'Unknown'}</span>;
};

export const SourceBadge = ({ sourceType, runId }) => {
  if (sourceType === 'ENHANCED') {
    return (
      <span className="badge badge-reviewer" title="Produced from an explicitly selected enhancement run">
        {runId ? `Enhanced · Run #${runId}` : 'Enhanced'}
      </span>
    );
  }
  return (
    <span className="badge badge-org" title="Produced from the normal Phase 3 derived image">
      Derived
    </span>
  );
};

export const FaceStatusBadge = ({ status, faceCount }) => {
  if (status === 'COMPLETE') {
    return (
      <span className="badge badge-success">Faces: {faceCount ?? 0}</span>
    );
  }
  if (status === 'FAILED') {
    return <span className="badge badge-admin">Face scan failed</span>;
  }
  if (status === 'PROCESSING') {
    return <span className="badge badge-reviewer">Scanning…</span>;
  }
  return <span className="badge badge-reporter">No face scan</span>;
};

export default FaceStatusBadge;
