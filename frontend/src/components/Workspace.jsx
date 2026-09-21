import React from 'react';
import { PhotoStatusBadge, FaceStatusBadge } from './StatusBadge';

export const PageHeader = ({ title, subtitle, actions }) => (
  <div className="page-header">
    <div>
      <h1 className="page-title">{title}</h1>
      {subtitle ? <p className="page-subtitle">{subtitle}</p> : null}
    </div>
    {actions ? <div className="page-actions">{actions}</div> : null}
  </div>
);

export const SectionHeader = ({ title, meta, actions }) => (
  <div className="section-header">
    <h3 className="section-title">
      {title}
      {meta ? <span className="section-meta">{meta}</span> : null}
    </h3>
    {actions || null}
  </div>
);

export const StatCard = ({ label, value, tone }) => (
  <div className="stat-card">
    <span className="stat-value">{value}</span>
    <span className="stat-label">{label}</span>
    {tone ? <span className={`stat-dot stat-dot-${tone}`} aria-hidden="true" /> : null}
  </div>
);

export const PhotoCard = ({
  photo,
  alt,
  canDelete,
  onDelete,
  deleting,
  busy,
  canDetect,
  detecting,
  onDetect,
  onRedetect,
  onRetryProcessing,
  retrying,
  showRetry,
}) => {
  const faceStatus = photo.face_detection_status || 'NOT_RUN';
  const hasFaceResult = faceStatus && faceStatus !== 'NOT_RUN';
  return (
    <div className="photo-card">
      <div className="photo-thumb-wrap">
        <img
          src={photo.view_url}
          alt={alt}
          className="photo-thumb"
          loading="lazy"
        />
        {canDelete && (
          <button
            type="button"
            className="btn btn-danger btn-sm photo-delete-btn"
            onClick={() => onDelete(photo.id)}
            disabled={deleting || busy}
            aria-label={`Delete photo ${photo.id}`}
            title="Delete photo"
          >
            {deleting ? '…' : '✕'}
          </button>
        )}
      </div>
      <div className="photo-status-rows">
        <div className="photo-status-row">
          <PhotoStatusBadge status={photo.processing_status} />
          {showRetry && (
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => onRetryProcessing(photo.id)}
              disabled={retrying || busy}
              title="Retry processing"
            >
              {retrying ? '…' : '↻ Retry'}
            </button>
          )}
        </div>
        {photo.processing_status === 'FAILED' && photo.processing_error && (
          <div className="photo-error">{photo.processing_error}</div>
        )}
        <div className="photo-status-row">
          <FaceStatusBadge status={faceStatus} faceCount={photo.face_count} />
          {canDetect && photo.processing_status === 'READY' && !hasFaceResult && (
            <button
              type="button"
              className="btn btn-ai btn-sm"
              onClick={() => onDetect(photo.id)}
              disabled={detecting || busy}
              title="Run face detection"
            >
              {detecting ? '…' : 'Detect faces'}
            </button>
          )}
          {canDetect && photo.processing_status === 'READY' && hasFaceResult && (
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => onRedetect(photo.id)}
              disabled={detecting || busy}
              title="Re-run face detection (previous runs are kept)"
            >
              {detecting ? '…' : '↻ Re-detect'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default PageHeader;
