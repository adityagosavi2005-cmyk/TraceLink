import React from 'react';

export const LoadingState = ({ label }) => (
  <div className="state-block" role="status" aria-live="polite">
    <span className="spinner" aria-hidden="true" />
    <span>{label || 'Loading…'}</span>
  </div>
);

export const EmptyState = ({ title, detail, action }) => (
  <div className="state-block">
    <span className="state-title">{title}</span>
    {detail ? <span>{detail}</span> : null}
    {action || null}
  </div>
);

export const ErrorState = ({ detail, onRetry }) => (
  <div className="state-block" role="alert">
    <span className="state-title">Something went wrong</span>
    <span>{detail || 'Failed to load. Please try again.'}</span>
    {onRetry ? (
      <button type="button" className="btn btn-secondary btn-sm" onClick={onRetry}>
        Retry
      </button>
    ) : null}
  </div>
);

export default EmptyState;
