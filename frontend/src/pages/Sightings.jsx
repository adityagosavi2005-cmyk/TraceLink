import React from 'react';
import './Dashboard.css';

export const Sightings = () => {
  return (
    <div className="coming-soon-container">
      <div className="coming-soon-icon">👁️</div>
      <h1 className="coming-soon-title">Sighting Reports</h1>
      <p className="coming-soon-text">
        The Sighting Report tracking module is in planning. This will allow verified field agents and public reporters to log sighting locations, enter timestamps, and reference corresponding case files.
      </p>
      <div style={{ display: 'inline-flex', padding: '0.5rem 1rem', border: '1px dashed var(--border-color)', borderRadius: 'var(--radius-md)', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
        API routing hooks ready: `/auth/me` integration validates reporting privileges.
      </div>
    </div>
  );
};

export default Sightings;
