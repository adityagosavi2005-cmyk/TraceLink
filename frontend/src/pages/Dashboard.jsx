import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import './Dashboard.css';

export const Dashboard = () => {
  const { user } = useAuth();
  const navigate = useNavigate();

  const getRoleDescription = (role) => {
    switch (role) {
      case 'ADMIN':
        return 'System Administrator. Access to audit logs, role modifications, and configurations.';
      case 'REVIEWER':
        return 'Sighting Auditor. Authorized to review and approve submitted sighting reports and case links.';
      case 'ORGANIZATION_MEMBER':
        return 'Active Investigator. Full access to create, update, and manage active missing person case files.';
      case 'REPORTER':
      default:
        return 'Sighting Submitter. Authorized to file sighting reports and view public bulletin details.';
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', textAlign: 'left' }}>
      {/* Banner */}
      <div className="dashboard-hero">
        <h1 className="hero-title">TraceLink System Overview</h1>
        <p className="hero-subtitle">
          Welcome to the TraceLink case resolution portal. This platform is designed to consolidate missing person reports, index sighting data points, and coordinate with verified organization members.
        </p>
      </div>

      <div className="dashboard-grid">
        {/* Investigator card */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', textAlign: 'left' }}>
          <h2 style={{ fontSize: '1.25rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>Active Session Credentials</h2>
          <div className="system-status-list">
            <div className="status-item">
              <span className="status-label">Investigator Name</span>
              <span className="status-value">{user?.name}</span>
            </div>
            <div className="status-item">
              <span className="status-label">Email Account</span>
              <span className="status-value">{user?.email}</span>
            </div>
            <div className="status-item">
              <span className="status-label">Assigned Security Role</span>
              <span className="status-value">{user?.role}</span>
            </div>
          </div>
          <div style={{ backgroundColor: 'var(--bg-primary)', padding: '0.75rem', borderRadius: 'var(--radius-sm)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <strong>Permissions Scope:</strong> {getRoleDescription(user?.role)}
          </div>
        </div>

        {/* System parameters card */}
        <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem', textAlign: 'left' }}>
          <h2 style={{ fontSize: '1.25rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>Quick Actions</h2>
          <div className="action-card-list">
            <div className="action-card" onClick={() => navigate('/cases')}>
              <h3>📁 View Case Files</h3>
              <p>Index, filter, and track ongoing missing-person investigations.</p>
            </div>
            <div className="action-card" onClick={() => navigate('/sightings')}>
              <h3>👁️ Sighting Reports</h3>
              <p>Review sightings submitted by public reporters or verification agents.</p>
            </div>
          </div>
          
          <div className="system-status-list" style={{ marginTop: '0.5rem' }}>
            <div className="status-item">
              <span className="status-label">Database Connection</span>
              <span className="status-value" style={{ color: 'var(--success)' }}>Online</span>
            </div>
            <div className="status-item">
              <span className="status-label">Alembic Migrations</span>
              <span className="status-value">df8103a83530 (Latest)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
