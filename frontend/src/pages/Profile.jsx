import React from 'react';
import { useAuth } from '../hooks/useAuth';
import './Dashboard.css';

export const Profile = () => {
  const { user } = useAuth();

  const getRoleBadge = (role) => {
    switch (role) {
      case 'ADMIN':
        return <span className="badge badge-admin">Administrator</span>;
      case 'REVIEWER':
        return <span className="badge badge-reviewer">Reviewer</span>;
      case 'ORGANIZATION_MEMBER':
        return <span className="badge badge-org">Org Member</span>;
      case 'REPORTER':
      default:
        return <span className="badge badge-reporter">Reporter</span>;
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', textAlign: 'left' }}>
      <div className="card profile-card">
        <h2 style={{ fontSize: '1.25rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem', marginBottom: '0.5rem' }}>
          Investigator Information
        </h2>
        
        <div className="profile-field">
          <span className="profile-label">Database User ID</span>
          <span className="profile-value" style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>
            {user?.id}
          </span>
        </div>

        <div className="profile-field">
          <span className="profile-label">Full Name</span>
          <span className="profile-value">{user?.name}</span>
        </div>

        <div className="profile-field">
          <span className="profile-label">Email Address</span>
          <span className="profile-value">{user?.email}</span>
        </div>

        <div className="profile-field">
          <span className="profile-label">Security Role</span>
          <span className="profile-value">
            {user && getRoleBadge(user.role)}
          </span>
        </div>
      </div>
    </div>
  );
};

export default Profile;
