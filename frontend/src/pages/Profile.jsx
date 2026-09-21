import React, { useState, useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';
import { orgService } from '../services/api';
import { PageHeader, SectionHeader } from '../components/Workspace';
import { RoleBadge } from '../components/RoleBadge';
import { LoadingState, EmptyState } from '../components/StateBlocks';
import './Dashboard.css';

const ROLE_SCOPES = {
  ADMIN: 'Full access: cases, sightings, evidence, face detection, and administration.',
  REVIEWER: 'Can view cases and sightings, and trigger face detection for investigator review.',
  ORGANIZATION_MEMBER: 'Can create cases, manage organization case files, and report sightings.',
  REPORTER: 'Can file cases and sighting reports, and upload evidence to owned cases.',
};

export const Profile = () => {
  const { user } = useAuth();
  const [orgs, setOrgs] = useState([]);
  const [myRoles, setMyRoles] = useState({});
  const [loadingOrgs, setLoadingOrgs] = useState(true);

  useEffect(() => {
    const fetchOrgs = async () => {
      try {
        const data = await orgService.listOrganizations();
        const list = Array.isArray(data) ? data : [];
        setOrgs(list);
        const roles = {};
        await Promise.all(
          list.map(async (org) => {
            try {
              const members = await orgService.listMembers(org.id);
              const mine = (Array.isArray(members) ? members : []).find(
                (m) => m.user_id === user?.id
              );
              if (mine) roles[org.id] = mine.role;
            } catch (err) {
              // Per-org membership may be restricted; skip silently.
            }
          })
        );
        setMyRoles(roles);
      } catch (err) {
        setOrgs([]);
      } finally {
        setLoadingOrgs(false);
      }
    };
    if (user) fetchOrgs();
  }, [user]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', textAlign: 'left' }}>
      <PageHeader
        title="Profile"
        subtitle="Account identity, role, and organization memberships."
        actions={<RoleBadge role={user?.role} />}
      />

      <div className="card profile-card" style={{ maxWidth: '720px' }}>
        <SectionHeader title="Account" />
        <div className="profile-field">
          <span className="profile-label">User ID</span>
          <span className="profile-value mono">#{user?.id}</span>
        </div>
        <div className="profile-field">
          <span className="profile-label">Full name</span>
          <span className="profile-value">{user?.name}</span>
        </div>
        <div className="profile-field">
          <span className="profile-label">Email</span>
          <span className="profile-value">{user?.email}</span>
        </div>
        <div className="profile-field">
          <span className="profile-label">Role</span>
          <span className="profile-value">
            <RoleBadge role={user?.role} />
          </span>
        </div>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '0.75rem' }}>
          {ROLE_SCOPES[user?.role] || ROLE_SCOPES.REPORTER}
        </p>
      </div>

      <div className="card profile-card" style={{ maxWidth: '720px' }}>
        <SectionHeader
          title="Organizations"
          meta={loadingOrgs ? '…' : `${orgs.length} visible`}
        />
        {loadingOrgs ? (
          <LoadingState label="Loading organizations…" />
        ) : orgs.length === 0 ? (
          <EmptyState
            title="No organizations"
            detail="You are not a member of any organization, or none are visible to your role."
          />
        ) : (
          <div className="recent-list">
            {orgs.map((org) => (
              <div key={org.id} className="recent-item" style={{ cursor: 'default' }}>
                <span className="recent-id">#{org.id}</span>
                <span className="recent-main">
                  <span className="recent-title">{org.name}</span>
                  <span className="recent-sub">
                    {myRoles[org.id] ? `Your role: ${myRoles[org.id]}` : 'Membership details unavailable'}
                  </span>
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Profile;
