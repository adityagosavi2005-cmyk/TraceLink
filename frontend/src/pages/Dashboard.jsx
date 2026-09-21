import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { caseService, sightingService } from '../services/api';
import { PageHeader, SectionHeader, StatCard } from '../components/Workspace';
import { CaseStatusBadge, SightingStatusBadge } from '../components/StatusBadge';
import { RoleBadge } from '../components/RoleBadge';
import { LoadingState, EmptyState } from '../components/StateBlocks';
import './Dashboard.css';

const ROLE_DESCRIPTIONS = {
  ADMIN: 'System administrator. Full access across cases, sightings, evidence, and face-detection operations.',
  REVIEWER: 'Reviewer. Can view cases and sightings, and trigger face detection for investigator review.',
  ORGANIZATION_MEMBER: 'Organization member. Can create and manage cases and report sightings within their organizations.',
  REPORTER: 'Reporter. Can file cases and sighting reports and upload evidence photos to their own cases.',
};

const formatDate = (dateStr) => {
  if (!dateStr) return '—';
  const date = new Date(dateStr);
  return date.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
};

export const Dashboard = () => {
  const { user } = useAuth();
  const navigate = useNavigate();

  const [cases, setCases] = useState([]);
  const [sightings, setSightings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchOverview = async () => {
      setLoading(true);
      setError(null);
      try {
        const [caseData, sightingData] = await Promise.all([
          caseService.getCases(),
          sightingService.listAllSightings(),
        ]);
        setCases(Array.isArray(caseData) ? caseData : []);
        setSightings(Array.isArray(sightingData) ? sightingData : []);
      } catch (err) {
        setError(
          err.response?.data?.detail || 'Failed to load workspace overview.'
        );
      } finally {
        setLoading(false);
      }
    };
    fetchOverview();
  }, []);

  const openCases = cases.filter((c) => c.status === 'OPEN').length;
  const underReview = cases.filter((c) => c.status === 'UNDER_REVIEW').length;
  const recentCases = [...cases]
    .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
    .slice(0, 5);
  const recentSightings = [...sightings]
    .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
    .slice(0, 5);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', textAlign: 'left' }}>
      <PageHeader
        title={`Welcome, ${user?.name || 'Investigator'}`}
        subtitle="Investigation workspace — cases, sightings, and review activity at a glance."
        actions={<RoleBadge role={user?.role} />}
      />

      {loading ? (
        <LoadingState label="Loading workspace overview…" />
      ) : error ? (
        <div className="alert alert-error" role="alert">{error}</div>
      ) : (
        <>
          <div className="stat-grid" aria-label="Workspace summary">
            <StatCard label="Cases in scope" value={cases.length} tone="info" />
            <StatCard label="Open cases" value={openCases} tone="success" />
            <StatCard label="Under review" value={underReview} tone="warning" />
            <StatCard label="Sightings reported" value={sightings.length} tone="info" />
          </div>

          <div className="dashboard-grid">
            <div className="card">
              <SectionHeader
                title="Recent cases"
                meta={`${recentCases.length} of ${cases.length}`}
                actions={
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => navigate('/cases')}
                  >
                    All cases
                  </button>
                }
              />
              {recentCases.length === 0 ? (
                <EmptyState
                  title="No cases yet"
                  detail="Cases you can access will appear here."
                />
              ) : (
                <div className="recent-list">
                  {recentCases.map((c) => (
                    <button
                      key={c.id}
                      type="button"
                      className="recent-item"
                      onClick={() => navigate('/cases')}
                    >
                      <span className="recent-id">#{c.id}</span>
                      <span className="recent-main">
                        <span className="recent-title">{c.title}</span>
                        <span className="recent-sub">{formatDate(c.created_at)}</span>
                      </span>
                      <CaseStatusBadge status={c.status} />
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="card">
              <SectionHeader
                title="Recent sightings"
                meta={`${recentSightings.length} of ${sightings.length}`}
                actions={
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => navigate('/sightings')}
                  >
                    All sightings
                  </button>
                }
              />
              {recentSightings.length === 0 ? (
                <EmptyState
                  title="No sightings yet"
                  detail="Reported sightings will appear here."
                />
              ) : (
                <div className="recent-list">
                  {recentSightings.map((s) => (
                    <button
                      key={s.id}
                      type="button"
                      className="recent-item"
                      onClick={() => navigate('/sightings')}
                    >
                      <span className="recent-id">#{s.id}</span>
                      <span className="recent-main">
                        <span className="recent-title">{s.location_text}</span>
                        <span className="recent-sub">
                          Case #{s.case_id} · {formatDate(s.sighting_at)}
                        </span>
                      </span>
                      <SightingStatusBadge status={s.status} />
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="card">
            <SectionHeader title="Your access" />
            <div className="system-status-list">
              <div className="status-item">
                <span className="status-label">Signed in as</span>
                <span className="status-value">{user?.email}</span>
              </div>
              <div className="status-item">
                <span className="status-label">Role</span>
                <span className="status-value">{user?.role}</span>
              </div>
            </div>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '0.75rem' }}>
              {ROLE_DESCRIPTIONS[user?.role] || ROLE_DESCRIPTIONS.REPORTER}
            </p>
          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard;
