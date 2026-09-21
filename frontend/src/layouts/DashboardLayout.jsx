import React from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { RoleBadge } from '../components/RoleBadge';
import './DashboardLayout.css';

const ICONS = {
  dashboard: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <rect x="3" y="3" width="7" height="9" rx="1.5" />
      <rect x="14" y="3" width="7" height="5" rx="1.5" />
      <rect x="14" y="12" width="7" height="9" rx="1.5" />
      <rect x="3" y="16" width="7" height="5" rx="1.5" />
    </svg>
  ),
  cases: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" strokeLinecap="round" />
    </svg>
  ),
  sightings: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path d="M2 12s3.5-6.5 10-6.5S22 12 22 12s-3.5 6.5-10 6.5S2 12 2 12Z" />
      <circle cx="12" cy="12" r="2.8" />
    </svg>
  ),
  profile: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <circle cx="12" cy="8" r="3.6" />
      <path d="M4.5 20c1.4-3.6 4.2-5.4 7.5-5.4s6.1 1.8 7.5 5.4" strokeLinecap="round" />
    </svg>
  ),
  admin: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path d="M12 3 4.5 6v5.5c0 4.6 3.2 8 7.5 9.5 4.3-1.5 7.5-4.9 7.5-9.5V6L12 3Z" strokeLinejoin="round" />
    </svg>
  ),
  logout: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <path d="M14 4H6v16h8M10 12h11m0 0-3.5-3.5M21 12l-3.5 3.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
};

export const DashboardLayout = () => {
  const { user, logoutUser } = useAuth();
  const location = useLocation();

  const getPageTitle = () => {
    switch (location.pathname) {
      case '/':
        return 'Dashboard';
      case '/cases':
        return 'Cases';
      case '/sightings':
        return 'Sightings';
      case '/profile':
        return 'Profile';
      case '/admin':
        return 'Administration';
      default:
        return 'TraceLink';
    }
  };

  const links = [
    { to: '/', label: 'Dashboard', icon: ICONS.dashboard, end: true },
    { to: '/cases', label: 'Cases', icon: ICONS.cases },
    { to: '/sightings', label: 'Sightings', icon: ICONS.sightings },
    { to: '/profile', label: 'Profile', icon: ICONS.profile },
  ];
  if (user?.role === 'ADMIN') {
    links.push({ to: '/admin', label: 'Admin', icon: ICONS.admin });
  }

  return (
    <div className="layout-container">
      <a className="skip-link" href="#main-content">Skip to content</a>
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-logo">
            <div className="brand-dot" aria-hidden="true" />
            <span>TraceLink</span>
          </div>
          {user && (
            <button
              type="button"
              className="btn btn-secondary btn-sm sidebar-logout-compact"
              onClick={logoutUser}
            >
              {ICONS.logout}
              <span>Sign out</span>
            </button>
          )}
        </div>

        <nav className="sidebar-nav" aria-label="Primary">
          {links.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) => `nav-link-item ${isActive ? 'active' : ''}`}
              end={link.end}
            >
              {link.icon}
              <span>{link.label}</span>
            </NavLink>
          ))}
        </nav>

        {/* User Identity Panel at Sidebar Bottom */}
        {user && (
          <div className="sidebar-footer">
            <div className="user-profile-summary">
              <span className="user-name" title={user.name}>{user.name}</span>
              <div className="user-role-badge">
                <RoleBadge role={user.role} />
              </div>
            </div>
            <button className="btn btn-secondary btn-logout" onClick={logoutUser}>
              {ICONS.logout}
              <span>Sign Out</span>
            </button>
          </div>
        )}
      </aside>

      {/* Main Panel */}
      <main className="main-content" id="main-content">
        <header className="top-bar">
          <div className="page-title-area">
            <h2>{getPageTitle()}</h2>
          </div>
          <div className="status-indicator-area">
            <span className="status-dot" aria-hidden="true" />
            <span>Signed in{user ? ` as ${user.name}` : ''}</span>
          </div>
        </header>

        {/* Dynamic Outlet for router views */}
        <div className="page-content-wrapper">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default DashboardLayout;
