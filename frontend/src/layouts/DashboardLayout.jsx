import React from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import './DashboardLayout.css';

export const DashboardLayout = () => {
  const { user, logoutUser } = useAuth();
  const location = useLocation();

  const getPageTitle = () => {
    switch (location.pathname) {
      case '/':
        return 'Dashboard Overview';
      case '/cases':
        return 'Case File Management';
      case '/sightings':
        return 'Sighting Reports';
      case '/profile':
        return 'Investigator Profile';
      case '/admin':
        return 'Administrator Management';
      default:
        return 'TraceLink Platform';
    }
  };

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
    <div className="layout-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-logo">
            <div className="brand-dot" />
            <span>TraceLink</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <NavLink 
            to="/" 
            className={({ isActive }) => `nav-link-item ${isActive ? 'active' : ''}`}
            end
          >
            <span style={{ fontSize: '1.2rem' }}>📁</span>
            <span>Dashboard</span>
          </NavLink>

          <NavLink 
            to="/cases" 
            className={({ isActive }) => `nav-link-item ${isActive ? 'active' : ''}`}
          >
            <span style={{ fontSize: '1.2rem' }}>🔍</span>
            <span>Cases</span>
          </NavLink>

          <NavLink 
            to="/sightings" 
            className={({ isActive }) => `nav-link-item ${isActive ? 'active' : ''}`}
          >
            <span style={{ fontSize: '1.2rem' }}>👁️</span>
            <span>Sightings</span>
          </NavLink>

          <NavLink 
            to="/profile" 
            className={({ isActive }) => `nav-link-item ${isActive ? 'active' : ''}`}
          >
            <span style={{ fontSize: '1.2rem' }}>👤</span>
            <span>Profile</span>
          </NavLink>

          {user?.role === 'ADMIN' && (
            <NavLink 
              to="/admin" 
              className={({ isActive }) => `nav-link-item ${isActive ? 'active' : ''}`}
            >
              <span style={{ fontSize: '1.2rem' }}>🛡️</span>
              <span>Admin Panel</span>
            </NavLink>
          )}
        </nav>

        {/* User Identity Panel at Sidebar Bottom */}
        {user && (
          <div className="sidebar-footer">
            <div className="user-profile-summary">
              <span className="user-name" title={user.name}>{user.name}</span>
              <div className="user-role-badge">
                {getRoleBadge(user.role)}
              </div>
            </div>
            <button className="btn btn-secondary btn-logout" onClick={logoutUser}>
              <span>🚪</span>
              <span>Sign Out</span>
            </button>
          </div>
        )}
      </aside>

      {/* Main Panel */}
      <main className="main-content">
        <header className="top-bar">
          <div className="page-title-area">
            <h2>{getPageTitle()}</h2>
          </div>
          <div className="status-indicator-area">
            <span className="status-dot" />
            <span>Active Session Verified</span>
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
