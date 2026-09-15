import React, { useState } from 'react';
import { useAuth } from '../hooks/useAuth';
import { Navigate } from 'react-router-dom';
import { authService } from '../services/api';
import './Dashboard.css';

export const Admin = () => {
  const { user, loading: authLoading } = useAuth();
  
  // Form states
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  
  // UX Feedback states
  const [loading, setLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // Authorization Protection
  if (authLoading) {
    return null; // Let the layout/App load first
  }

  if (!user || user.role !== 'ADMIN') {
    return <Navigate to="/" replace />;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMessage(null);
    setSuccessMessage(null);

    if (!name.trim() || !email.trim() || !password.trim()) {
      setErrorMessage('Please fill in all the required fields.');
      return;
    }

    setLoading(true);

    try {
      await authService.createAdmin({
        name: name.trim(),
        email: email.trim(),
        password: password
      });

      setSuccessMessage('Administrator account created successfully!');
      setName('');
      setEmail('');
      setPassword('');
    } catch (err) {
      console.error('Error creating administrator:', err);
      if (err.response) {
        const { status, data } = err.response;
        if (status === 400 || status === 409) {
          setErrorMessage(data?.detail || 'An account with this email address already exists.');
        } else if (status === 403) {
          setErrorMessage('You do not have permission to create administrator accounts.');
        } else if (status === 422) {
          if (data?.detail && Array.isArray(data.detail)) {
            const messages = data.detail.map(errObj => `${errObj.loc.slice(1).join('.')}: ${errObj.msg}`).join(', ');
            setErrorMessage(`Validation error: ${messages}`);
          } else {
            setErrorMessage('Invalid input format provided. Please check the entered fields.');
          }
        } else {
          setErrorMessage('A server error occurred. Please try again later.');
        }
      } else {
        setErrorMessage('Failed to connect to the server. Please verify your connection.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', textAlign: 'left' }}>
      <div className="card profile-card" style={{ maxWidth: '600px', width: '100%' }}>
        <h2 style={{ fontSize: '1.25rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem', marginBottom: '1.5rem' }}>
          Administrator Management
        </h2>
        
        {successMessage && (
          <div className="alert alert-success">
            {successMessage}
          </div>
        )}
        
        {errorMessage && (
          <div className="alert alert-error">
            {errorMessage}
          </div>
        )}

        <form onSubmit={handleSubmit} noValidate>
          <div className="form-group">
            <label className="form-label" htmlFor="admin-name">Full Name</label>
            <input
              id="admin-name"
              type="text"
              className="form-control"
              placeholder="e.g. Inspector Miller"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={loading}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="admin-email">Email Address</label>
            <input
              id="admin-email"
              type="email"
              className="form-control"
              placeholder="e.g. miller@tracelink.gov"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={loading}
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="admin-password">Password</label>
            <input
              id="admin-password"
              type="password"
              className="form-control"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={loading}
              required
            />
          </div>

          <div style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'flex-end' }}>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading}
              style={{ minWidth: '160px' }}
            >
              {loading ? 'Creating...' : 'Create Admin Account'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Admin;
