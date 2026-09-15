import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { authService } from '../services/api';
import './Login.css';

export const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const { loginUser } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (!email || !password) {
      setError('Please provide both email and password.');
      return;
    }

    setIsSubmitting(true);
    try {
      // 1. Call login endpoint to fetch access token
      const data = await authService.login(email, password);
      
      // 2. Load token into session context
      await loginUser(data.access_token);
      
      // 3. Forward to secure dashboard
      navigate('/');
    } catch (err) {
      console.error('Login error:', err);
      if (err.response && err.response.data && err.response.data.detail) {
        const detail = err.response.data.detail;
        if (typeof detail === 'string') {
          setError(detail);
        } else if (Array.isArray(detail)) {
          const message = detail
            .map((errItem) => (errItem && errItem.msg) || JSON.stringify(errItem))
            .join(', ');
          setError(message || 'Validation error occurred.');
        } else if (typeof detail === 'object' && detail !== null) {
          setError(detail.msg || JSON.stringify(detail));
        } else {
          setError('Login failed. Please check backend connection and credentials.');
        }
      } else {
        setError('Login failed. Please check backend connection and credentials.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <div className="auth-logo">
            <div className="auth-logo-dot" />
            <span>TraceLink</span>
          </div>
          <h1 className="auth-title">Sign In to TraceLink</h1>
          <p className="auth-subtitle">Case Management & Missing Person Investigation</p>
        </div>

        {error && (
          <div className="alert alert-error">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="email-input">Email Address</label>
            <input
              id="email-input"
              type="email"
              className="form-control"
              placeholder="investigator@tracelink.org"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              disabled={isSubmitting}
              autoComplete="email"
              required
            />
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="password-input">Password</label>
            <input
              id="password-input"
              type="password"
              className="form-control"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={isSubmitting}
              autoComplete="current-password"
              required
            />
          </div>

          <button
            type="submit"
            className="btn btn-primary btn-block"
            disabled={isSubmitting}
            style={{ marginTop: '1.5rem' }}
          >
            {isSubmitting ? (
              <>
                <span className="spinner" />
                <span>Verifying credentials...</span>
              </>
            ) : (
              <span>Sign In</span>
            )}
          </button>
        </form>

        <div className="auth-footer">
          Don't have an investigator account?{' '}
          <Link to="/register" style={{ fontWeight: 500 }}>
            Register here
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Login;
