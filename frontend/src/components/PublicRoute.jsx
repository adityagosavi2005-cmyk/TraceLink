import React from 'react';
import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

export const PublicRoute = () => {
  const { user, loading } = useAuth();

  if (loading) {
    return null; // Silent load on auth pages
  }

  return !user ? <Outlet /> : <Navigate to="/" replace />;
};

export default PublicRoute;
