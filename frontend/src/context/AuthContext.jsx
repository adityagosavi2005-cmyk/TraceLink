import React, { createContext, useState, useEffect, useCallback } from 'react';
import { authService } from '../services/api';

export const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(localStorage.getItem('token'));
  const [loading, setLoading] = useState(true);

  const logoutUser = useCallback(() => {
    localStorage.removeItem('token');
    setToken(null);
    setUser(null);
  }, []);

  const loginUser = useCallback(async (newToken) => {
    localStorage.setItem('token', newToken);
    setToken(newToken);
    try {
      const userData = await authService.getCurrentUser();
      setUser(userData);
      return userData;
    } catch (err) {
      logoutUser();
      throw err;
    }
  }, [logoutUser]);

  // Verify token on mount or token change
  useEffect(() => {
    const verifyToken = async () => {
      if (token) {
        try {
          const userData = await authService.getCurrentUser();
          setUser(userData);
        } catch (err) {
          console.error("Token verification failed:", err);
          logoutUser();
        }
      } else {
        setUser(null);
      }
      setLoading(false);
    };

    verifyToken();
  }, [token, logoutUser]);

  // Listen for the global auth-unauthorized event
  useEffect(() => {
    const handleUnauthorized = () => {
      logoutUser();
    };

    window.addEventListener('auth-unauthorized', handleUnauthorized);
    return () => {
      window.removeEventListener('auth-unauthorized', handleUnauthorized);
    };
  }, [logoutUser]);

  const value = {
    user,
    token,
    loading,
    loginUser,
    logoutUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
