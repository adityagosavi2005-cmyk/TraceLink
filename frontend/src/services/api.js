import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request Interceptor: Attach JWT Token if present in localStorage
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response Interceptor: Handle 401 Unauthorized globally
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Dispatches custom event to trigger logout inside the AuthContext
      window.dispatchEvent(new Event('auth-unauthorized'));
    }
    return Promise.reject(error);
  }
);

export const authService = {
  register: async (name, email, password) => {
    const response = await api.post('/auth/register', { name, email, password });
    return response.data;
  },

  login: async (email, password) => {
    const response = await api.post('/auth/login', { email, password });
    return response.data;
  },

  getCurrentUser: async () => {
    const response = await api.get('/auth/me');
    return response.data;
  },

  createAdmin: async (data) => {
    const response = await api.post('/auth/admin', data);
    return response.data;
  },
};

export const caseService = {
  getCases: async () => {
    const response = await api.get('/cases');
    return response.data;
  },

  getCase: async (id) => {
    const response = await api.get(`/cases/${id}`);
    return response.data;
  },

  createCase: async (caseData) => {
    const response = await api.post('/cases', caseData);
    return response.data;
  },

  updateCase: async (id, data) => {
    const response = await api.patch(`/cases/${id}`, data);
    return response.data;
  },

  deleteCase: async (id) => {
    const response = await api.delete(`/cases/${id}`);
    return response.data;
  },
};

export default api;

