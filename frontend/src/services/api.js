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

export const orgService = {
  listOrganizations: async () => {
    const response = await api.get('/organizations');
    return response.data;
  },

  getOrganization: async (id) => {
    const response = await api.get(`/organizations/${id}`);
    return response.data;
  },

  listMembers: async (orgId) => {
    const response = await api.get(`/organizations/${orgId}/members`);
    return response.data;
  },
};

export const photoService = {
  listPhotos: async (caseId) => {
    const response = await api.get(`/cases/${caseId}/photos`);
    return response.data;
  },

  getPhoto: async (caseId, photoId) => {
    const response = await api.get(`/cases/${caseId}/photos/${photoId}`);
    return response.data;
  },

  uploadPhoto: async (caseId, file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post(`/cases/${caseId}/photos`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  deletePhoto: async (caseId, photoId) => {
    const response = await api.delete(`/cases/${caseId}/photos/${photoId}`);
    return response.data;
  },

  retryPhoto: async (caseId, photoId) => {
    const response = await api.post(`/cases/${caseId}/photos/${photoId}/retry`);
    return response.data;
  },

  detectFaces: async (caseId, photoId, enhancementRunId = null) => {
    const query =
      enhancementRunId !== null && enhancementRunId !== undefined
        ? `?enhancement_run_id=${encodeURIComponent(enhancementRunId)}`
        : '';
    const response = await api.post(
      `/cases/${caseId}/photos/${photoId}/faces/detect${query}`
    );
    return response.data;
  },

  redetectFaces: async (caseId, photoId, enhancementRunId = null) => {
    const query =
      enhancementRunId !== null && enhancementRunId !== undefined
        ? `?enhancement_run_id=${encodeURIComponent(enhancementRunId)}`
        : '';
    const response = await api.post(
      `/cases/${caseId}/photos/${photoId}/faces/redetect${query}`
    );
    return response.data;
  },

  getFaces: async (caseId, photoId) => {
    const response = await api.get(`/cases/${caseId}/photos/${photoId}/faces`);
    return response.data;
  },
};

export const sightingService = {
  listSightings: async (caseId) => {
    const response = await api.get(`/cases/${caseId}/sightings`);
    return response.data;
  },

  listAllSightings: async () => {
    const response = await api.get('/sightings');
    return response.data;
  },

  getSighting: async (caseId, sightingId) => {
    const response = await api.get(`/cases/${caseId}/sightings/${sightingId}`);
    return response.data;
  },

  createSighting: async (caseId, sightingData) => {
    const response = await api.post(`/cases/${caseId}/sightings`, sightingData);
    return response.data;
  },

  updateSighting: async (caseId, sightingId, data) => {
    const response = await api.patch(`/cases/${caseId}/sightings/${sightingId}`, data);
    return response.data;
  },

  deleteSighting: async (caseId, sightingId) => {
    const response = await api.delete(`/cases/${caseId}/sightings/${sightingId}`);
    return response.data;
  },

  listSightingPhotos: async (caseId, sightingId) => {
    const response = await api.get(`/cases/${caseId}/sightings/${sightingId}/photos`);
    return response.data;
  },

  uploadSightingPhoto: async (caseId, sightingId, file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post(
      `/cases/${caseId}/sightings/${sightingId}/photos`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } },
    );
    return response.data;
  },

  deleteSightingPhoto: async (caseId, sightingId, photoId) => {
    const response = await api.delete(
      `/cases/${caseId}/sightings/${sightingId}/photos/${photoId}`
    );
    return response.data;
  },

  retrySightingPhoto: async (caseId, sightingId, photoId) => {
    const response = await api.post(
      `/cases/${caseId}/sightings/${sightingId}/photos/${photoId}/retry`
    );
    return response.data;
  },

  detectSightingFaces: async (
    caseId,
    sightingId,
    photoId,
    enhancementRunId = null
  ) => {
    const query =
      enhancementRunId !== null && enhancementRunId !== undefined
        ? `?enhancement_run_id=${encodeURIComponent(enhancementRunId)}`
        : '';
    const response = await api.post(
      `/cases/${caseId}/sightings/${sightingId}/photos/${photoId}/faces/detect${query}`
    );
    return response.data;
  },

  redetectSightingFaces: async (
    caseId,
    sightingId,
    photoId,
    enhancementRunId = null
  ) => {
    const query =
      enhancementRunId !== null && enhancementRunId !== undefined
        ? `?enhancement_run_id=${encodeURIComponent(enhancementRunId)}`
        : '';
    const response = await api.post(
      `/cases/${caseId}/sightings/${sightingId}/photos/${photoId}/faces/redetect${query}`
    );
    return response.data;
  },

  getSightingFaces: async (caseId, sightingId, photoId) => {
    const response = await api.get(
      `/cases/${caseId}/sightings/${sightingId}/photos/${photoId}/faces`
    );
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

export const enhancementService = {
  triggerCaseEnhancement: async (caseId, photoId) => {
    const response = await api.post(
      `/cases/${caseId}/photos/${photoId}/enhancements`
    );
    return response.data;
  },

  listCaseEnhancements: async (caseId, photoId) => {
    const response = await api.get(
      `/cases/${caseId}/photos/${photoId}/enhancements`
    );
    return response.data;
  },

  getCaseEnhancement: async (caseId, photoId, runId) => {
    const response = await api.get(
      `/cases/${caseId}/photos/${photoId}/enhancements/${runId}`
    );
    return response.data;
  },

  retryCaseEnhancement: async (caseId, photoId, runId) => {
    const response = await api.post(
      `/cases/${caseId}/photos/${photoId}/enhancements/${runId}/retry`
    );
    return response.data;
  },

  triggerSightingEnhancement: async (caseId, sightingId, photoId) => {
    const response = await api.post(
      `/cases/${caseId}/sightings/${sightingId}/photos/${photoId}/enhancements`
    );
    return response.data;
  },

  listSightingEnhancements: async (caseId, sightingId, photoId) => {
    const response = await api.get(
      `/cases/${caseId}/sightings/${sightingId}/photos/${photoId}/enhancements`
    );
    return response.data;
  },

  getSightingEnhancement: async (caseId, sightingId, photoId, runId) => {
    const response = await api.get(
      `/cases/${caseId}/sightings/${sightingId}/photos/${photoId}/enhancements/${runId}`
    );
    return response.data;
  },

  retrySightingEnhancement: async (caseId, sightingId, photoId, runId) => {
    const response = await api.post(
      `/cases/${caseId}/sightings/${sightingId}/photos/${photoId}/enhancements/${runId}/retry`
    );
    return response.data;
  },
};

const similarityBody = (options = {}) => {
  // Server-side defaults apply when a field is omitted; only send
  // explicitly chosen values so backend limits always govern.
  const body = {};
  if (options.topK !== undefined && options.topK !== null) {
    body.top_k = options.topK;
  }
  if (options.threshold !== undefined && options.threshold !== null) {
    body.threshold = options.threshold;
  }
  return body;
};

export const similarityService = {
  searchCaseFaceSimilar: async (caseId, photoId, faceId, options = {}) => {
    const response = await api.post(
      `/cases/${caseId}/photos/${photoId}/faces/${faceId}/similar`,
      similarityBody(options)
    );
    return response.data;
  },

  searchSightingFaceSimilar: async (
    caseId,
    sightingId,
    photoId,
    faceId,
    options = {}
  ) => {
    const response = await api.post(
      `/cases/${caseId}/sightings/${sightingId}/photos/${photoId}/faces/${faceId}/similar`,
      similarityBody(options)
    );
    return response.data;
  },
};

export default api;

