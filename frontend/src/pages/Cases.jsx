import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { caseService, photoService, sightingService } from '../services/api';
import { PageHeader, SectionHeader, PhotoCard } from '../components/Workspace';
import { CaseStatusBadge } from '../components/StatusBadge';
import { Modal } from '../components/Modal';
import { PhotoAIModal } from '../components/PhotoAIModal';
import { LoadingState, EmptyState, ErrorState } from '../components/StateBlocks';
import './Cases.css';

export const Cases = () => {
  const { user } = useAuth();
  const navigate = useNavigate();
  
  // Case listing state
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Frontend filter and search state
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');

  // Modals state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [caseToDelete, setCaseToDelete] = useState(null);
  const [deleteSubmitting, setDeleteSubmitting] = useState(false);
  const [deleteError, setDeleteError] = useState(null);

  // Create case form state
  const [createTitle, setCreateTitle] = useState('');
  const [createDescription, setCreateDescription] = useState('');
  // Phase 0 structured fields (optional)
  const [createFullName, setCreateFullName] = useState('');
  const [createAgeYears, setCreateAgeYears] = useState('');
  const [createLastSeenLocation, setCreateLastSeenLocation] = useState('');
  const [createError, setCreateError] = useState(null);
  const [createSuccess, setCreateSuccess] = useState(null);
  const [createSubmitting, setCreateSubmitting] = useState(false);

  // Detail view state
  const [detailCase, setDetailCase] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState(null);

  // Case editing state
  const [isEditing, setIsEditing] = useState(false);
  const [editTitle, setEditTitle] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editStatus, setEditStatus] = useState('');
  const [editFullName, setEditFullName] = useState('');
  const [editAgeYears, setEditAgeYears] = useState('');
  const [editLastSeenLocation, setEditLastSeenLocation] = useState('');
  const [editSubmitting, setEditSubmitting] = useState(false);
  const [editError, setEditError] = useState(null);
  const [editSuccess, setEditSuccess] = useState(null);

  // Phase 1 evidence photo state
  const [photos, setPhotos] = useState([]);
  const [photosLoading, setPhotosLoading] = useState(false);
  const [photosError, setPhotosError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [deletingPhotoId, setDeletingPhotoId] = useState(null);
  const [retryingPhotoId, setRetryingPhotoId] = useState(null);
  const [detectingFaceId, setDetectingFaceId] = useState(null);
  const [aiPhotoId, setAiPhotoId] = useState(null);

  // Phase 2 sightings state (report list inside the case detail view)
  const [caseSightings, setCaseSightings] = useState([]);
  const [sightingsLoading, setSightingsLoading] = useState(false);
  const [sightingsError, setSightingsError] = useState(null);

  const fetchCases = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await caseService.getCases();
      setCases(data);
    } catch (err) {
      console.error('Error fetching cases:', err);
      if (err.response && err.response.data && err.response.data.detail) {
        setError(err.response.data.detail);
      } else {
        setError('Failed to fetch cases. Please verify your connection to the TraceLink server.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCases();
  }, []);

  const fetchPhotos = async (caseId) => {
    setPhotosLoading(true);
    setPhotosError(null);
    try {
      const data = await photoService.listPhotos(caseId);
      setPhotos(data);
    } catch (err) {
      console.error(`Error fetching photos for case ${caseId}:`, err);
      if (err.response && err.response.status === 403) {
        setPhotosError('You do not have permission to view evidence photos.');
      } else {
        setPhotosError('Failed to load evidence photos.');
      }
      setPhotos([]);
    } finally {
      setPhotosLoading(false);
    }
  };

  const handlePhotoUpload = async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file || !detailCase) return;
    setUploadError(null);

    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setUploadError('Only JPEG, PNG, and WebP images are accepted.');
      e.target.value = null;
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setUploadError('Image exceeds the maximum allowed size (10 MB).');
      e.target.value = null;
      return;
    }

    setUploading(true);
    try {
      await photoService.uploadPhoto(detailCase.id, file);
      await fetchPhotos(detailCase.id);
    } catch (err) {
      console.error('Error uploading photo:', err);
      if (err.response && err.response.status === 403) {
        setUploadError('You do not have permission to add photos to this case.');
      } else if (err.response && err.response.status === 413) {
        setUploadError('Image exceeds the maximum allowed size (10 MB).');
      } else if (err.response && (err.response.status === 415 || err.response.status === 400)) {
        setUploadError(err.response.data?.detail || 'Invalid image file.');
      } else {
        setUploadError('Failed to upload photo. Please try again.');
      }
    } finally {
      setUploading(false);
      e.target.value = null;
    }
  };

  const handlePhotoDelete = async (photoId) => {
    if (!detailCase) return;
    if (!window.confirm('Delete this evidence photo? This cannot be undone.')) return;
    setDeletingPhotoId(photoId);
    setUploadError(null);
    try {
      await photoService.deletePhoto(detailCase.id, photoId);
      await fetchPhotos(detailCase.id);
    } catch (err) {
      console.error('Error deleting photo:', err);
      setUploadError('Failed to delete photo. Please try again.');
    } finally {
      setDeletingPhotoId(null);
    }
  };

  const handlePhotoRetry = async (photoId) => {
    if (!detailCase) return;
    setRetryingPhotoId(photoId);
    setUploadError(null);
    try {
      await photoService.retryPhoto(detailCase.id, photoId);
      await fetchPhotos(detailCase.id);
    } catch (err) {
      console.error('Error retrying photo processing:', err);
      if (err.response && err.response.status === 409) {
        setUploadError('Photo is already being processed; try again later.');
      } else if (err.response && err.response.status === 403) {
        setUploadError('You do not have permission to process photos in this case.');
      } else {
        setUploadError('Failed to retry photo processing. Please try again.');
      }
    } finally {
      setRetryingPhotoId(null);
    }
  };

  const handleFaceDetect = async (photoId, redetect) => {
    if (!detailCase) return;
    setDetectingFaceId(photoId);
    setUploadError(null);
    try {
      if (redetect) {
        await photoService.redetectFaces(detailCase.id, photoId);
      } else {
        await photoService.detectFaces(detailCase.id, photoId);
      }
      await fetchPhotos(detailCase.id);
    } catch (err) {
      console.error('Error running face detection:', err);
      if (err.response && err.response.status === 403) {
        setUploadError('You do not have permission to run face detection.');
      } else if (err.response && err.response.status === 409) {
        setUploadError(err.response.data?.detail || 'Photo is not ready for face detection.');
      } else {
        setUploadError('Failed to run face detection. Please try again.');
      }
    } finally {
      setDetectingFaceId(null);
    }
  };

  const fetchCaseSightings = async (caseId) => {
    setSightingsLoading(true);
    setSightingsError(null);
    try {
      const data = await sightingService.listSightings(caseId);
      setCaseSightings(data);
    } catch (err) {
      console.error(`Error fetching sightings for case ${caseId}:`, err);
      if (err.response && err.response.status === 403) {
        setSightingsError('You do not have permission to view sightings.');
      } else {
        setSightingsError('Failed to load sightings.');
      }
      setCaseSightings([]);
    } finally {
      setSightingsLoading(false);
    }
  };

  const handleViewDetails = async (caseId) => {
    setShowDetailModal(true);
    setAiPhotoId(null);
    setIsEditing(false);
    setEditError(null);
    setEditSuccess(null);
    setDetailLoading(true);
    setDetailError(null);
    setDetailCase(null);
    setPhotos([]);
    setPhotosError(null);
    setUploadError(null);
    setCaseSightings([]);
    setSightingsError(null);
    try {
      const data = await caseService.getCase(caseId);
      setDetailCase(data);
      fetchPhotos(caseId);
      fetchCaseSightings(caseId);
    } catch (err) {
      console.error(`Error fetching case details for ID ${caseId}:`, err);
      if (err.response && err.response.status === 403) {
        setDetailError('You do not have permission to view this case.');
      } else if (err.response && err.response.status === 404) {
        setDetailError('Case not found.');
      } else if (err.response && err.response.data && err.response.data.detail) {
        setDetailError(err.response.data.detail);
      } else {
        setDetailError('Failed to load case details.');
      }
    } finally {
      setDetailLoading(false);
    }
  };

  const startEditing = () => {
    if (detailCase) {
      setEditTitle(detailCase.title);
      setEditDescription(detailCase.description);
      setEditStatus(detailCase.status);
      setEditFullName(detailCase.full_name || '');
      setEditAgeYears(detailCase.age_years ?? '');
      setEditLastSeenLocation(detailCase.last_seen_location || '');
      setEditError(null);
      setEditSuccess(null);
      setIsEditing(true);
    }
  };

  const cancelEditing = () => {
    setIsEditing(false);
    setEditError(null);
    setEditSuccess(null);
  };

  const handleUpdateSubmit = async (e) => {
    if (e) e.preventDefault();
    setEditError(null);
    setEditSuccess(null);

    if (!editTitle.trim() || !editDescription.trim()) {
      setEditError('Please provide both a title and a description.');
      return;
    }

    // Phase 0: optional age must be an integer in range when provided.
    let parsedEditAge = null;
    if (String(editAgeYears).trim() !== '') {
      parsedEditAge = Number(editAgeYears);
      if (!Number.isInteger(parsedEditAge) || parsedEditAge < 0 || parsedEditAge > 150) {
        setEditError('Age must be a whole number between 0 and 150.');
        return;
      }
    }

    setEditSubmitting(true);
    try {
      const updatedCase = await caseService.updateCase(detailCase.id, {
        title: editTitle.trim(),
        description: editDescription.trim(),
        status: editStatus,
        full_name: editFullName.trim() ? editFullName.trim() : null,
        age_years: parsedEditAge,
        last_seen_location: editLastSeenLocation.trim() ? editLastSeenLocation.trim() : null,
      });
      setEditSuccess('Case updated successfully!');
      
      // Refresh list
      await fetchCases();

      // Refresh detail view
      setDetailCase(updatedCase);

      // Exit edit mode after a brief moment to show success
      setTimeout(() => {
        setIsEditing(false);
        setEditSuccess(null);
      }, 1000);

    } catch (err) {
      console.error('Error updating case:', err);
      if (err.response) {
        const { status, data } = err.response;
        if (status === 403) {
          setEditError('You do not have permission to update this case.');
        } else if (status === 404) {
          setEditError('Case not found.');
        } else if (data && data.detail) {
          const detail = data.detail;
          if (typeof detail === 'string') {
            setEditError(detail);
          } else if (Array.isArray(detail)) {
            setEditError(detail.map(d => d.msg).join(', '));
          } else {
            setEditError('Validation failed.');
          }
        } else {
          setEditError('Failed to save changes. Please try again.');
        }
      } else {
        setEditError('Connection failed. Please verify your connection.');
      }
    } finally {
      setEditSubmitting(false);
    }
  };

  const handleCreateSubmit = async (e) => {
    e.preventDefault();
    setCreateError(null);
    setCreateSuccess(null);

    if (!createTitle.trim() || !createDescription.trim()) {
      setCreateError('Please provide both a title and a description.');
      return;
    }

    // Phase 0: optional age must be an integer in range when provided.
    let parsedCreateAge;
    if (String(createAgeYears).trim() !== '') {
      parsedCreateAge = Number(createAgeYears);
      if (!Number.isInteger(parsedCreateAge) || parsedCreateAge < 0 || parsedCreateAge > 150) {
        setCreateError('Age must be a whole number between 0 and 150.');
        return;
      }
    }

    setCreateSubmitting(true);
    try {
      const createPayload = {
        title: createTitle,
        description: createDescription,
      };
      if (createFullName.trim()) createPayload.full_name = createFullName.trim();
      if (parsedCreateAge !== undefined) createPayload.age_years = parsedCreateAge;
      if (createLastSeenLocation.trim()) createPayload.last_seen_location = createLastSeenLocation.trim();
      const newCase = await caseService.createCase(createPayload);
      setCreateSuccess('Case registered successfully!');
      setCreateTitle('');
      setCreateDescription('');
      setCreateFullName('');
      setCreateAgeYears('');
      setCreateLastSeenLocation('');
      
      // Refresh list
      await fetchCases();

      // Automatically transition after success
      setTimeout(() => {
        setShowCreateModal(false);
        setCreateSuccess(null);
        if (newCase && newCase.id) {
          handleViewDetails(newCase.id);
        }
      }, 1200);
    } catch (err) {
      console.error('Error creating case:', err);
      if (err.response && err.response.data && err.response.data.detail) {
        const detail = err.response.data.detail;
        if (typeof detail === 'string') {
          setCreateError(detail);
        } else if (Array.isArray(detail)) {
          setCreateError(detail.map(d => d.msg).join(', '));
        } else {
          setCreateError('Validation failed.');
        }
      } else if (err.response && err.response.status === 403) {
        setCreateError('Permission denied. You do not have authorization to create cases.');
      } else {
        setCreateError('Failed to register the case. Please verify connection and try again.');
      }
    } finally {
      setCreateSubmitting(false);
    }
  };

  const promptDeleteCase = (caseItem) => {
    if (!caseItem) return;
    setCaseToDelete(caseItem);
    setDeleteError(null);
    setShowDeleteModal(true);
  };

  const cancelDeleteCase = () => {
    if (deleteSubmitting) return;
    setShowDeleteModal(false);
    setCaseToDelete(null);
    setDeleteError(null);
  };

  const handleConfirmDelete = async () => {
    if (!caseToDelete) return;
    setDeleteSubmitting(true);
    setDeleteError(null);
    try {
      await caseService.deleteCase(caseToDelete.id);
      setShowDeleteModal(false);
      const deletedId = caseToDelete.id;
      setCaseToDelete(null);
      if (detailCase && detailCase.id === deletedId) {
        setShowDetailModal(false);
        setDetailCase(null);
      }
      await fetchCases();
    } catch (err) {
      console.error('Error deleting case:', err);
      if (err.response && err.response.status === 403) {
        setDeleteError('You do not have permission to delete this case.');
      } else if (err.response && err.response.status === 404) {
        setDeleteError('Case not found or already deleted.');
      } else if (err.response && err.response.data && err.response.data.detail) {
        const detail = err.response.data.detail;
        setDeleteError(typeof detail === 'string' ? detail : 'Failed to delete case.');
      } else {
        setDeleteError('Failed to delete case. Please try again.');
      }
    } finally {
      setDeleteSubmitting(false);
    }
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return 'N/A';
    const date = new Date(dateStr);
    return date.toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };



  const canCreateCase = user && ['REPORTER', 'ORGANIZATION_MEMBER', 'ADMIN'].includes(user.role);

  // Client-side filtering of retrieved cases
  const filteredCases = cases.filter((item) => {
    const matchesSearch =
      item.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesStatus =
      statusFilter === 'ALL' || item.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const canModifyDetail = user && detailCase && (user.role === 'ADMIN' || detailCase.created_by === user.id);

  // Narrow Phase 4 permission: ONLY admin and reviewer may trigger
  // face detection. Case-edit permission (including case ownership)
  // does not imply detection permission.
  const canDetectFaces = user && detailCase && (
    user.role === 'ADMIN' || user.role === 'REVIEWER'
  );

  // Locked Phase 7 permission: ONLY admin and reviewer may trigger
  // enhancement. Other roles can still view runs and results.
  const canEnhance = user && detailCase && (
    user.role === 'ADMIN' || user.role === 'REVIEWER'
  );

  const aiPhoto = aiPhotoId
    ? photos.find((item) => item.id === aiPhotoId) || null
    : null;

  return (
    <div className="cases-page-container">
      <PageHeader
        title="Cases"
        subtitle="Missing-person case files. Select a case to review evidence, sightings, and face-detection results."
        actions={
          canCreateCase ? (
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => {
                setCreateError(null);
                setCreateSuccess(null);
                setShowCreateModal(true);
              }}
            >
              Create case
            </button>
          ) : null
        }
      />
      {/* Top Controls Action Bar */}
      <div className="cases-actions-bar">
        <div className="search-filter-group">
          <input
            type="text"
            className="form-control"
            placeholder="Search by title or description..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            aria-label="Search cases"
            style={{ flex: 2 }}
          />
          <select
            className="form-control"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            aria-label="Filter cases by status"
            style={{ flex: 1, minWidth: '130px' }}
          >
            <option value="ALL">All Statuses</option>
            <option value="OPEN">Open</option>
            <option value="UNDER_REVIEW">Under Review</option>
            <option value="RESOLVED">Resolved</option>
            <option value="CLOSED">Closed</option>
          </select>
        </div>
        <span className="text-muted" style={{ fontSize: '0.8rem' }} aria-live="polite">
          {loading ? '…' : `${filteredCases.length} of ${cases.length} cases`}
        </span>
      </div>

      {/* Main Cases Content */}
      {loading ? (
        <LoadingState label="Loading cases…" />
      ) : error ? (
        <ErrorState detail={error} onRetry={fetchCases} />
      ) : filteredCases.length === 0 ? (
        <EmptyState
          title="No cases found"
          detail={
            searchTerm || statusFilter !== 'ALL'
              ? 'No cases match the current search or filters.'
              : 'There are no missing-person cases currently recorded.'
          }
          action={
            !searchTerm && statusFilter === 'ALL' && canCreateCase ? (
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => setShowCreateModal(true)}
              >
                Register first case
              </button>
            ) : null
          }
        />
      ) : (
        <div className="cases-table-container">
          <table className="cases-table">
            <thead>
              <tr>
                <th style={{ width: '80px' }}>ID</th>
                <th>Title</th>
                <th style={{ width: '150px' }}>Status</th>
                <th style={{ width: '180px' }}>Created At</th>
                <th style={{ width: '180px' }}>Updated At</th>
                <th style={{ width: '100px', textAlign: 'center' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredCases.map((caseItem) => (
                <tr key={caseItem.id}>
                  <td style={{ fontWeight: 'bold', fontFamily: 'monospace' }}>
                    #{caseItem.id}
                  </td>
                  <td>
                    <div style={{ fontWeight: '500' }}>{caseItem.title}</div>
                    <div 
                      style={{ 
                        fontSize: '0.8rem', 
                        color: 'var(--text-secondary)',
                        textOverflow: 'ellipsis',
                        overflow: 'hidden',
                        whiteSpace: 'nowrap',
                        maxWidth: '300px'
                      }}
                    >
                      {caseItem.description}
                    </div>
                  </td>
                  <td><CaseStatusBadge status={caseItem.status} /></td>
                  <td>{formatDate(caseItem.created_at)}</td>
                  <td>{formatDate(caseItem.updated_at)}</td>
                  <td style={{ textAlign: 'center' }}>
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => handleViewDetails(caseItem.id)}
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* CREATE CASE MODAL */}
      {showCreateModal && (
        <Modal
          title="Register new case"
          onClose={() => !createSubmitting && setShowCreateModal(false)}
        >
            <form onSubmit={handleCreateSubmit}>
              <div>
                {createError && (
                  <div className="alert alert-error">
                    {createError}
                  </div>
                )}
                {createSuccess && (
                  <div className="alert alert-success">
                    {createSuccess}
                  </div>
                )}

                <div className="form-group">
                  <label className="form-label" htmlFor="case-title">Case Title</label>
                  <input
                    id="case-title"
                    type="text"
                    className="form-control"
                    placeholder="e.g. Missing Person - Jane Doe (Ref #9822)"
                    value={createTitle}
                    onChange={(e) => setCreateTitle(e.target.value)}
                    disabled={createSubmitting || !!createSuccess}
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="case-desc">Description & Characteristics</label>
                  <textarea
                    id="case-desc"
                    className="form-control"
                    placeholder="Provide details including last seen location, clothes description, and physical identifiers."
                    value={createDescription}
                    onChange={(e) => setCreateDescription(e.target.value)}
                    disabled={createSubmitting || !!createSuccess}
                    rows="6"
                    style={{ resize: 'vertical' }}
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="case-fullname">Missing Person&apos;s Full Name (optional)</label>
                  <input
                    id="case-fullname"
                    type="text"
                    className="form-control"
                    placeholder="e.g. Jane Doe"
                    value={createFullName}
                    onChange={(e) => setCreateFullName(e.target.value)}
                    disabled={createSubmitting || !!createSuccess}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="case-age">Age in Years (optional)</label>
                  <input
                    id="case-age"
                    type="number"
                    className="form-control"
                    placeholder="e.g. 29"
                    min="0"
                    max="150"
                    value={createAgeYears}
                    onChange={(e) => setCreateAgeYears(e.target.value)}
                    disabled={createSubmitting || !!createSuccess}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="case-loc">Last Seen Location (optional)</label>
                  <input
                    id="case-loc"
                    type="text"
                    className="form-control"
                    placeholder="e.g. Central Station, Platform 2"
                    value={createLastSeenLocation}
                    onChange={(e) => setCreateLastSeenLocation(e.target.value)}
                    disabled={createSubmitting || !!createSuccess}
                  />
                </div>
              </div>
              <div className="modal-footer">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => setShowCreateModal(false)}
                  disabled={createSubmitting}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={createSubmitting || !!createSuccess}
                >
                  {createSubmitting ? 'Registering…' : 'Register case'}
                </button>
              </div>
            </form>
        </Modal>
      )}

      {/* VIEW DETAILS MODAL */}
      {showDetailModal && (
        <Modal
          wide
          title={isEditing ? 'Edit case' : `Case #${detailCase?.id || ''}`}
          onClose={() => !editSubmitting && setShowDetailModal(false)}
        >
              {detailLoading ? (
                <LoadingState label="Loading case detail…" />
              ) : detailError ? (
                <ErrorState detail={detailError} />
              ) : detailCase ? (
                isEditing ? (
                  <form onSubmit={handleUpdateSubmit}>
                    {editError && (
                      <div className="alert alert-error">
                        {editError}
                      </div>
                    )}
                    {editSuccess && (
                      <div className="alert alert-success">
                        {editSuccess}
                      </div>
                    )}

                    <div className="form-group">
                      <label className="form-label" htmlFor="edit-title">Case Title</label>
                      <input
                        id="edit-title"
                        type="text"
                        className="form-control"
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        disabled={editSubmitting || !!editSuccess}
                        required
                      />
                    </div>

                    <div className="form-group">
                      <label className="form-label" htmlFor="edit-status">Status State</label>
                      <select
                        id="edit-status"
                        className="form-control"
                        value={editStatus}
                        onChange={(e) => setEditStatus(e.target.value)}
                        disabled={editSubmitting || !!editSuccess}
                      >
                        <option value="OPEN">Open</option>
                        <option value="UNDER_REVIEW">Under Review</option>
                        <option value="RESOLVED">Resolved</option>
                        <option value="CLOSED">Closed</option>
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label" htmlFor="edit-desc">Description & Characteristics</label>
                      <textarea
                        id="edit-desc"
                        className="form-control"
                        value={editDescription}
                        onChange={(e) => setEditDescription(e.target.value)}
                        disabled={editSubmitting || !!editSuccess}
                        rows="6"
                        style={{ resize: 'vertical' }}
                        required
                      />
                    </div>

                    <div className="form-group">
                      <label className="form-label" htmlFor="edit-fullname">Missing Person&apos;s Full Name (optional)</label>
                      <input
                        id="edit-fullname"
                        type="text"
                        className="form-control"
                        value={editFullName}
                        onChange={(e) => setEditFullName(e.target.value)}
                        disabled={editSubmitting || !!editSuccess}
                      />
                    </div>

                    <div className="form-group">
                      <label className="form-label" htmlFor="edit-age">Age in Years (optional)</label>
                      <input
                        id="edit-age"
                        type="number"
                        className="form-control"
                        min="0"
                        max="150"
                        value={editAgeYears}
                        onChange={(e) => setEditAgeYears(e.target.value)}
                        disabled={editSubmitting || !!editSuccess}
                      />
                    </div>

                    <div className="form-group">
                      <label className="form-label" htmlFor="edit-loc">Last Seen Location (optional)</label>
                      <input
                        id="edit-loc"
                        type="text"
                        className="form-control"
                        value={editLastSeenLocation}
                        onChange={(e) => setEditLastSeenLocation(e.target.value)}
                        disabled={editSubmitting || !!editSuccess}
                      />
                    </div>
                  </form>
                ) : (
                  <div className="detail-grid">
                    <div className="detail-row">
                      <span className="detail-label">Case ID</span>
                      <span className="detail-value" style={{ fontWeight: 'bold', fontFamily: 'monospace' }}>
                        #{detailCase.id}
                      </span>
                    </div>

                    <div className="detail-row">
                      <span className="detail-label">Case Title</span>
                      <span className="detail-value" style={{ fontWeight: '600' }}>
                        {detailCase.title}
                      </span>
                    </div>

                    <div className="detail-row">
                      <span className="detail-label">Status State</span>
                      <span className="detail-value">
                        <CaseStatusBadge status={detailCase.status} />
                      </span>
                    </div>

                    {detailCase.full_name && (
                      <div className="detail-row">
                        <span className="detail-label">Missing Person</span>
                        <span className="detail-value" style={{ fontWeight: '600' }}>
                          {detailCase.full_name}
                        </span>
                      </div>
                    )}

                    {(detailCase.age_years ?? null) !== null && (
                      <div className="detail-row">
                        <span className="detail-label">Age</span>
                        <span className="detail-value">
                          {detailCase.age_years} years
                        </span>
                      </div>
                    )}

                    {detailCase.last_seen_location && (
                      <div className="detail-row">
                        <span className="detail-label">Last Seen Location</span>
                        <span className="detail-value">
                          {detailCase.last_seen_location}
                        </span>
                      </div>
                    )}

                    <div className="detail-row">
                      <span className="detail-label">Investigator ID</span>
                      <span className="detail-value" style={{ fontFamily: 'monospace' }}>
                        User ID: {detailCase.created_by}
                      </span>
                    </div>

                    <div className="detail-row">
                      <span className="detail-label">Created Date</span>
                      <span className="detail-value">
                        {formatDate(detailCase.created_at)}
                      </span>
                    </div>

                    <div className="detail-row">
                      <span className="detail-label">Last Modified</span>
                      <span className="detail-value">
                        {formatDate(detailCase.updated_at)}
                      </span>
                    </div>

                    <div style={{ marginTop: '1rem' }}>
                      <div className="detail-label" style={{ marginBottom: '0.5rem' }}>Case Details & Notes:</div>
                      <div className="detail-value-desc">
                        {detailCase.description}
                      </div>
                    </div>

                    <div className="evidence-section">
                      <SectionHeader
                        title="Evidence"
                        meta={photosLoading ? '…' : `${photos.length} photo${photos.length === 1 ? '' : 's'}`}
                      />
                      {uploadError && (
                        <div className="alert alert-error" role="alert">
                          {uploadError}
                        </div>
                      )}
                      {photosLoading ? (
                        <LoadingState label="Loading evidence photos…" />
                      ) : photosError ? (
                        <ErrorState detail={photosError} onRetry={() => fetchPhotos(detailCase.id)} />
                      ) : photos.length === 0 ? (
                        <EmptyState
                          title="No evidence photos"
                          detail={canModifyDetail ? 'Upload the first evidence photo below.' : null}
                        />
                      ) : (
                        <div className="photo-grid">
                          {photos.map((photo) => (
                            <PhotoCard
                              key={photo.id}
                              photo={photo}
                              alt={`Case evidence photo ${photo.id}`}
                              canDelete={canModifyDetail}
                              onDelete={handlePhotoDelete}
                              deleting={deletingPhotoId === photo.id}
                              busy={uploading}
                              canDetect={canDetectFaces}
                              detecting={detectingFaceId === photo.id}
                              onDetect={(id) => handleFaceDetect(id, false)}
                              onRedetect={(id) => handleFaceDetect(id, true)}
                              onRetryProcessing={handlePhotoRetry}
                              retrying={retryingPhotoId === photo.id}
                              showRetry={canModifyDetail && (photo.processing_status === 'FAILED' || photo.processing_status === 'UPLOADED')}
                              onOpenAI={(id) => setAiPhotoId(id)}
                            />
                          ))}
                        </div>
                      )}
                      {canModifyDetail && (
                        <div>
                          <label className="btn btn-secondary" style={{ cursor: uploading ? 'wait' : 'pointer' }}>
                            {uploading ? 'Uploading…' : 'Upload photo (JPEG/PNG/WebP, ≤10 MB)'}
                            <input
                              type="file"
                              accept="image/jpeg,image/png,image/webp"
                              onChange={handlePhotoUpload}
                              disabled={uploading}
                              style={{ display: 'none' }}
                            />
                          </label>
                        </div>
                      )}
                    </div>

                    <div className="evidence-section">
                      <SectionHeader
                        title="Sightings"
                        meta={sightingsLoading ? '…' : `${caseSightings.length} linked`}
                      />
                      {sightingsLoading ? (
                        <p className="cases-state-text">Loading sightings...</p>
                      ) : sightingsError ? (
                        <p className="cases-state-text">{sightingsError}</p>
                      ) : caseSightings.length === 0 ? (
                        <p className="cases-state-text">No sightings reported for this case yet.</p>
                      ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                          {caseSightings.slice(0, 5).map((sighting) => (
                            <div
                              key={sighting.id}
                              style={{ border: '1px solid var(--border-color)', borderRadius: 'var(--radius-sm)', padding: '0.5rem 0.75rem', fontSize: '0.85rem' }}
                            >
                              <div style={{ fontWeight: '600' }}>
                                #{sighting.id} — {sighting.location_text}
                              </div>
                              <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
                                {formatDate(sighting.sighting_at)} · {sighting.status}
                              </div>
                            </div>
                          ))}
                          {caseSightings.length > 5 && (
                            <p className="cases-state-text">
                              …and {caseSightings.length - 5} more.
                            </p>
                          )}
                        </div>
                      )}
                      <div>
                        <button
                          type="button"
                          className="btn btn-secondary"
                          onClick={() => {
                            setShowDetailModal(false);
                            navigate('/sightings');
                          }}
                        >
                          Open sighting reports
                        </button>
                      </div>
                    </div>
                  </div>
                )
              ) : null}
            <div className="modal-footer">
              {detailCase && !detailLoading && !detailError && (
                user && (user.role === 'ADMIN' || detailCase.created_by === user.id)
              ) && (
                isEditing ? (
                  <>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={cancelEditing}
                      disabled={editSubmitting}
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      className="btn btn-primary"
                      onClick={handleUpdateSubmit}
                      disabled={editSubmitting || !!editSuccess}
                    >
                      {editSubmitting ? 'Saving…' : 'Save changes'}
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      type="button"
                      className="btn btn-danger"
                      onClick={() => promptDeleteCase(detailCase)}
                    >
                      Delete case
                    </button>
                    <button
                      type="button"
                      className="btn btn-primary"
                      onClick={startEditing}
                    >
                      Edit case
                    </button>
                  </>
                )
              )}
              {!isEditing && (
                <button type="button" className="btn btn-secondary" onClick={() => { setShowDetailModal(false); setAiPhotoId(null); }}>
                  Close
                </button>
              )}
            </div>
        </Modal>
      )}

      {/* AI WORKFLOW MODAL */}
      {showDetailModal && aiPhoto && detailCase && (
        <PhotoAIModal
          photo={aiPhoto}
          photoKind="case"
          caseId={detailCase.id}
          sightingId={null}
          canEnhance={canEnhance}
          onClose={() => setAiPhotoId(null)}
          onPhotosChanged={() => fetchPhotos(detailCase.id)}
        />
      )}

      {/* DELETE CONFIRMATION MODAL */}
      {showDeleteModal && caseToDelete && (
        <Modal title="Confirm case deletion" onClose={cancelDeleteCase}>
              {deleteError && (
                <div className="alert alert-error" role="alert">
                  {deleteError}
                </div>
              )}
              <p style={{ fontSize: '0.95rem', color: 'var(--text-primary)', marginBottom: '0.75rem' }}>
                Permanently delete <strong>Case #{caseToDelete.id}</strong> (&ldquo;{caseToDelete.title}&rdquo;)?
              </p>
              <p style={{ fontSize: '0.85rem', color: 'var(--error)' }}>
                This cannot be undone. All records for this case, including evidence and sightings, will be removed.
              </p>
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={cancelDeleteCase}
                disabled={deleteSubmitting}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-danger"
                onClick={handleConfirmDelete}
                disabled={deleteSubmitting}
              >
                {deleteSubmitting ? 'Deleting…' : 'Delete case'}
              </button>
            </div>
        </Modal>
      )}
    </div>
  );
};

export default Cases;
