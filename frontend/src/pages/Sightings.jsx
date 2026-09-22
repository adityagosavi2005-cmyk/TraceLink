import React, { useState, useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';
import { caseService, sightingService } from '../services/api';
import { PageHeader, SectionHeader, PhotoCard } from '../components/Workspace';
import { SightingStatusBadge } from '../components/StatusBadge';
import { Modal } from '../components/Modal';
import { PhotoAIModal } from '../components/PhotoAIModal';
import { LoadingState, EmptyState, ErrorState } from '../components/StateBlocks';
import './Cases.css';

const SIGHTING_STATUSES = ['REPORTED', 'UNDER_REVIEW', 'VERIFIED', 'DISMISSED'];

export const Sightings = () => {
  const { user } = useAuth();

  // Listing state
  const [sightings, setSightings] = useState([]);
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Filter state
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState('ALL');

  // Modals state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [sightingToDelete, setSightingToDelete] = useState(null);
  const [deleteSubmitting, setDeleteSubmitting] = useState(false);
  const [deleteError, setDeleteError] = useState(null);

  // Create form state
  const [createCaseId, setCreateCaseId] = useState('');
  const [createSightedAt, setCreateSightedAt] = useState('');
  const [createLocation, setCreateLocation] = useState('');
  const [createDescription, setCreateDescription] = useState('');
  const [createLatitude, setCreateLatitude] = useState('');
  const [createLongitude, setCreateLongitude] = useState('');
  const [createContact, setCreateContact] = useState('');
  const [createError, setCreateError] = useState(null);
  const [createSuccess, setCreateSuccess] = useState(null);
  const [createSubmitting, setCreateSubmitting] = useState(false);

  // Detail view state
  const [detailSighting, setDetailSighting] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState(null);

  // Editing state
  const [isEditing, setIsEditing] = useState(false);
  const [editLocation, setEditLocation] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editStatus, setEditStatus] = useState('');
  const [editContact, setEditContact] = useState('');
  const [editSubmitting, setEditSubmitting] = useState(false);
  const [editError, setEditError] = useState(null);
  const [editSuccess, setEditSuccess] = useState(null);

  // Sighting photo state
  const [photos, setPhotos] = useState([]);
  const [photosLoading, setPhotosLoading] = useState(false);
  const [photosError, setPhotosError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const [deletingPhotoId, setDeletingPhotoId] = useState(null);
  const [retryingPhotoId, setRetryingPhotoId] = useState(null);
  const [detectingFaceId, setDetectingFaceId] = useState(null);
  const [aiPhotoId, setAiPhotoId] = useState(null);

  const fetchSightings = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sightingData, caseData] = await Promise.all([
        sightingService.listAllSightings(),
        caseService.getCases(),
      ]);
      setSightings(sightingData);
      setCases(caseData);
    } catch (err) {
      console.error('Error fetching sightings:', err);
      if (err.response && err.response.data && err.response.data.detail) {
        setError(err.response.data.detail);
      } else {
        setError('Failed to fetch sightings. Please verify your connection to the TraceLink server.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSightings();
  }, []);

  const caseTitleFor = (caseId) => {
    const match = cases.find((c) => c.id === caseId);
    return match ? `#${match.id} — ${match.title}` : `Case #${caseId}`;
  };

  const fetchPhotos = async (caseId, sightingId) => {
    setPhotosLoading(true);
    setPhotosError(null);
    try {
      const data = await sightingService.listSightingPhotos(caseId, sightingId);
      setPhotos(data);
    } catch (err) {
      console.error(`Error fetching photos for sighting ${sightingId}:`, err);
      if (err.response && err.response.status === 403) {
        setPhotosError('You do not have permission to view sighting photos.');
      } else {
        setPhotosError('Failed to load sighting photos.');
      }
      setPhotos([]);
    } finally {
      setPhotosLoading(false);
    }
  };

  const handlePhotoUpload = async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file || !detailSighting) return;
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
      await sightingService.uploadSightingPhoto(detailSighting.case_id, detailSighting.id, file);
      await fetchPhotos(detailSighting.case_id, detailSighting.id);
    } catch (err) {
      console.error('Error uploading sighting photo:', err);
      if (err.response && err.response.status === 403) {
        setUploadError('You do not have permission to add photos to this sighting.');
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
    if (!detailSighting) return;
    if (!window.confirm('Delete this sighting photo? This cannot be undone.')) return;
    setDeletingPhotoId(photoId);
    setUploadError(null);
    try {
      await sightingService.deleteSightingPhoto(detailSighting.case_id, detailSighting.id, photoId);
      await fetchPhotos(detailSighting.case_id, detailSighting.id);
    } catch (err) {
      console.error('Error deleting sighting photo:', err);
      setUploadError('Failed to delete photo. Please try again.');
    } finally {
      setDeletingPhotoId(null);
    }
  };

  const handlePhotoRetry = async (photoId) => {
    if (!detailSighting) return;
    setRetryingPhotoId(photoId);
    setUploadError(null);
    try {
      await sightingService.retrySightingPhoto(detailSighting.case_id, detailSighting.id, photoId);
      await fetchPhotos(detailSighting.case_id, detailSighting.id);
    } catch (err) {
      console.error('Error retrying sighting photo processing:', err);
      if (err.response && err.response.status === 409) {
        setUploadError('Photo is already being processed; try again later.');
      } else if (err.response && err.response.status === 403) {
        setUploadError('You do not have permission to process photos in this sighting.');
      } else {
        setUploadError('Failed to retry photo processing. Please try again.');
      }
    } finally {
      setRetryingPhotoId(null);
    }
  };

  const handleFaceDetect = async (photoId, redetect) => {
    if (!detailSighting) return;
    setDetectingFaceId(photoId);
    setUploadError(null);
    try {
      if (redetect) {
        await sightingService.redetectSightingFaces(detailSighting.case_id, detailSighting.id, photoId);
      } else {
        await sightingService.detectSightingFaces(detailSighting.case_id, detailSighting.id, photoId);
      }
      await fetchPhotos(detailSighting.case_id, detailSighting.id);
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

  const handleViewDetails = async (sighting) => {
    setShowDetailModal(true);
    setAiPhotoId(null);
    setIsEditing(false);
    setEditError(null);
    setEditSuccess(null);
    setDetailLoading(true);
    setDetailError(null);
    setDetailSighting(null);
    setPhotos([]);
    setPhotosError(null);
    setUploadError(null);
    try {
      const data = await sightingService.getSighting(sighting.case_id, sighting.id);
      setDetailSighting(data);
      fetchPhotos(data.case_id, data.id);
    } catch (err) {
      console.error(`Error fetching sighting ${sighting.id}:`, err);
      if (err.response && err.response.status === 403) {
        setDetailError('You do not have permission to view this sighting.');
      } else if (err.response && err.response.status === 404) {
        setDetailError('Sighting not found.');
      } else if (err.response && err.response.data && err.response.data.detail) {
        setDetailError(err.response.data.detail);
      } else {
        setDetailError('Failed to load sighting details.');
      }
    } finally {
      setDetailLoading(false);
    }
  };

  const startEditing = () => {
    if (detailSighting) {
      setEditLocation(detailSighting.location_text || '');
      setEditDescription(detailSighting.description || '');
      setEditStatus(detailSighting.status || 'REPORTED');
      setEditContact(detailSighting.contact_info || '');
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

    if (!editLocation.trim() || !editDescription.trim()) {
      setEditError('Please provide both a location and a description.');
      return;
    }

    setEditSubmitting(true);
    try {
      const updated = await sightingService.updateSighting(detailSighting.case_id, detailSighting.id, {
        location_text: editLocation.trim(),
        description: editDescription.trim(),
        status: editStatus,
        contact_info: editContact.trim() ? editContact.trim() : null,
      });
      setEditSuccess('Sighting updated successfully!');
      await fetchSightings();
      setDetailSighting(updated);
      setTimeout(() => {
        setIsEditing(false);
        setEditSuccess(null);
      }, 1000);
    } catch (err) {
      console.error('Error updating sighting:', err);
      if (err.response && err.response.status === 403) {
        setEditError('You do not have permission to update this sighting.');
      } else if (err.response && err.response.data && err.response.data.detail) {
        const detail = err.response.data.detail;
        setEditError(typeof detail === 'string' ? detail : 'Validation failed.');
      } else {
        setEditError('Failed to save changes. Please try again.');
      }
    } finally {
      setEditSubmitting(false);
    }
  };

  const handleCreateSubmit = async (e) => {
    e.preventDefault();
    setCreateError(null);
    setCreateSuccess(null);

    if (!createCaseId) {
      setCreateError('Please select the related case.');
      return;
    }
    if (!createSightedAt) {
      setCreateError('Please provide the date and time of the sighting.');
      return;
    }
    if (!createLocation.trim() || !createDescription.trim()) {
      setCreateError('Please provide both a location and a description.');
      return;
    }

    // Coordinates are an optional pair.
    let parsedLat = null;
    let parsedLng = null;
    if (String(createLatitude).trim() !== '' || String(createLongitude).trim() !== '') {
      if (String(createLatitude).trim() === '' || String(createLongitude).trim() === '') {
        setCreateError('Latitude and longitude must be provided together.');
        return;
      }
      parsedLat = Number(createLatitude);
      parsedLng = Number(createLongitude);
      if (!Number.isFinite(parsedLat) || parsedLat < -90 || parsedLat > 90) {
        setCreateError('Latitude must be a number between -90 and 90.');
        return;
      }
      if (!Number.isFinite(parsedLng) || parsedLng < -180 || parsedLng > 180) {
        setCreateError('Longitude must be a number between -180 and 180.');
        return;
      }
    }

    const sightedAt = new Date(createSightedAt);
    if (Number.isNaN(sightedAt.getTime())) {
      setCreateError('The sighting date and time are invalid.');
      return;
    }
    if (sightedAt.getTime() > Date.now() + 5 * 60 * 1000) {
      setCreateError('The sighting date and time cannot be in the future.');
      return;
    }

    setCreateSubmitting(true);
    try {
      const payload = {
        sighting_at: sightedAt.toISOString(),
        location_text: createLocation.trim(),
        description: createDescription.trim(),
      };
      if (parsedLat !== null) {
        payload.latitude = parsedLat;
        payload.longitude = parsedLng;
      }
      if (createContact.trim()) payload.contact_info = createContact.trim();
      const created = await sightingService.createSighting(Number(createCaseId), payload);
      setCreateSuccess('Sighting reported successfully!');
      setCreateCaseId('');
      setCreateSightedAt('');
      setCreateLocation('');
      setCreateDescription('');
      setCreateLatitude('');
      setCreateLongitude('');
      setCreateContact('');
      await fetchSightings();
      setTimeout(() => {
        setShowCreateModal(false);
        setCreateSuccess(null);
        if (created && created.id) {
          handleViewDetails(created);
        }
      }, 1200);
    } catch (err) {
      console.error('Error creating sighting:', err);
      if (err.response && err.response.data && err.response.data.detail) {
        const detail = err.response.data.detail;
        if (typeof detail === 'string') {
          setCreateError(detail);
        } else if (Array.isArray(detail)) {
          setCreateError(detail.map((d) => d.msg).join(', '));
        } else {
          setCreateError('Validation failed.');
        }
      } else if (err.response && err.response.status === 403) {
        setCreateError('Permission denied. You cannot report sightings for this case.');
      } else {
        setCreateError('Failed to report the sighting. Please verify connection and try again.');
      }
    } finally {
      setCreateSubmitting(false);
    }
  };

  const promptDeleteSighting = (sighting) => {
    if (!sighting) return;
    setSightingToDelete(sighting);
    setDeleteError(null);
    setShowDeleteModal(true);
  };

  const cancelDeleteSighting = () => {
    if (deleteSubmitting) return;
    setShowDeleteModal(false);
    setSightingToDelete(null);
    setDeleteError(null);
  };

  const handleConfirmDelete = async () => {
    if (!sightingToDelete) return;
    setDeleteSubmitting(true);
    setDeleteError(null);
    try {
      await sightingService.deleteSighting(sightingToDelete.case_id, sightingToDelete.id);
      setShowDeleteModal(false);
      const deletedId = sightingToDelete.id;
      setSightingToDelete(null);
      if (detailSighting && detailSighting.id === deletedId) {
        setShowDetailModal(false);
        setDetailSighting(null);
      }
      await fetchSightings();
    } catch (err) {
      console.error('Error deleting sighting:', err);
      if (err.response && err.response.status === 403) {
        setDeleteError('You do not have permission to delete this sighting.');
      } else if (err.response && err.response.status === 404) {
        setDeleteError('Sighting not found or already deleted.');
      } else {
        setDeleteError('Failed to delete sighting. Please try again.');
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

  const canCreateSighting = user && ['REPORTER', 'ORGANIZATION_MEMBER', 'ADMIN'].includes(user.role);
  const canModifyDetail = user && detailSighting && (
    user.role === 'ADMIN' || detailSighting.reported_by === user.id
  );

  // Narrow Phase 4 permission: ONLY admin and reviewer may trigger
  // face detection. Sighting-reporter and case-edit relationships
  // do not imply detection permission.
  const canDetectFaces = user && detailSighting && (
    user.role === 'ADMIN' || user.role === 'REVIEWER'
  );

  // Locked Phase 7 permission: ONLY admin and reviewer may trigger
  // enhancement. Other roles can still view runs and results.
  const canEnhance = user && detailSighting && (
    user.role === 'ADMIN' || user.role === 'REVIEWER'
  );

  const aiPhoto = aiPhotoId
    ? photos.find((item) => item.id === aiPhotoId) || null
    : null;

  const filteredSightings = sightings.filter((item) => {
    const haystack = `${item.location_text} ${item.description}`.toLowerCase();
    const matchesSearch = haystack.includes(searchTerm.toLowerCase());
    const matchesStatus = statusFilter === 'ALL' || item.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  return (
    <div className="cases-page-container">
      <PageHeader
        title="Sightings"
        subtitle="Reported observations linked to missing-person cases. Select a report to review details and evidence."
        actions={
          canCreateSighting ? (
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => {
                setCreateError(null);
                setCreateSuccess(null);
                setShowCreateModal(true);
              }}
            >
              Report sighting
            </button>
          ) : null
        }
      />
      <div className="cases-actions-bar">
        <div className="search-filter-group">
          <input
            type="text"
            className="form-control"
            placeholder="Search by location or description..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            aria-label="Search sightings"
            style={{ flex: 2 }}
          />
          <select
            className="form-control"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            aria-label="Filter sightings by status"
            style={{ flex: 1, minWidth: '130px' }}
          >
            <option value="ALL">All Statuses</option>
            {SIGHTING_STATUSES.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
        <span className="text-muted" style={{ fontSize: '0.8rem' }} aria-live="polite">
          {loading ? '…' : `${filteredSightings.length} of ${sightings.length} reports`}
        </span>
      </div>

      {loading ? (
        <LoadingState label="Loading sightings…" />
      ) : error ? (
        <ErrorState detail={error} onRetry={fetchSightings} />
      ) : filteredSightings.length === 0 ? (
        <EmptyState
          title="No sightings found"
          detail={
            searchTerm || statusFilter !== 'ALL'
              ? 'No sighting reports match the current search or filters.'
              : 'There are no sighting reports currently recorded.'
          }
          action={
            !searchTerm && statusFilter === 'ALL' && canCreateSighting ? (
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => setShowCreateModal(true)}
              >
                Report first sighting
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
                <th>Case</th>
                <th>Location</th>
                <th style={{ width: '150px' }}>Status</th>
                <th style={{ width: '180px' }}>Sighted At</th>
                <th style={{ width: '100px', textAlign: 'center' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredSightings.map((sighting) => (
                <tr key={sighting.id}>
                  <td style={{ fontWeight: 'bold', fontFamily: 'monospace' }}>
                    #{sighting.id}
                  </td>
                  <td>
                    <div style={{ fontWeight: '500' }}>{caseTitleFor(sighting.case_id)}</div>
                  </td>
                  <td>
                    <div style={{ fontWeight: '500' }}>{sighting.location_text}</div>
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
                      {sighting.description}
                    </div>
                  </td>
                  <td><SightingStatusBadge status={sighting.status} /></td>
                  <td>{formatDate(sighting.sighting_at)}</td>
                  <td style={{ textAlign: 'center' }}>
                    <button
                      className="btn btn-primary btn-sm"
                      onClick={() => handleViewDetails(sighting)}
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

      {/* CREATE SIGHTING MODAL */}
      {showCreateModal && (
        <Modal
          title="Report new sighting"
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
                  <label className="form-label" htmlFor="sighting-case">Related Case</label>
                  <select
                    id="sighting-case"
                    className="form-control"
                    value={createCaseId}
                    onChange={(e) => setCreateCaseId(e.target.value)}
                    disabled={createSubmitting || !!createSuccess}
                    required
                  >
                    <option value="">Select a case...</option>
                    {cases.map((c) => (
                      <option key={c.id} value={c.id}>
                        #{c.id} — {c.title}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="sighting-at">Date & Time of Sighting</label>
                  <input
                    id="sighting-at"
                    type="datetime-local"
                    className="form-control"
                    value={createSightedAt}
                    onChange={(e) => setCreateSightedAt(e.target.value)}
                    disabled={createSubmitting || !!createSuccess}
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="sighting-loc">Location</label>
                  <input
                    id="sighting-loc"
                    type="text"
                    className="form-control"
                    placeholder="e.g. Central Station, Platform 2"
                    value={createLocation}
                    onChange={(e) => setCreateLocation(e.target.value)}
                    disabled={createSubmitting || !!createSuccess}
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="sighting-desc">Description</label>
                  <textarea
                    id="sighting-desc"
                    className="form-control"
                    placeholder="Describe what was observed."
                    value={createDescription}
                    onChange={(e) => setCreateDescription(e.target.value)}
                    disabled={createSubmitting || !!createSuccess}
                    rows="4"
                    style={{ resize: 'vertical' }}
                    required
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="sighting-lat">Latitude (optional, with longitude)</label>
                  <input
                    id="sighting-lat"
                    type="number"
                    step="any"
                    min="-90"
                    max="90"
                    className="form-control"
                    placeholder="e.g. 12.9716"
                    value={createLatitude}
                    onChange={(e) => setCreateLatitude(e.target.value)}
                    disabled={createSubmitting || !!createSuccess}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="sighting-lng">Longitude (optional, with latitude)</label>
                  <input
                    id="sighting-lng"
                    type="number"
                    step="any"
                    min="-180"
                    max="180"
                    className="form-control"
                    placeholder="e.g. 77.5946"
                    value={createLongitude}
                    onChange={(e) => setCreateLongitude(e.target.value)}
                    disabled={createSubmitting || !!createSuccess}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="sighting-contact">Contact Info (optional)</label>
                  <input
                    id="sighting-contact"
                    type="text"
                    className="form-control"
                    placeholder="How can investigators reach you?"
                    value={createContact}
                    onChange={(e) => setCreateContact(e.target.value)}
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
                  {createSubmitting ? 'Reporting…' : 'Report sighting'}
                </button>
              </div>
            </form>
        </Modal>
      )}

      {/* VIEW DETAILS MODAL */}
      {showDetailModal && (
        <Modal
          wide
          title={isEditing ? 'Edit sighting' : `Sighting #${detailSighting?.id || ''}`}
          onClose={() => !editSubmitting && setShowDetailModal(false)}
        >
              {detailLoading ? (
                <LoadingState label="Loading sighting detail…" />
              ) : detailError ? (
                <ErrorState detail={detailError} />
              ) : detailSighting ? (
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
                      <label className="form-label" htmlFor="edit-sighting-status">Status State</label>
                      <select
                        id="edit-sighting-status"
                        className="form-control"
                        value={editStatus}
                        onChange={(e) => setEditStatus(e.target.value)}
                        disabled={editSubmitting || !!editSuccess}
                      >
                        {SIGHTING_STATUSES.map((s) => (
                          <option key={s} value={s}>{s}</option>
                        ))}
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="form-label" htmlFor="edit-sighting-loc">Location</label>
                      <input
                        id="edit-sighting-loc"
                        type="text"
                        className="form-control"
                        value={editLocation}
                        onChange={(e) => setEditLocation(e.target.value)}
                        disabled={editSubmitting || !!editSuccess}
                        required
                      />
                    </div>

                    <div className="form-group">
                      <label className="form-label" htmlFor="edit-sighting-desc">Description</label>
                      <textarea
                        id="edit-sighting-desc"
                        className="form-control"
                        value={editDescription}
                        onChange={(e) => setEditDescription(e.target.value)}
                        disabled={editSubmitting || !!editSuccess}
                        rows="4"
                        style={{ resize: 'vertical' }}
                        required
                      />
                    </div>

                    <div className="form-group">
                      <label className="form-label" htmlFor="edit-sighting-contact">Contact Info (optional)</label>
                      <input
                        id="edit-sighting-contact"
                        type="text"
                        className="form-control"
                        value={editContact}
                        onChange={(e) => setEditContact(e.target.value)}
                        disabled={editSubmitting || !!editSuccess}
                      />
                    </div>
                  </form>
                ) : (
                  <div className="detail-grid">
                    <div className="detail-row">
                      <span className="detail-label">Sighting ID</span>
                      <span className="detail-value" style={{ fontWeight: 'bold', fontFamily: 'monospace' }}>
                        #{detailSighting.id}
                      </span>
                    </div>

                    <div className="detail-row">
                      <span className="detail-label">Related Case</span>
                      <span className="detail-value" style={{ fontWeight: '600' }}>
                        {caseTitleFor(detailSighting.case_id)}
                      </span>
                    </div>

                    <div className="detail-row">
                      <span className="detail-label">Status State</span>
                      <span className="detail-value">
                        <SightingStatusBadge status={detailSighting.status} />
                      </span>
                    </div>

                    <div className="detail-row">
                      <span className="detail-label">Sighted At</span>
                      <span className="detail-value">
                        {formatDate(detailSighting.sighting_at)}
                      </span>
                    </div>

                    <div className="detail-row">
                      <span className="detail-label">Location</span>
                      <span className="detail-value">
                        {detailSighting.location_text}
                      </span>
                    </div>

                    {(detailSighting.latitude ?? null) !== null && (
                      <div className="detail-row">
                        <span className="detail-label">Coordinates</span>
                        <span className="detail-value" style={{ fontFamily: 'monospace' }}>
                          {detailSighting.latitude}, {detailSighting.longitude}
                        </span>
                      </div>
                    )}

                    {detailSighting.contact_info && (
                      <div className="detail-row">
                        <span className="detail-label">Contact Info</span>
                        <span className="detail-value">
                          {detailSighting.contact_info}
                        </span>
                      </div>
                    )}

                    <div className="detail-row">
                      <span className="detail-label">Reported By</span>
                      <span className="detail-value" style={{ fontFamily: 'monospace' }}>
                        User ID: {detailSighting.reported_by}
                      </span>
                    </div>

                    <div style={{ marginTop: '1rem' }}>
                      <div className="detail-label" style={{ marginBottom: '0.5rem' }}>Observation Details:</div>
                      <div className="detail-value-desc">
                        {detailSighting.description}
                      </div>
                    </div>

                    <div className="evidence-section">
                      <SectionHeader
                        title="Sighting photos"
                        meta={photosLoading ? '…' : `${photos.length} photo${photos.length === 1 ? '' : 's'}`}
                      />
                      {uploadError && (
                        <div className="alert alert-error" role="alert">
                          {uploadError}
                        </div>
                      )}
                      {photosLoading ? (
                        <LoadingState label="Loading sighting photos…" />
                      ) : photosError ? (
                        <ErrorState detail={photosError} onRetry={() => fetchPhotos(detailSighting.case_id, detailSighting.id)} />
                      ) : photos.length === 0 ? (
                        <EmptyState
                          title="No sighting photos"
                          detail={canModifyDetail ? 'Upload the first photo below.' : null}
                        />
                      ) : (
                        <div className="photo-grid">
                          {photos.map((photo) => (
                            <PhotoCard
                              key={photo.id}
                              photo={photo}
                              alt={`Sighting photo ${photo.id}`}
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
                  </div>
                )
              ) : null}
            <div className="modal-footer">
              {detailSighting && !detailLoading && !detailError && canModifyDetail && (
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
                      onClick={() => promptDeleteSighting(detailSighting)}
                    >
                      Delete sighting
                    </button>
                    <button
                      type="button"
                      className="btn btn-primary"
                      onClick={startEditing}
                    >
                      Edit sighting
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
      {showDetailModal && aiPhoto && detailSighting && (
        <PhotoAIModal
          photo={aiPhoto}
          photoKind="sighting"
          caseId={detailSighting.case_id}
          sightingId={detailSighting.id}
          canEnhance={canEnhance}
          onClose={() => setAiPhotoId(null)}
          onPhotosChanged={() => fetchPhotos(detailSighting.case_id, detailSighting.id)}
        />
      )}

      {/* DELETE CONFIRMATION MODAL */}
      {showDeleteModal && sightingToDelete && (
        <Modal title="Confirm sighting deletion" onClose={cancelDeleteSighting}>
              {deleteError && (
                <div className="alert alert-error" role="alert">
                  {deleteError}
                </div>
              )}
              <p style={{ fontSize: '0.95rem', color: 'var(--text-primary)', marginBottom: '0.75rem' }}>
                Permanently delete <strong>Sighting #{sightingToDelete.id}</strong>?
              </p>
              <p style={{ fontSize: '0.85rem', color: 'var(--error)' }}>
                This cannot be undone. The report and its photos will be removed.
              </p>
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={cancelDeleteSighting}
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
                {deleteSubmitting ? 'Deleting…' : 'Delete sighting'}
              </button>
            </div>
        </Modal>
      )}
    </div>
  );
};

export default Sightings;
