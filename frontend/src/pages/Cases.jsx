import React, { useState, useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';
import { caseService } from '../services/api';
import './Cases.css';

export const Cases = () => {
  const { user } = useAuth();
  
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
  const [editSubmitting, setEditSubmitting] = useState(false);
  const [editError, setEditError] = useState(null);
  const [editSuccess, setEditSuccess] = useState(null);

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

  const handleViewDetails = async (caseId) => {
    setShowDetailModal(true);
    setIsEditing(false);
    setEditError(null);
    setEditSuccess(null);
    setDetailLoading(true);
    setDetailError(null);
    setDetailCase(null);
    try {
      const data = await caseService.getCase(caseId);
      setDetailCase(data);
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

    setEditSubmitting(true);
    try {
      const updatedCase = await caseService.updateCase(detailCase.id, {
        title: editTitle.trim(),
        description: editDescription.trim(),
        status: editStatus,
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

    setCreateSubmitting(true);
    try {
      const newCase = await caseService.createCase({
        title: createTitle,
        description: createDescription,
      });
      setCreateSuccess('Case registered successfully!');
      setCreateTitle('');
      setCreateDescription('');
      
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

  const getStatusBadge = (status) => {
    switch (status) {
      case 'OPEN':
        return <span className="badge badge-org">Open</span>;
      case 'UNDER_REVIEW':
        return <span className="badge badge-reviewer">Under Review</span>;
      case 'RESOLVED':
        return <span className="badge badge-success">Resolved</span>;
      case 'CLOSED':
        return <span className="badge badge-reporter">Closed</span>;
      default:
        return <span className="badge">{status}</span>;
    }
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

  return (
    <div className="cases-page-container">
      {/* Top Controls Action Bar */}
      <div className="cases-actions-bar">
        <div className="search-filter-group">
          <input
            type="text"
            className="form-control"
            placeholder="Search by title or description..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            style={{ flex: 2 }}
          />
          <select
            className="form-control"
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            style={{ flex: 1, minWidth: '130px' }}
          >
            <option value="ALL">All Statuses</option>
            <option value="OPEN">Open</option>
            <option value="UNDER_REVIEW">Under Review</option>
            <option value="RESOLVED">Resolved</option>
            <option value="CLOSED">Closed</option>
          </select>
        </div>

        {canCreateCase && (
          <button 
            className="btn btn-primary" 
            onClick={() => {
              setCreateError(null);
              setCreateSuccess(null);
              setShowCreateModal(true);
            }}
          >
            ➕ Create Case
          </button>
        )}
      </div>

      {/* Main Cases Content */}
      {loading ? (
        <div className="cases-state-container">
          <span className="spinner" style={{ display: 'inline-block', marginBottom: '1rem' }} />
          <h2 style={{ fontSize: '1.15rem' }}>Loading Case Profiles...</h2>
          <p className="cases-state-text">Fetching data from the TraceLink central index.</p>
        </div>
      ) : error ? (
        <div className="alert alert-error text-center" style={{ padding: '2rem' }}>
          <h3>⚠️ System Index Unreachable</h3>
          <p style={{ marginTop: '0.5rem', fontSize: '0.9rem' }}>{error}</p>
          <button className="btn btn-secondary mt-4" onClick={fetchCases}>
            Retry Connection
          </button>
        </div>
      ) : filteredCases.length === 0 ? (
        <div className="cases-state-container">
          <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>📁</div>
          <h2 style={{ fontSize: '1.15rem' }}>No Cases Found</h2>
          <p className="cases-state-text">
            {searchTerm || statusFilter !== 'ALL'
              ? 'No case profiles match your current search queries or filters.'
              : 'There are no missing-person case profiles currently recorded.'}
          </p>
          {!searchTerm && statusFilter === 'ALL' && canCreateCase && (
            <button
              className="btn btn-primary mt-4"
              onClick={() => setShowCreateModal(true)}
            >
              Register First Case
            </button>
          )}
        </div>
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
                  <td>{getStatusBadge(caseItem.status)}</td>
                  <td>{formatDate(caseItem.created_at)}</td>
                  <td>{formatDate(caseItem.updated_at)}</td>
                  <td style={{ textAlign: 'center' }}>
                    <button
                      className="btn btn-secondary btn-block"
                      style={{ padding: '0.4rem 0.75rem', fontSize: '0.8rem' }}
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
        <div className="modal-overlay" onClick={() => !createSubmitting && setShowCreateModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Register New Case Profile</h2>
              <button 
                className="modal-close-btn" 
                onClick={() => setShowCreateModal(false)}
                disabled={createSubmitting}
              >
                &times;
              </button>
            </div>
            <form onSubmit={handleCreateSubmit}>
              <div className="modal-body">
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
                  {createSubmitting ? 'Registering...' : 'Register Case'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* VIEW DETAILS MODAL */}
      {showDetailModal && (
        <div className="modal-overlay" onClick={() => !editSubmitting && !isEditing && setShowDetailModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>{isEditing ? 'Edit Investigation Profile' : 'Case Investigation Profile'}</h2>
              <button 
                className="modal-close-btn" 
                onClick={() => !editSubmitting && setShowDetailModal(false)}
                disabled={editSubmitting}
              >
                &times;
              </button>
            </div>
            <div className="modal-body">
              {detailLoading ? (
                <div style={{ textAlign: 'center', padding: '2rem 0' }}>
                  <span className="spinner" style={{ display: 'inline-block', marginBottom: '1rem' }} />
                  <p className="cases-state-text">Loading case profile detail...</p>
                </div>
              ) : detailError ? (
                <div className="alert alert-error text-center" style={{ margin: 0 }}>
                  <h4>Access Failure</h4>
                  <p style={{ marginTop: '0.5rem', fontSize: '0.875rem' }}>{detailError}</p>
                </div>
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
                        {getStatusBadge(detailCase.status)}
                      </span>
                    </div>

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
                  </div>
                )
              ) : null}
            </div>
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
                      {editSubmitting ? 'Saving...' : 'Save Changes'}
                    </button>
                  </>
                ) : (
                  <>
                    <button 
                      type="button"
                      className="btn btn-danger" 
                      onClick={() => promptDeleteCase(detailCase)}
                    >
                      🗑️ Delete Case
                    </button>
                    <button 
                      type="button"
                      className="btn btn-primary" 
                      onClick={startEditing}
                    >
                      ✏️ Edit Case
                    </button>
                  </>
                )
              )}
              {!isEditing && (
                <button className="btn btn-secondary" onClick={() => setShowDetailModal(false)}>
                  Close
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* DELETE CONFIRMATION MODAL */}
      {showDeleteModal && caseToDelete && (
        <div className="modal-overlay" onClick={cancelDeleteCase}>
          <div className="modal-content" style={{ maxWidth: '460px' }} onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Confirm Case Deletion</h2>
              <button 
                className="modal-close-btn" 
                onClick={cancelDeleteCase}
                disabled={deleteSubmitting}
              >
                &times;
              </button>
            </div>
            <div className="modal-body">
              {deleteError && (
                <div className="alert alert-error">
                  {deleteError}
                </div>
              )}
              <p style={{ fontSize: '0.95rem', color: 'var(--text-primary)', marginBottom: '0.75rem' }}>
                Are you sure you want to permanently delete <strong>Case #{caseToDelete.id}</strong> (&ldquo;{caseToDelete.title}&rdquo;)?
              </p>
              <p style={{ fontSize: '0.85rem', color: 'var(--error)' }}>
                ⚠️ This action cannot be undone. All records associated with this case will be permanently removed.
              </p>
            </div>
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
                {deleteSubmitting ? 'Deleting...' : 'Delete Case'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Cases;
