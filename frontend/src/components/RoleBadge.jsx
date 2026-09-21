import React from 'react';

const ROLE_TONES = {
  ADMIN: ['badge-admin', 'Administrator'],
  REVIEWER: ['badge-reviewer', 'Reviewer'],
  ORGANIZATION_MEMBER: ['badge-org', 'Org Member'],
  REPORTER: ['badge-reporter', 'Reporter'],
};

export const RoleBadge = ({ role }) => {
  const [tone, label] = ROLE_TONES[role] || ROLE_TONES.REPORTER;
  return <span className={`badge ${tone}`}>{label}</span>;
};

export default RoleBadge;
