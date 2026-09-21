"""phase 0: organizations, memberships, structured case fields

Revision ID: b7e2f41a9c06
Revises: 9c0ca14afef7
Create Date: 2026-09-15

Phase 0 foundation hardening:
- new `organizations` and `memberships` tables (global UserRole untouched)
- `cases.organization_id` NULLABLE: NULL = personal case, shared with no
  organization. Existing rows keep NULL and stay fully valid.
- optional structured missing-person columns on `cases`; every one is
  nullable so pre-Phase-0 rows remain valid unchanged. `title` and
  `description` are preserved as-is.

Non-destructive: only CREATE TABLE / ADD COLUMN; no data migration,
no invented organization assignments.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7e2f41a9c06'
down_revision: Union[str, Sequence[str], None] = '9c0ca14afef7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

org_role_enum = sa.Enum(
    'ORG_ADMIN', 'INVESTIGATOR', 'VIEWER', name='orgrole'
)


def upgrade() -> None:
    # --- organizations ---
    # NOTE: no explicit org_role_enum.create() here. op.create_table()
    # below creates the PostgreSQL ENUM type implicitly via the
    # memberships.role column; an explicit create() on top of that
    # emits CREATE TYPE twice and fails with DuplicateObject.
    # (op.add_column does NOT auto-create enums, but op.create_table DOES.)
    op.create_table(
        'organizations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_by', sa.Integer(), nullable=False),
        sa.Column(
            'created_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            'updated_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.ForeignKeyConstraint(['created_by'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_organizations_created_by'),
        'organizations', ['created_by'], unique=False,
    )
    op.create_index(
        op.f('ix_organizations_id'), 'organizations', ['id'], unique=False,
    )
    op.create_index(
        op.f('ix_organizations_name'), 'organizations', ['name'], unique=False,
    )

    # --- memberships ---
    op.create_table(
        'memberships',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('organization_id', sa.Integer(), nullable=False),
        sa.Column('role', org_role_enum, nullable=False),
        sa.Column(
            'created_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            'updated_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.ForeignKeyConstraint(['organization_id'], ['organizations.id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'user_id', 'organization_id', name='uq_membership_user_org'
        ),
    )
    op.create_index(
        op.f('ix_memberships_id'), 'memberships', ['id'], unique=False,
    )
    op.create_index(
        op.f('ix_memberships_organization_id'),
        'memberships', ['organization_id'], unique=False,
    )
    op.create_index(
        op.f('ix_memberships_user_id'),
        'memberships', ['user_id'], unique=False,
    )

    # --- cases: organization scope + structured person fields ---
    # All nullable: existing rows are untouched and remain valid.
    op.add_column(
        'cases', sa.Column('organization_id', sa.Integer(), nullable=True)
    )
    op.create_foreign_key(
        'fk_cases_organization_id',
        'cases', 'organizations',
        ['organization_id'], ['id'],
    )
    op.create_index(
        op.f('ix_cases_organization_id'),
        'cases', ['organization_id'], unique=False,
    )
    op.add_column(
        'cases', sa.Column('full_name', sa.String(length=200), nullable=True)
    )
    op.add_column(
        'cases', sa.Column('age_years', sa.Integer(), nullable=True)
    )
    op.add_column(
        'cases',
        sa.Column('age_estimate_note', sa.String(length=255), nullable=True),
    )
    op.add_column(
        'cases',
        sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        'cases',
        sa.Column('last_seen_location', sa.String(length=500), nullable=True),
    )
    op.add_column(
        'cases', sa.Column('clothing_description', sa.Text(), nullable=True)
    )
    op.add_column(
        'cases', sa.Column('distinguishing_marks', sa.Text(), nullable=True)
    )
    op.add_column(
        'cases',
        sa.Column('contact_info', sa.String(length=500), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('cases', 'contact_info')
    op.drop_column('cases', 'distinguishing_marks')
    op.drop_column('cases', 'clothing_description')
    op.drop_column('cases', 'last_seen_location')
    op.drop_column('cases', 'last_seen_at')
    op.drop_column('cases', 'age_estimate_note')
    op.drop_column('cases', 'age_years')
    op.drop_column('cases', 'full_name')
    op.drop_index(op.f('ix_cases_organization_id'), table_name='cases')
    op.drop_constraint(
        'fk_cases_organization_id', 'cases', type_='foreignkey'
    )
    op.drop_column('cases', 'organization_id')

    op.drop_index(op.f('ix_memberships_user_id'), table_name='memberships')
    op.drop_index(
        op.f('ix_memberships_organization_id'), table_name='memberships'
    )
    op.drop_index(op.f('ix_memberships_id'), table_name='memberships')
    op.drop_table('memberships')

    op.drop_index(
        op.f('ix_organizations_name'), table_name='organizations'
    )
    op.drop_index(op.f('ix_organizations_id'), table_name='organizations')
    op.drop_index(
        op.f('ix_organizations_created_by'), table_name='organizations'
    )
    op.drop_table('organizations')
    org_role_enum.drop(op.get_bind(), checkfirst=True)
