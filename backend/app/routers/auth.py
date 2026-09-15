from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import (hash_password, verify_password, create_access_token)
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin
from app.core.dependencies import get_current_user, require_role
from app.models.user import User, UserRole
from app.schemas.user import AdminCreate


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


@router.post(
    "/register",
    status_code=status.HTTP_201_CREATED
)
def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    existing_user = (
        db.query(User)
        .filter(User.email == user_data.email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

    new_user = User(
        name=user_data.name,
        email=user_data.email,
        password_hash=hash_password(user_data.password)
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {
        "message": "User registered successfully",
        "user_id": new_user.id
    }


@router.post("/login")
def login(
    user_data: UserLogin,
    db: Session = Depends(get_db)
):
    user = (
        db.query(User)
        .filter(User.email == user_data.email)
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    if not verify_password(
        user_data.password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    access_token = create_access_token(
        user_id=user.id,
        role=user.role.value
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.get("/me")
def get_me(
    current_user: User = Depends(get_current_user)
):
    return {
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "role": current_user.role.value
    }

@router.get("/admin-test")
def admin_test(
    current_user: User = Depends(
        require_role(UserRole.ADMIN)
    )
):
    return {
        "message": "You are an admin",
        "user_id": current_user.id
    }


@router.post(
    "/admin",
    status_code=status.HTTP_201_CREATED
)
def create_admin(
    admin_data: AdminCreate,
    current_user: User = Depends(
        require_role(UserRole.ADMIN)
    ),
    db: Session = Depends(get_db)
):
    existing_user = db.query(User).filter(
        User.email == admin_data.email
    ).first()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    admin = User(
        name=admin_data.name,
        email=admin_data.email,
        password_hash=hash_password(admin_data.password),
        role=UserRole.ADMIN
    )

    db.add(admin)
    db.commit()
    db.refresh(admin)

    return {
        "message": "Admin created successfully",
        "admin_id": admin.id,
        "name": admin.name,
        "email": admin.email,
        "role": admin.role
    }