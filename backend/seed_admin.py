import sys
import os

# Add the current directory to python path to resolve imports correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.database import SessionLocal
from app.core.security import hash_password
from app.models.user import User, UserRole


def bootstrap_admin():
    db = SessionLocal()
    try:
        email = "admin@tracelink.com"

        # Check if the user already exists
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            if existing_user.role == UserRole.ADMIN:
                print(f"Admin already exists with email '{email}'. Exiting successfully.")
                return
            else:
                print(
                    f"Error: Email '{email}' already belongs to a non-admin user ({existing_user.role.value}). "
                    "Role will not be modified. Exiting without modifying."
                )
                return

        # Create new admin user
        admin_user = User(
            name="Local Admin",
            email=email,
            password_hash=hash_password("AdminSecurePass123!"),
            role=UserRole.ADMIN
        )
        db.add(admin_user)
        db.commit()
        print(f"Successfully bootstrapped admin account: {email}")

    except Exception as e:
        print(f"Error bootstrapping admin: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    bootstrap_admin()
