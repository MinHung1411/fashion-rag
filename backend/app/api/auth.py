from typing import Optional, Dict, Any
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session
from ..core.database import get_db
from ..core.security import hash_password, verify_password, create_access_token
from ..models.user import User
from ..schemas.auth import UserRegister, UserLogin, UserOut, Token
from ..services.auth_service import get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=Token)
def register(user_in: UserRegister, db: Session = Depends(get_db)):
    # Check if username exists
    existing_username = db.query(User).filter(User.username == user_in.username).first()
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Tên người dùng đã được sử dụng. Vui lòng chọn tên khác."
        )

    # Check if email exists
    existing_email = db.query(User).filter(User.email == user_in.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email đã được đăng ký. Vui lòng chọn email khác."
        )

    new_user = User(
        username=user_in.username,
        email=user_in.email,
        full_name=user_in.full_name or user_in.username,
        hashed_password=hash_password(user_in.password),
        avatar=f"https://api.dicebear.com/7.x/bottts/svg?seed={user_in.username}"
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    token = create_access_token(subject=new_user.id)
    return Token(access_token=token, user=UserOut.model_validate(new_user))

@router.post("/login", response_model=Token)
def login(login_in: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == login_in.username).first()
    if not user or not verify_password(login_in.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tên đăng nhập hoặc mật khẩu không chính xác."
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tài khoản này đã bị khóa."
        )

    token = create_access_token(subject=user.id)
    return Token(access_token=token, user=UserOut.model_validate(user))

@router.post("/quick-demo", response_model=Token)
def quick_demo_login(payload: Optional[Dict[str, Any]] = Body(default=None), db: Session = Depends(get_db)):
    guest_id = (payload or {}).get("guest_id")
    if guest_id:
        clean_guest = "guest_" + "".join(c for c in guest_id if c.isalnum())[:16]
        user = db.query(User).filter(User.username == clean_guest).first()
        if not user:
            user = User(
                username=clean_guest,
                email=f"{clean_guest}@fashionrag.ai",
                full_name="Khách trải nghiệm",
                hashed_password=hash_password("guest123"),
                avatar=f"https://api.dicebear.com/7.x/bottts/svg?seed={clean_guest}",
                is_active=True
            )
            db.add(user)
            db.commit()
            db.refresh(user)
    else:
        user = db.query(User).filter(User.username == "demo").first()
        if not user:
            user = User(
                username="demo",
                email="demo@fashionrag.ai",
                full_name="Khách trải nghiệm",
                hashed_password=hash_password("demo123"),
                avatar="https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=150&auto=format&fit=crop&q=80",
                is_active=True
            )
            db.add(user)
            db.commit()
            db.refresh(user)

    token = create_access_token(subject=user.id)
    return Token(access_token=token, user=UserOut.model_validate(user))

@router.get("/me", response_model=UserOut)
def get_me(current_user: User = Depends(get_current_user)):
    return UserOut.model_validate(current_user)
