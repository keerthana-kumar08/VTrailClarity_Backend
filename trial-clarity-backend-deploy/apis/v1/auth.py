from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from secrets import token_hex

from sqlmodel import Session, select

# from models import User, UserPasscode, Organisation
# from core.database import get_db_session
# from apis.v1.schemas import UserCreate, UserResponse, LoginRequest

from ...models import User, UserPasscode, Organisation
from ...core.database import get_db_session
from .schemas import UserCreate, UserResponse, LoginRequest


router = APIRouter()

bearer_scheme = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme), db: Session = Depends(get_db_session)) -> User:
	credentials_exception = HTTPException(
		status_code=status.HTTP_401_UNAUTHORIZED,
		detail="Could not validate credentials",
		headers={"WWW-Authenticate": "Bearer"},
	)
	
	user_passcode = db.get(UserPasscode, credentials.credentials)

	if not user_passcode:
		raise credentials_exception

	if user_passcode.expiry and user_passcode.expiry < datetime.now(timezone.utc):
		raise HTTPException(
			status_code=status.HTTP_401_UNAUTHORIZED,
			detail="Token has expired",
			headers={"WWW-Authenticate": "Bearer"},
		)
	return user_passcode.user

@router.post("/register/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_in: UserCreate, db: Session = Depends(get_db_session)) -> UserResponse:
	user = db.exec(select(User).where(User.email == user_in.email)).first()
	if user:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Email already registered"
		)

	if user_in.organisation_id:
		organisation = db.get(Organisation, user_in.organisation_id)
		if not organisation:
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail="Organisation doesn't exist!"
			)

	hashed_password = User.get_password_hash(user_in.password)
	passcode = token_hex(32)
	user_data = User( **user_in.model_dump(), hashed_password=hashed_password)
	db.add(user_data)
	db.commit()
	db.refresh(user_data)
	user_passcode_data = UserPasscode.model_validate({ "user_id": user_data.id, "passcode": passcode })
	db.add(user_passcode_data)
	db.commit()
	return UserResponse(
		id=user_data.id,
		first_name=user_data.first_name,
		last_name=user_data.last_name,
		email=user_data.email,
		phone=user_data.phone,
		is_superuser=user_data.is_superuser,
		passcode=passcode,
		organisation_id=user_data.organisation_id
	)

@router.post("/login/")
async def login(login_data: LoginRequest, db: Session = Depends(get_db_session)) -> dict:
    user = db.exec(select(User).where(User.email == login_data.email)).first()
    if not user or not user.verify_password(login_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    passcode = token_hex(32)
    user_passcode_data = UserPasscode.model_validate({"user_id": user.id, "passcode": passcode})
    db.add(user_passcode_data)
    db.commit()
    db.refresh(user_passcode_data)    
    return {
		"status": 200,
		"message": "Success",
		"data" : {
			"passcode_type": "Passcode", 
			"passcode": passcode
		}
	}