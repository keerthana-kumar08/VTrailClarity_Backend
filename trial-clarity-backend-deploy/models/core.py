from datetime import datetime, timezone, timedelta

from passlib.context import CryptContext

from sqlmodel import Field, Relationship, Column
from sqlalchemy.dialects.postgresql import JSONB

# from models.base import BaseModel, RawBaseModel
# from core.config import settings
from typing import Optional
from .base import BaseModel, RawBaseModel
from ..core.config import settings

pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")


class TrialDesign(BaseModel, table=True):
	id: int = Field(primary_key=True, index=True)
	name: str
	code: int  | None
	description: str | None
	
	design_trials: list["Trial"] = Relationship(back_populates="trial_design")


class TrialType(BaseModel, table=True):
	id: int = Field(primary_key=True, index=True)
	name: str
	code: int  | None
	description: str  | None

	type_trials: list["Trial"] = Relationship(back_populates="trial_type")


class Organisation(BaseModel, table=True):
	id: int = Field(primary_key=True, index=True)
	name: str
	email: str
	phone: str

	organisation_users: list["User"] = Relationship(back_populates="organisation")
	organisation_trials: list["Trial"] = Relationship(back_populates="organisation")


class User(BaseModel, table=True):
	id: int = Field(primary_key=True, index=True)
	organisation_id: int | None = Field(foreign_key="organisation.id", nullable=True)
	first_name: str | None = None
	last_name: str | None = None
	email: str = Field(unique=True, index=True)
	phone: str = Field(unique=True, index=True)
	is_superuser: bool = Field(default=False)
	hashed_password: str

	organisation: Organisation = Relationship(back_populates="organisation_users")
	user_passcodes: list["UserPasscode"] = Relationship(back_populates="user")

	def verify_password(self, password: str) -> bool:
		return pwd_context.verify(password, self.hashed_password)

	@staticmethod
	def get_password_hash(password: str) -> str:
		return pwd_context.hash(password)


class UserPasscode(RawBaseModel, table=True):
	user_id: int = Field(foreign_key="user.id")
	passcode: str = Field(primary_key=True, index=True)
	expiry: datetime = Field(default=datetime.now(timezone.utc) + timedelta(days=settings.TOKEN_EXPIRE_DAYS), nullable=False)

	user: User = Relationship(back_populates="user_passcodes")


class Trial(BaseModel, table=True):
    id: int = Field(primary_key=True, index=True)
    organisation_id: int = Field(foreign_key="organisation.id")
    type_id: int | None = Field(default=None, foreign_key="trialtype.id", nullable=True)
    design_id: int | None = Field(default=None, foreign_key="trialdesign.id", nullable=True)
    name: str
    description: str | None
    start_date: datetime | None
    end_date: datetime | None
    control_preset: bool
    randomized: bool
    blinded: bool

    alpha: Optional[float] = Field(default=None, nullable=True)
    beta: Optional[float] = Field(default=None, nullable=True)
    delta: Optional[str] = Field(default=None, nullable=True)

    organisation: Organisation = Relationship(back_populates="organisation_trials")
    trial_files: list["TrialFile"] = Relationship(back_populates="trial")
    trial_endpoint: list["TrialEndPoint"] = Relationship(back_populates="trial_record")
    trial_type: Optional[TrialType] = Relationship(back_populates="type_trials")
    trial_design: Optional[TrialDesign] = Relationship(back_populates="design_trials")
    trial_result_data: list["TrialResult"] = Relationship(back_populates="trial_result")


class TrialEndPoint(BaseModel, table=True):
	id: int = Field(primary_key=True, index=True)
	trial_id: int = Field(foreign_key="trial.id")
	column: str
	name: str | None
	is_primary: bool = Field(default=False)
	is_weekly_cumilative: bool = Field(default=True)
	column_data: dict = Field(sa_column=Column(JSONB))
	
	trial_record: Trial = Relationship(back_populates="trial_endpoint")
	result_data: list["TrialResult"] = Relationship(back_populates="result_endpoint")


class TrialFile(BaseModel, table=True):
	id: int = Field(primary_key=True, index=True)
	trial_id: int = Field(foreign_key="trial.id")
	name : str
	row_count : int
	file_type : str
	data: dict = Field(sa_column=Column(JSONB))
	file_headers: list = Field(sa_column=Column(JSONB))

	trial: Trial = Relationship(back_populates="trial_files")
	trial_data: list["TrialData"] = Relationship(back_populates="trial_file")


class TrialData(BaseModel, table=True):
	id: int = Field(primary_key=True, index=True)
	trial_file_id: int = Field(foreign_key="trialfile.id")
	row: int
	attribute_key: str
	attribute_value: str

	trial_file: TrialFile = Relationship(back_populates="trial_data")


class TrialResult(BaseModel, table=True):
	id: int = Field(primary_key=True, index=True)
	trial_id: int = Field(foreign_key="trial.id")
	endpoint_id: int | None = Field(default=None, foreign_key="trialendpoint.id", nullable=True)
	start_date: datetime | None
	end_date: datetime | None
	week_start_date: datetime | None
	columns: str
	bayesian_value: dict = Field(sa_column=Column(JSONB))
	result: dict = Field(sa_column=Column(JSONB))
	avg_result: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
	avg_validation: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
	is_system_generated: bool = Field(default=False)
	rec_to_stop: bool = Field(default=False)
	
	trial_result: Trial = Relationship(back_populates="trial_result_data")
	result_endpoint: TrialEndPoint = Relationship(back_populates="result_data")
	chat_history: list["TrialResultChatHistory"] = Relationship(back_populates="result")


class TrialResultChatHistory(BaseModel, table=True):
    id: int = Field(primary_key=True, index=True)
    result_id: int = Field(foreign_key="trialresult.id")
    role: str
    content: str
    timestamp: datetime

    result: TrialResult = Relationship(back_populates="chat_history")

    