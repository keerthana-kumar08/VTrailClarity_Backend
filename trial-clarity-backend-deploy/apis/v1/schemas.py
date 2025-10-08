from datetime import datetime

from typing import Optional, List
from pydantic import BaseModel


class OrganisationCreate(BaseModel):
	name: str
	email: str
	phone: str

class OrganisationResponse(OrganisationCreate):
	id: int

class TrialCreate(BaseModel):
    id: Optional[int] = None
    organisation_id: int
    name: str
    type_id: Optional[int] = None
    description: Optional[str] = None
    # primary_column: Optional[str] = None
    # secondary_column: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    design_id: Optional[int] = None
    control_preset: bool
    randomized: bool
    blinded: bool
    alpha: Optional[float] = None
    beta: Optional[float] = None
    delta: Optional[str] = None


class EndPointColumn(BaseModel):
	type: str
	variable_name: str
	group: Optional[str] = None
	operation: Optional[str] = None
	condition: Optional[str] = None
	continous_value: Optional[float] = None
	cut_off: Optional[float] = None
	category_value: Optional[str] = None
	req_validate: Optional[bool]  = None


class TrialEndPointResponse(BaseModel):
	id: int
	name: Optional[str] = None
	trial_id: int
	column: str
	column_data: Optional[List[EndPointColumn]] = None
	is_primary: bool
	is_weekly_cumilative: bool


class TrialResponse(BaseModel):
	id: int
	organisation_id: int
	name: str
	type_id : int | None
	trial_type :Optional[str] = None
	description: str | None
	endpoints: Optional[List[TrialEndPointResponse]] = None
	start_date: datetime
	end_date: datetime
	design_id : int | None
	trial_design: Optional[str] = None
	control_preset: bool
	randomized: bool
	blinded: bool
	rec_to_stop: Optional[bool] = False
	result_id: Optional[int] = None
	alpha: Optional[float] = None
	beta: Optional[float] = None
	delta: Optional[str] = None


class UserCreate(BaseModel):
	organisation_id: Optional[int]
	first_name: Optional[str]
	last_name: Optional[str]
	email: str
	phone: str
	password: str

class UserResponse(BaseModel):
	id: int
	first_name: Optional[str]
	last_name: Optional[str]
	email: str
	phone: str
	is_superuser: bool
	organisation_id: int | None
	passcode: str


class file_headers(BaseModel):
	name: Optional[str]
	values_type: Optional[str]


class LoginRequest(BaseModel):
    email: str
    password: str


class TrialPlotRequest(BaseModel):
	trial_id: int
	columns: str
	endpoint_id: Optional[int] = None
	endpoint_data: Optional[TrialEndPointResponse] = None
	cumulative_dates: Optional[dict] = None


class TrialTypeResponse(BaseModel):
	id: int
	name: str
	
class TrialDesignResponse(BaseModel):
	id: int
	name: str


class TrialChatHistoryResponse(BaseModel):
	id: int
	role: str
	content: str
	timestamp: datetime | None
	

class TrialEndPointCreate(BaseModel):
	id: Optional[int] = None
	name: Optional[str] = None
	trial_id: int
	column: str
	is_primary: bool
	is_weekly_cumilative: bool
	column_data: List[EndPointColumn]


class TrialUniqueValueRequest(BaseModel):
	trial_id: int
	columns: str