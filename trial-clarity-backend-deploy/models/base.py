from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class RawBaseModel(SQLModel):
	status: bool = Field(default=True, index=True)
	deleted: bool = Field(default=False, index=True)

	created_at: datetime = Field(default=datetime.now(timezone.utc), nullable=False)
	updated_at: datetime = Field(default_factory=datetime.now, nullable=False)
	deleted_at: datetime = Field(default=None, nullable=True)

class BaseModel(RawBaseModel):
	created_by: int = Field(default=None, nullable=True)
	updated_by: int = Field(default=None, nullable=True)
	deleted_by: int = Field(default=None, nullable=True)
