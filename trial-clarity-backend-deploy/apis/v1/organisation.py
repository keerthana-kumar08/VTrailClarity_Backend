from fastapi import APIRouter, Depends, Query, status

from sqlmodel import Session, select

# from models.core import Organisation
# from core.database import get_db_session
# from apis.v1.schemas import OrganisationCreate, OrganisationResponse

from ...models.core import Organisation
from ...core.database import get_db_session

from .schemas import OrganisationCreate, OrganisationResponse

router = APIRouter()


@router.post("/create-organisation", response_model=OrganisationResponse, status_code=status.HTTP_201_CREATED)
async def create_organisation(org_data: OrganisationCreate, db: Session = Depends(get_db_session)) -> Organisation:
	organisation = Organisation(**org_data.model_dump())
	db.add(organisation)
	db.commit()
	db.refresh(organisation)
	return organisation

@router.get("/organisations", response_model=list[OrganisationResponse], status_code=status.HTTP_201_CREATED)
async def organisations(offset: int = 0, limit: int = Query(default=10, le=100), db: Session = Depends(get_db_session)):
	return db.exec(select(Organisation).where(Organisation.status == True, Organisation.deleted == False).offset(offset).limit(limit)).all()
