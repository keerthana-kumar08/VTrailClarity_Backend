from fastapi import APIRouter

from . import auth, organisation, trial


api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["AUTH"])
api_router.include_router(organisation.router, prefix="/organisation", tags=["ORGANISATION"])
api_router.include_router(trial.router, prefix="/trial", tags=["TRIAL"])
