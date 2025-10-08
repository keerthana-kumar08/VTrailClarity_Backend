import asyncio

from fastapi import FastAPI
from contextlib import asynccontextmanager
from sqlmodel import create_engine, SQLModel

from .apis import v1
from .core.config import settings
from .utils.scheduler import batch_insert_messages     
from fastapi.middleware.cors import CORSMiddleware


engine = create_engine(settings.DATABASE_URI, echo=False)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_db_client(app)
    asyncio.create_task(batch_insert_messages())
    yield

async def startup_db_client(app):
    create_db_and_tables()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

app.include_router(v1.api_router)
