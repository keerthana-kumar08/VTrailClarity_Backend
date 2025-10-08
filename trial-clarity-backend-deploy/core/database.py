from sqlmodel import create_engine, Session
from contextlib import contextmanager

from .config import settings


# engine = create_engine(settings.DATABASE_URI)


# @contextmanager
# def db_session_context():
#     db = next(get_db_session())  # Get the generator's yielded value
#     try:
#         yield db
#     finally:
#         db.close()



engine = create_engine(
    settings.DATABASE_URI,
    pool_size=10,        # recommended defaults
    max_overflow=5,
    pool_timeout=30,
    pool_recycle=1800
)

def get_db_session():
    with Session(engine) as session:
        yield session

@contextmanager
def db_session_context():
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()