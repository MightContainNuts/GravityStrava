from sqlmodel import create_engine, Session, SQLModel
from .config import settings
import app.models # Register models

sqlite_file_name = "strava_data.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=settings.debug if hasattr(settings, 'debug') else False)

def get_session():
    with Session(engine) as session:
        yield session

def create_db_and_tables():
    from app import models # Import to register models
    SQLModel.metadata.create_all(engine)
