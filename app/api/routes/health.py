from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from app.core.database import get_session
from app.models import UserHealth
from app.services.withings import WithingsClient

router = APIRouter(tags=["health"])

@router.get("/sync/health")
async def sync_health(session: Session = Depends(get_session)):
    withings = WithingsClient(session)
    if not withings.has_token():
        raise HTTPException(status_code=401, detail="Withings not authenticated")
    
    measures = await withings.fetch_measures()
    withings.sync_measures(measures)
    return {"status": "success"}

@router.get("/health/latest")
def get_latest_health(session: Session = Depends(get_session)):
    statement = select(UserHealth).order_by(UserHealth.date.desc())
    latest = session.exec(statement).first()
    return latest
