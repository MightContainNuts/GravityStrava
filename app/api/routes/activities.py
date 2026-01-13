from typing import List
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select
from app.core.database import get_session
from app.models import Activity, ActivityRead
from app.services.strava import StravaClient

router = APIRouter(tags=["activities"])

@router.get("/sync/activities")
async def sync_activities(session: Session = Depends(get_session)):
    strava = StravaClient(session)
    if not strava.has_token():
        raise HTTPException(status_code=401, detail="Strava not authenticated")
    
    # Sync last 30 days by default or something
    activities = await strava.fetch_activities()
    count = strava.sync_activities(activities)
    return {"status": "success", "new_activities": count}

@router.get("/activities", response_model=List[ActivityRead])
def list_activities(session: Session = Depends(get_session)):
    return session.exec(select(Activity).order_by(Activity.start_date.desc())).all()

@router.get("/activities/pandas")
def get_activities_pandas(session: Session = Depends(get_session)):
    import numpy as np
    statement = select(Activity).order_by(Activity.start_date.desc())
    results = session.exec(statement).all()
    
    # Convert to list of dicts for pandas
    data = [a.model_dump() for a in results]
    df = pd.DataFrame(data)
    
    # Clean up dates for JSON
    if not df.empty:
        df['start_date'] = df['start_date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df['start_date_local'] = df['start_date_local'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
    # Handle NaN values for JSON compliance
    return df.replace({np.nan: None}).to_dict(orient="records")

@router.post("/strava/sync-history")
async def sync_history(session: Session = Depends(get_session)):
    """Sync all historic activities."""
    strava = StravaClient(session)
    if not strava.has_token():
        raise HTTPException(status_code=401, detail="Strava not authenticated")
    
    activities = await strava.fetch_activities()
    count = strava.sync_activities(activities)
    return {"status": "success", "total_activities": count}
