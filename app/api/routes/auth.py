from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlmodel import Session
from app.core.database import get_session
from app.services.strava import StravaClient
from app.services.withings import WithingsClient

router = APIRouter(tags=["auth"])

@router.get("/login")
def login(session: Session = Depends(get_session)):
    strava = StravaClient(session)
    return RedirectResponse(strava.get_auth_url())

@router.get("/withings/login")
def withings_login(session: Session = Depends(get_session)):
    withings = WithingsClient(session)
    return RedirectResponse(withings.get_auth_url())

@router.get("/callback")
async def callback(code: str, session: Session = Depends(get_session), state: str = None):
    # Handle both Strava and Withings callbacks
    # In a more robust app, state would differentiate or have different endpoints
    if state == "withings_auth_state":
        withings = WithingsClient(session)
        await withings.exchange_token(code)
    else:
        strava = StravaClient(session)
        await strava.exchange_token(code)
    
    return RedirectResponse(url="/dashboard")

@router.get("/withings/refresh")
async def withings_refresh(session: Session = Depends(get_session)):
    withings = WithingsClient(session)
    if not withings.has_token():
        return RedirectResponse(withings.get_auth_url())
    measures = await withings.fetch_measures()
    withings.sync_measures(measures)
    return RedirectResponse(url="/dashboard")

@router.get("/strava/refresh")
async def strava_refresh(session: Session = Depends(get_session)):
    strava = StravaClient(session)
    if not strava.has_token():
        return RedirectResponse(strava.get_auth_url())
    activities = await strava.fetch_activities()
    strava.sync_activities(activities)
    return RedirectResponse(url="/dashboard")
