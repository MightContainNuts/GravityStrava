from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlmodel import Session
from app.core.database import get_session
from app.services.strava import StravaClient
from app.services.withings import WithingsClient

router = APIRouter(tags=["views"])
templates = Jinja2Templates(directory="templates")

@router.get("/")
def read_root(session: Session = Depends(get_session)):
    strava = StravaClient(session)
    withings = WithingsClient(session)
    
    status = {
        "strava_auth": strava.has_token(),
        "withings_auth": withings.has_token()
    }
    
    # In a real app we'd Render a simple status page or redirect to /dashboard if authed
    return status

@router.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    import time
    return templates.TemplateResponse("dashboard.html", {"request": request, "v": int(time.time())})
