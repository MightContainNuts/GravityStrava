from typing import Optional
from pydantic import BaseModel
from datetime import datetime

class ActivityBase(BaseModel):
    strava_id: int
    name: str
    distance: float
    moving_time: int
    elapsed_time: int
    total_elevation_gain: float
    type: str
    sport_type: str
    start_date: datetime
    start_date_local: datetime
    timezone: str
    utc_offset: float
    average_speed: float
    max_speed: float
    has_heartrate: bool
    average_heartrate: Optional[float] = None
    max_heartrate: Optional[float] = None

class ActivityRead(ActivityBase):
    id: int

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_at: int
    athlete: dict
