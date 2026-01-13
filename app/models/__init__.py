from datetime import datetime
from typing import Optional, List
from sqlmodel import Field, SQLModel
from pydantic import BaseModel

class Activity(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    strava_id: int = Field(index=True, unique=True)
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
    average_watts: Optional[float] = None
    max_watts: Optional[float] = None

class AthleteProfile(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    athlete_id: int = Field(index=True, unique=True)
    ftp: Optional[float] = None # Strava FTP
    ai_estimated_ftp: Optional[float] = None
    max_hr: Optional[float] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ActivityAnalysis(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    activity_id: int = Field(foreign_key="activity.id", index=True, unique=True)
    
    # Core Summary (derived from LLM)
    summary: Optional[str] = None
    session_type: Optional[str] = None
    session_intent_match: Optional[bool] = None
    suggested_title: Optional[str] = None
    
    # Calculated Metrics
    duration: float
    distance: float
    avg_power: Optional[float] = None
    normalized_power: Optional[float] = None
    avg_hr: Optional[float] = None
    max_hr: Optional[float] = None
    avg_cadence: Optional[float] = None
    elevation_gain: float
    
    # Efficiency
    power_hr_ratio: Optional[float] = None
    aerobic_efficiency_index: Optional[float] = None
    hr_drift: Optional[float] = None
    
    # Durability
    power_fade: Optional[float] = None
    late_ride_hr_diff: Optional[float] = None
    cadence_stability: Optional[float] = None
    
    # Intensity
    variability_index: Optional[float] = None
    time_above_hr_cap: Optional[float] = None
    tss: Optional[float] = None # Training Stress Score
    
    # Performance
    best_20min_power: Optional[float] = None
    ftp_estimate: Optional[float] = None
    avg_w_kg: Optional[float] = None
    ftp_w_kg: Optional[float] = None
    
    # Peak Power Duras
    peak_5s: Optional[float] = None
    peak_30s: Optional[float] = None
    peak_1min: Optional[float] = None
    peak_5min: Optional[float] = None
    peak_10min: Optional[float] = None
    peak_20min: Optional[float] = None
    peak_60min: Optional[float] = None
    
    # Subjective
    rpe: Optional[int] = None
    notes: Optional[str] = None
    
    # Zone Distribution (stored as JSON strings)
    hr_zones: Optional[str] = None
    power_zones: Optional[str] = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserHealth(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    date: datetime = Field(index=True)
    weight: Optional[float] = None
    fat_ratio: Optional[float] = None
    fat_mass_weight: Optional[float] = None
    muscle_mass: Optional[float] = None
    bone_mass: Optional[float] = None
    hydration: Optional[float] = None
    pulse: Optional[int] = None # RHR or manual pulse
    fitness_level: Optional[int] = None # Withings Fitness Score (VO2 Max estimate)
    source: str = "withings"
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DailyTrainingLoad(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    date: datetime = Field(index=True, unique=True)
    ctl: float # Fitness (Long-term load)
    atl: float # Fatigue (Short-term load)
    tsb: float # Form (Balance)
    total_tss: float
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class StravaToken(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    access_token: str
    refresh_token: str
    expires_at: int
    athlete_id: int

class WithingsToken(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    access_token: str
    refresh_token: str
    expires_at: int
    userid: str

# Pydantic Schemas for API
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
