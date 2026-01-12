from contextlib import asynccontextmanager
from typing import List
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session, select, or_
from database import create_db_and_tables, get_session, Activity, ActivityAnalysis, UserHealth, AthleteProfile, DailyTrainingLoad
from strava_api import StravaClient
from withings_api import WithingsClient
from ai_analysis import run_activity_analysis, update_daily_training_load
from models import ActivityRead
from config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(title="Strava to SQLite Sync", lifespan=lifespan)

# Static and Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_root(session: Session = Depends(get_session)):
    client = StravaClient(session)
    if not client.has_token():
        return RedirectResponse(url="/auth/login")
    
    try:
        # This will trigger token refresh if needed and sync
        after = client.get_latest_activity_timestamp()
        activities_data = await client.fetch_activities(after=after)
        client.sync_activities(activities_data)
        return RedirectResponse(url="/dashboard")
    except Exception as e:
        # Fallback to login if something goes wrong with the existing token
        if "No Strava token found" in str(e) or "401" in str(e):
            return RedirectResponse(url="/auth/login")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/auth/login")
def login(session: Session = Depends(get_session)):
    if not settings.strava_client_id or not settings.strava_client_secret:
        raise HTTPException(
            status_code=500,
            detail="Strava API credentials are not configured. Please set STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET in your .env file."
        )
    client = StravaClient(session)
    return RedirectResponse(client.get_auth_url())

@app.get("/auth/withings/login")
def withings_login(session: Session = Depends(get_session)):
    if not settings.withings_client_id or not settings.withings_client_secret:
        raise HTTPException(
            status_code=500,
            detail="Withings API credentials are not configured."
        )
    client = WithingsClient(session)
    return RedirectResponse(client.get_auth_url())

@app.get("/callback")
async def callback(code: str, session: Session = Depends(get_session), state: str = None):
    # Differentiate between Strava and Withings based on state
    if state == "withings_auth_state":
        client = WithingsClient(session)
        try:
            await client.exchange_token(code)
            measures = await client.fetch_measures()
            client.sync_measures(measures)
            return RedirectResponse(url="/dashboard")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Withings Link Failed: {str(e)}")
    else:
        # Default to Strava flow
        client = StravaClient(session)
        try:
            await client.exchange_token(code)
            return RedirectResponse(url="/")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Strava Link Failed: {str(e)}")

@app.get("/withings/refresh")
async def withings_refresh(session: Session = Depends(get_session)):
    client = WithingsClient(session)
    try:
        measures = await client.fetch_measures()
        client.sync_measures(measures)
        return RedirectResponse(url="/dashboard")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Withings Refresh Failed: {str(e)}")

@app.get("/strava/refresh")
async def strava_refresh(session: Session = Depends(get_session)):
    client = StravaClient(session)
    try:
        # Sync Athlete Profile (FTP)
        try:
            athlete_data = await client.get_athlete()
            client.sync_athlete(athlete_data)
        except Exception as profile_err:
            print(f"Profile sync skipped: {profile_err}")
            
        after = client.get_latest_activity_timestamp()
        activities_data = await client.fetch_activities(after=after)
        client.sync_activities(activities_data)
        await client.sync_athlete_zones()
        return RedirectResponse(url="/dashboard")
    except StravaClient.StravaReauthRequired:
        return RedirectResponse(url=client.get_auth_url())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Strava Refresh Failed: {str(e)}")

@app.post("/health/sync")
async def sync_health(session: Session = Depends(get_session)):
    client = WithingsClient(session)
    try:
        # Get last sync date
        statement = select(UserHealth).order_by(UserHealth.date.desc())
        latest = session.exec(statement).first()
        startdate = int(latest.date.timestamp()) if latest else None
        
        measures = await client.fetch_measures(startdate=startdate)
        client.sync_measures(measures)
        return {"message": f"Successfully synced {len(measures)} health records."}
    except Exception as e:
        error_msg = str(e) or f"An unexpected error occurred: {type(e).__name__}"
        print(f"Sync failed: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=error_msg)

@app.post("/activities/sync")
async def sync_activities(session: Session = Depends(get_session)):
    client = StravaClient(session)
    try:
        after = client.get_latest_activity_timestamp()
        activities_data = await client.fetch_activities(after=after)
        client.sync_activities(activities_data)
        await client.sync_athlete_zones()
        return {"message": f"Successfully synced {len(activities_data)} new activities."}
    except StravaClient.StravaReauthRequired:
        return {"reauth_url": client.get_auth_url(), "message": "Re-authorization required for full profile access."}
    except Exception as e:
        error_msg = str(e) or f"An unexpected error occurred: {type(e).__name__}"
        print(f"Sync failed: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=error_msg)

@app.get("/activities/pandas")
def get_activities_pandas(session: Session = Depends(get_session)):
    try:
        statement = select(Activity).order_by(Activity.start_date.desc())
        activities = session.exec(statement).all()
        
        if not activities:
            return []

        df = pd.DataFrame([a.model_dump() for a in activities])
        
        # Convert datetime columns to ISO format strings for reliability
        for col in df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
            df[col] = df[col].dt.strftime('%Y-%m-%dT%H:%M:%S')

        # Replace NaN with None for JSON compatibility. 
        # We cast to 'object' first to prevent None being turned back into NaN in float columns.
        result = df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")
        return result
    except Exception as e:
        print(f"ERROR in /activities/pandas: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/activities/", response_model=List[ActivityRead])
def list_activities(session: Session = Depends(get_session)):
    statement = select(Activity).order_by(Activity.start_date.desc())
    activities = session.exec(statement).all()
    return activities

@app.post("/ai/analyze-all")
async def analyze_all_activities(session: Session = Depends(get_session)):
    """Trigger AI analysis for all activities that haven't been processed yet."""
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        raise HTTPException(
            status_code=400, 
            detail="OpenAI API Key is missing or invalid. Please update your .env file."
        )
        
    client = StravaClient(session)
    # Find activities without analysis OR missing TSS (legacy), limit to 10 per batch
    statement = select(Activity).where(
        or_(
            Activity.id.not_in(select(ActivityAnalysis.activity_id)),
            Activity.id.in_(select(ActivityAnalysis.activity_id).where(ActivityAnalysis.tss == None))
        )
    ).order_by(Activity.start_date.desc()).limit(10)
    unprocessed = session.exec(statement).all()
    
    count = 0
    for activity in unprocessed:
        try:
            await run_activity_analysis(activity.id, session, client)
            count += 1
        except Exception as e:
            print(f"Failed to analyze {activity.id}: {e}")
    
    if count > 0:
        await update_daily_training_load(session)
            
    return {"message": f"Processed {count} activities."}

@app.get("/ai/feedback/{activity_id}")
async def get_ai_feedback(activity_id: int, session: Session = Depends(get_session)):
    """Get the AI analysis for a specific activity."""
    statement = select(ActivityAnalysis).where(ActivityAnalysis.activity_id == activity_id)
    analysis = session.exec(statement).first()
    if not analysis:
        # Try to run it if it doesn't exist
        client = StravaClient(session)
        analysis = await run_activity_analysis(activity_id, session, client)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found and failed to generate.")
            
    return analysis

@app.get("/ai/fitness-trends")
async def get_fitness_trends(session: Session = Depends(get_session)):
    """Analyze overall fitness trends based on existing analyses and health data."""
    statement = select(ActivityAnalysis).order_by(ActivityAnalysis.created_at.desc()).limit(30)
    analyses = session.exec(statement).all()
    
    # Get 30-day health snapshot
    health_stmt = select(UserHealth).order_by(UserHealth.date.desc()).limit(30)
    health_records = session.exec(health_stmt).all()
    
    trends = {
        "avg_efficiency": None,
        "avg_drift": None,
        "official_ftp": None,
        "ai_ftp": None,
        "official_w_kg": None,
        "weight_delta": 0,
        "latest_weight": None,
        "fitness_level": None,
        "count": len(analyses)
    }

    def clean_nan(val):
        """Convert NaN to None for JSON compliance."""
        import math
        if val is None: return None
        try:
            if math.isnan(val): return None
        except: pass
        return float(val)

    if analyses:
        df_a = pd.DataFrame([a.model_dump() for a in analyses])
        trends["avg_efficiency"] = clean_nan(df_a['power_hr_ratio'].mean()) if 'power_hr_ratio' in df_a else None
        trends["avg_drift"] = clean_nan(df_a['hr_drift'].mean()) if 'hr_drift' in df_a else None

    if health_records:
        df_h = pd.DataFrame([h.model_dump() for h in health_records])
        latest = health_records[0]
        trends["latest_weight"] = clean_nan(latest.weight)
        trends["fitness_level"] = int(latest.fitness_level) if latest.fitness_level else None
        
        # Simple delta from 30 days ago (or oldest in batch)
        if len(health_records) > 1:
            oldest = health_records[-1]
            trends["weight_delta"] = clean_nan(latest.weight - oldest.weight)
            
    # Final calculation for display
    profile_stmt = select(AthleteProfile).order_by(AthleteProfile.updated_at.desc())
    profile = session.exec(profile_stmt).first()
    
    current_ftp = 200.0
    if profile:
        current_ftp = profile.ai_estimated_ftp or profile.ftp or 200.0
    
    trends["official_ftp"] = float(current_ftp)
    if trends["latest_weight"] and float(trends["latest_weight"]) > 0:
        trends["official_w_kg"] = float(current_ftp) / float(trends["latest_weight"])
    else:
        trends["official_w_kg"] = 0.0
    
    # Validation/Fallback for Fitness Level
    if not trends["fitness_level"] and trends["official_w_kg"] > 0:
        # Estimate VO2max from W/kg (FTP based)
        # Formula: (10.8 * (W_FTP_kg / 0.85)) + 7
        # Assumes FTP is ~85% of 5-min max power (MAP)
        est_map_wkg = trends["official_w_kg"] / 0.85
        trends["fitness_level"] = int((10.8 * est_map_wkg) + 7)

    # Recovery Score Calculation (TSB based)
    dtl_stmt = select(DailyTrainingLoad).order_by(DailyTrainingLoad.date.desc())
    latest_load = session.exec(dtl_stmt).first()
    trends["recovery_score"] = None
    if latest_load:
        # Normalize TSB to 0-100 score
        # Adjusted range to handle heavy training blocks
        # TSB -60 (Deep Fatigue) -> 0%
        # TSB +30 (Very Fresh) -> 100%
        # Range of 90pts
        tsb = latest_load.tsb
        score = int(min(100, max(0, ((tsb + 60) / 90) * 100)))
        trends["recovery_score"] = score

    return trends

@app.get("/activities/zones-yearly")
def get_yearly_zones(session: Session = Depends(get_session)):
    """Aggregate zone time (HR and Power) for all cycling activities this year."""
    import json
    from datetime import datetime
    current_year = datetime.utcnow().year
    
    statement = select(Activity, ActivityAnalysis).join(ActivityAnalysis).where(
        Activity.type == 'Ride',
        Activity.start_date >= datetime(current_year, 1, 1)
    )
    results = session.exec(statement).all()
    
    totals = {
        "hr": {"Z1": 0, "Z2": 0, "Z3": 0, "Z4": 0, "Z5": 0},
        "power": {"Z1": 0, "Z2": 0, "Z3": 0, "Z4": 0, "Z5": 0, "Z6": 0, "Z7": 0}
    }
    
    for activity, analysis in results:
        if analysis.hr_zones:
            hr_z = json.loads(analysis.hr_zones)
            for z, val in hr_z.items():
                if z in totals["hr"]: totals["hr"][z] += val
        
        if analysis.power_zones:
            p_z = json.loads(analysis.power_zones)
            for z, val in p_z.items():
                if z in totals["power"]: totals["power"][z] += val
                
    return totals

@app.get("/activities/power-stats")
async def get_all_time_power_stats(session: Session = Depends(get_session)):
    """Fetch all-time peak power metrics across all analyzed activities, with metadata fallback."""
    # 1. Start with Analyzed Peaks (Rolling Segments)
    # Include all types that might have power: Ride, VirtualRide, EBikeRide
    valid_types = ['Ride', 'VirtualRide', 'EBikeRide']
    statement = select(Activity, ActivityAnalysis).join(ActivityAnalysis).where(Activity.type.in_(valid_types))
    results = session.exec(statement).all()
    
    stats = {
        "peak_1min": {"val": 0, "date": None, "id": None},
        "peak_5min": {"val": 0, "date": None, "id": None},
        "peak_20min": {"val": 0, "date": None, "id": None},
        "peak_60min": {"val": 0, "date": None, "id": None},
        "max_avg_p": {"val": 0, "date": None, "id": None}
    }
    
    for activity, analysis in results:
        # Check rolling peak durations
        for dur in ["peak_1min", "peak_5min", "peak_20min", "peak_60min"]:
            val = getattr(analysis, dur)
            if val and val > stats[dur]["val"]:
                stats[dur] = {"val": val, "date": activity.start_date, "id": activity.strava_id}
        
        # Sustained work capacity: Highest average power for any ride >= 60 minutes
        if activity.moving_time >= 3600 and (analysis.avg_power or 0) > 0:
            if analysis.avg_power > stats["max_avg_p"]["val"]:
                stats["max_avg_p"] = {"val": analysis.avg_power, "date": activity.start_date, "id": activity.strava_id}
                
    # 2. Metadata Fallback: Check ALL Rides (even unanalyzed ones) for Highest Average Power
    all_rides_stmt = select(Activity).where(
        Activity.type.in_(valid_types),
        Activity.moving_time >= 3600,
        Activity.average_watts != None,
        Activity.id.not_in(select(ActivityAnalysis.activity_id))
    )
    all_rides = session.exec(all_rides_stmt).all()
    for ride in all_rides:
        if ride.average_watts > stats["max_avg_p"]["val"]:
            stats["max_avg_p"] = {"val": ride.average_watts, "date": ride.start_date, "id": ride.strava_id}

    return stats

@app.get("/ai/readiness")
async def get_readiness_score(session: Session = Depends(get_session)):
    """Calculate a daily readiness score (0-100) based on TSB and RHR."""
    # 1. Get latest TSB
    tsb_stmt = select(DailyTrainingLoad).order_by(DailyTrainingLoad.date.desc())
    latest_load = session.exec(tsb_stmt).first()
    
    # 2. Get RHR trends
    health_stmt = select(UserHealth).where(UserHealth.pulse != None).order_by(UserHealth.date.desc()).limit(30)
    health_records = session.exec(health_stmt).all()
    
    if not latest_load and not health_records:
        return {"score": 50, "status": "Neutral", "message": "No data yet."}
    
    score = 70.0 # Base
    message = "Feeling good. Keep training as planned."
    status = "Great"
    
    # TSB Impact (Form)
    if latest_load:
        tsb = latest_load.tsb
        if tsb < -20:
            score -= 20
            message = "Fatigue is high. Consider a recovery day."
            status = "Tired"
        elif tsb < -10:
            score -= 10
            message = "Pushing hard. Listen to your body."
            status = "Productive"
        elif tsb > 10:
            score += 5
            message = "Fresh and ready for intensity."
            status = "Fresh"
            
    # RHR Impact
    if len(health_records) >= 3:
        latest_rhr = health_records[0].pulse
        avg_rhr = sum(h.pulse for h in health_records) / len(health_records)
        
        if latest_rhr > avg_rhr * 1.1:
            score -= 15
            message = "High resting HR. Possible overtraining or illness."
            status = "Caution"
        elif latest_rhr > avg_rhr * 1.05:
            score -= 5
            
    score = max(0, min(100, score))
    
    return {
        "score": int(score),
        "status": status,
        "message": message,
        "tsb": latest_load.tsb if latest_load else None,
        "rhr": health_records[0].pulse if health_records else None
    }

@app.get("/health/latest")
def get_latest_health(session: Session = Depends(get_session)):
    """Get the most recent health metrics."""
    statement = select(UserHealth).order_by(UserHealth.date.desc())
    latest = session.exec(statement).first()
    return latest

@app.post("/strava/sync-history")
async def sync_history(session: Session = Depends(get_session)):
    client = StravaClient(session)
    try:
        await client.sync_all_history()
        return {"message": "Successfully initiated historical metadata sync."}
    except Exception as e:
        error_msg = str(e) or f"An unexpected error occurred: {type(e).__name__}"
        print(f"Sync failed: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=error_msg)
