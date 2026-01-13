import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select, and_
from app.core.database import get_session
from app.models import Activity, ActivityAnalysis, UserHealth, DailyTrainingLoad
from app.services.analysis import run_activity_analysis, update_daily_training_load

router = APIRouter(tags=["ai"])

def clean_nan(val):
    """Convert NaN to None for JSON compliance."""
    if isinstance(val, float) and np.isnan(val):
        return None
    return val

@router.post("/ai/analyze-all")
async def analyze_all_activities(session: Session = Depends(get_session)):
    """Trigger AI analysis for all activities that haven't been processed yet."""
    # Get all activities without an analysis
    from app.services.strava import StravaClient
    strava = StravaClient(session)
    
    statement = select(Activity).where(
        ~select(ActivityAnalysis).where(ActivityAnalysis.activity_id == Activity.id).exists()
    )
    activities = session.exec(statement).all()
    
    analyzed_count = 0
    for activity in activities:
        try:
            await run_activity_analysis(activity.id, session, strava)
            analyzed_count += 1
        except Exception as e:
            print(f"Error analyzing activity {activity.id}: {e}")
            continue
            
    # After analyzing new activities, update the training load trends
    if analyzed_count > 0:
        update_daily_training_load(session)
        
    return {"status": "success", "analyzed_count": analyzed_count}

@router.get("/ai/feedback/{activity_id}")
async def get_ai_feedback(activity_id: int, session: Session = Depends(get_session)):
    """Get the AI analysis for a specific activity."""
    from app.services.strava import StravaClient
    strava = StravaClient(session)
    
    statement = select(ActivityAnalysis).where(ActivityAnalysis.activity_id == activity_id)
    analysis = session.exec(statement).first()
    
    if not analysis:
        # Try to run it now
        try:
            analysis = await run_activity_analysis(activity_id, session, strava)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
            
    return analysis

@router.get("/ai/fitness-trends")
async def get_fitness_trends(session: Session = Depends(get_session)):
    """Analyze overall fitness trends based on existing analyses and health data."""
    # Weight Trend
    health_stmt = select(UserHealth).where(UserHealth.weight != None).order_by(UserHealth.date.asc())
    health_data = session.exec(health_stmt).all()
    
    # Fitness Level Trend (VO2 Max approx)
    fitness_stmt = select(UserHealth).where(UserHealth.fitness_level != None).order_by(UserHealth.date.asc())
    fitness_data = session.exec(fitness_stmt).all()
    
    # Training Load Trend (CTL/ATL/TSB)
    load_stmt = select(DailyTrainingLoad).order_by(DailyTrainingLoad.date.asc())
    load_data = session.exec(load_stmt).all()
    
    # Efficiency Factor Trend (P/HR)
    # Get latest 30-day riding efficiency
    thirty_days_ago = pd.Timestamp.utcnow() - pd.Timedelta(days=30)
    eff_stmt = select(Activity, ActivityAnalysis).join(ActivityAnalysis).where(
        and_(Activity.start_date >= thirty_days_ago, Activity.type == 'Ride')
    ).order_by(Activity.start_date.asc())
    eff_results = session.exec(eff_stmt).all()
    
    trends = {
        "weight": [clean_nan(h.weight) for h in health_data[-30:]],
        "weight_dates": [h.date.strftime('%Y-%m-%d') for h in health_data[-30:]],
        "fitness_level": clean_nan(fitness_data[-1].fitness_level) if fitness_data else None,
        "ctl": [clean_nan(l.ctl) for l in load_data[-30:]],
        "atl": [clean_nan(l.atl) for l in load_data[-30:]],
        "tsb": [clean_nan(l.tsb) for l in load_data[-30:]],
        "load_dates": [l.date.strftime('%Y-%m-%d') for l in load_data[-30:]],
        "efficiency": [clean_nan(a.power_hr_ratio) for _, a in eff_results],
        "efficiency_dates": [act.start_date.strftime('%Y-%m-%d') for act, _ in eff_results]
    }
    
    # HR Drift Calculation (latest 30 days)
    drift_values = [a.hr_drift for _, a in eff_results if a.hr_drift is not None]
    trends["avg_drift"] = clean_nan(sum(drift_values) / len(drift_values)) if drift_values else None

    # Official FTP and W/kg from latest analysis
    latest_analysis = session.exec(select(ActivityAnalysis).order_by(ActivityAnalysis.created_at.desc())).first()
    trends["official_ftp"] = clean_nan(latest_analysis.ftp_estimate) if latest_analysis else 200
    trends["official_w_kg"] = clean_nan(latest_analysis.ftp_w_kg) if latest_analysis else 0

    # Fallback for fitness_level if Withings is empty
    if trends["fitness_level"] is None:
        if latest_analysis and latest_analysis.ftp_w_kg:
             # Very rough VO2 Max estimation
             trends["fitness_level"] = int(latest_analysis.ftp_w_kg * 7 + 10)

    # Recovery Score Calculation (TSB based)
    dtl_stmt = select(DailyTrainingLoad).order_by(DailyTrainingLoad.date.desc())
    latest_load = session.exec(dtl_stmt).first()
    trends["recovery_score"] = None
    if latest_load:
        # Normalize TSB to 0-100 score
        # TSB -60 (Deep Fatigue) -> 0%
        # TSB +30 (Very Fresh) -> 100%
        tsb = latest_load.tsb
        score = int(min(100, max(0, ((tsb + 60) / 90) * 100)))
        trends["recovery_score"] = score
        
    return trends

@router.get("/activities/zones-yearly")
def get_yearly_zones(session: Session = Depends(get_session)):
    """Aggregate zone time (HR and Power) for all cycling activities this year."""
    import json
    current_year = datetime.utcnow().year
    start_of_year = datetime(current_year, 1, 1)
    
    stmt = select(ActivityAnalysis).join(Activity).where(
        and_(Activity.start_date >= start_of_year, Activity.type == 'Ride')
    )
    results = session.exec(stmt).all()
    
    hr_totals = {}
    pwr_totals = {}
    
    for analysis in results:
        if analysis.hr_zones:
            zones = json.loads(analysis.hr_zones)
            for z, val in zones.items():
                hr_totals[z] = hr_totals.get(z, 0) + val
        if analysis.power_zones:
            zones = json.loads(analysis.power_zones)
            for z, val in zones.items():
                pwr_totals[z] = pwr_totals.get(z, 0) + val
                
    return {"hr": hr_totals, "power": pwr_totals}

@router.get("/activities/power-stats")
def get_all_time_power_stats(session: Session = Depends(get_session)):
    """Fetch all-time peak power metrics across all analyzed activities, with metadata fallback."""
    # Find max of each peak duration in ActivityAnalysis
    durations = ['peak_1min', 'peak_5min', 'peak_20min', 'peak_60min']
    stats = {}
    
    for dur in durations:
        stmt = select(Activity, ActivityAnalysis).join(ActivityAnalysis).where(
            getattr(ActivityAnalysis, dur) != None
        ).order_by(getattr(ActivityAnalysis, dur).desc())
        result = session.exec(stmt).first()
        if result:
            act, ana = result
            stats[dur] = {
                "val": getattr(ana, dur),
                "date": act.start_date.strftime('%Y-%m-%d'),
                "id": act.strava_id or act.id
            }
        else:
            stats[dur] = {"val": 0, "date": None, "id": None}
            
    # Calculate Highest Avg Power ( Rides > 60min )
    valid_types = ['Ride', 'VirtualRide', 'EBikeRide']
    stmt_avg = select(Activity, ActivityAnalysis).join(ActivityAnalysis).where(
        and_(
            Activity.type.in_(valid_types),
            Activity.moving_time >= 3600,
            ActivityAnalysis.avg_power != None
        )
    ).order_by(ActivityAnalysis.avg_power.desc())
    
    avg_result = session.exec(stmt_avg).first()
    if avg_result:
        act, ana = avg_result
        stats["max_avg_p"] = {
            "val": ana.avg_power,
            "date": act.start_date.strftime('%Y-%m-%d'),
            "id": act.strava_id or act.id
        }
    else:
        # Metadata fallback for max_avg_p
        meta_stmt = select(Activity).where(
            and_(
                Activity.type.in_(valid_types),
                Activity.moving_time >= 3600,
                Activity.average_watts != None
            )
        ).order_by(Activity.average_watts.desc())
        meta_result = session.exec(meta_stmt).first()
        if meta_result:
             stats["max_avg_p"] = {
                "val": meta_result.average_watts,
                "date": meta_result.start_date.strftime('%Y-%m-%d'),
                "id": meta_result.strava_id or meta_result.id
            }
        else:
            stats["max_avg_p"] = {"val": 0, "date": None, "id": None}
            
    return stats
