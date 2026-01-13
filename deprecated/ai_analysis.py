import numpy as np
import json
from datetime import datetime
from typing import TypedDict, Optional, List, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from database import Activity, ActivityAnalysis, UserHealth, AthleteProfile
from sqlmodel import Session, select, and_
from config import settings

# 1. Structured Output Schema
class AIAnalysisOutput(BaseModel):
    summary: str = Field(description="A concise, high-value analysis of the training session.")
    session_type: str = Field(description="Classification: Z2 (Endurance), R3 (Tempo/SweetSpot), Intervals, Recovery, or Torque.")
    session_intent_match: bool = Field(description="Did the execution (VI, HR zones) match the session type's expected profile?")
    rpe_estimate: int = Field(description="Estimated RPE (1-10) based on intensity and duration.")
    fitness_notes: str = Field(description="Technical observations on efficiency, decoupling, or durability.")
    suggested_title: str = Field(description="A punchy, descriptive title for Strava (e.g. 'Steady Z2 Endurance - 5% Drift').")

# 2. Graph State
class AnalysisState(TypedDict):
    activity_id: int
    raw_data: dict
    streams: dict
    metrics: dict
    ai_output: Optional[AIAnalysisOutput]
    error: Optional[str]
    athlete_ftp: float
    session: object
    strava_client: object

# 3. Calculation Utilities
def calculate_np(watts: List[float]) -> float:
    if not watts: return 0.0
    # 30s rolling average
    kernel_size = 30
    w = np.array(watts)
    if len(w) < kernel_size: return np.mean(w)
    
    rolling_avg = np.convolve(w, np.ones(kernel_size)/kernel_size, mode='valid')
    # Fourth power, mean, then fourth root
    return float(np.mean(rolling_avg**4)**0.25)

def calculate_hr_drift(watts: List[float], hr: List[float]) -> Optional[float]:
    """Calculates Aerobic Decoupling (drift) by comparing P/HR of 1st half vs 2nd half."""
    if not watts or not hr or len(watts) != len(hr): return None
    
    # Filter for moving/non-zero power
    w = np.array(watts)
    h = np.array(hr)
    mask = w > 10 # Ignore coasting
    w = w[mask]
    h = h[mask]
    
    if len(w) < 600: return None # Need at least 10min of active data
    
    mid = len(w) // 2
    ef1 = np.mean(w[:mid]) / np.mean(h[:mid]) if np.mean(h[:mid]) > 0 else 0
    ef2 = np.mean(w[mid:]) / np.mean(h[mid:]) if np.mean(h[mid:]) > 0 else 0
    
    if ef1 == 0: return None
    drift = ((ef1 - ef2) / ef1) * 100
    return float(drift)

def calculate_power_fade(watts: List[float]) -> Optional[float]:
    """Power last 20% vs first 20%."""
    if not watts or len(watts) < 500: return None
    w = np.array(watts)
    part = len(w) // 5
    p1 = np.mean(w[:part])
    p2 = np.mean(w[-part:])
    if p1 == 0: return 0.0
    return float(((p1 - p2) / p1) * 100)

def get_best_power(watts: List[float], seconds: int) -> float:
    if not watts or len(watts) < seconds: return 0.0
    w = np.array(watts)
    rolling_sum = np.convolve(w, np.ones(seconds), mode='valid')
    return float(np.max(rolling_sum) / seconds)

def calculate_zone_distribution(data: List[float], zones: List[float]) -> dict:
    """Calculates time spent in each zone (seconds). zones is a list of upper bounds."""
    if not data: return {}
    counts = {}
    d = np.array(data)
    for i, upper in enumerate(zones):
        lower = zones[i-1] if i > 0 else 0
        counts[f"Z{i+1}"] = int(np.sum((d >= lower) & (d < upper)))
    # Add final zone
    counts[f"Z{len(zones)+1}"] = int(np.sum(d >= zones[-1]))
    return counts

# 4. Graph Nodes
def extract_metrics_node(state: AnalysisState) -> AnalysisState:
    streams = state["streams"]
    activity = state["raw_data"]
    session = state["session"]
    
    watts = streams.get("watts", {}).get("data", [])
    hr = streams.get("heartrate", {}).get("data", [])
    cadence = streams.get("cadence", {}).get("data", [])
    time_stream = streams.get("time", {}).get("data", [])
    
    # Interpolate to 1Hz grid for accurate convolve-based peaks
    if time_stream and watts and len(time_stream) > 1:
        total_sec = int(max(time_stream))
        grid = np.arange(0, total_sec + 1)
        watts = np.interp(grid, time_stream, watts).tolist()
        if hr:
            hr = np.interp(grid, time_stream, hr).tolist()
        if cadence:
            cadence = np.interp(grid, time_stream, cadence).tolist()

    # 1. Fetch Athlete Profile for synced FTP and Max HR
    profile_stmt = select(AthleteProfile).limit(1)
    profile = session.exec(profile_stmt).first()
    
    ftp = profile.ftp if profile and profile.ftp else (state.get("athlete_ftp") or 200)
    official_max_hr = profile.max_hr if profile and profile.max_hr else (activity.get("max_heartrate") or 190)
    
    # 2. Extract Power Metrics
    # STRICT MODE: If we are analyzing, we prefer stream data. 
    # If watts stream is missing/empty, we assume NO VALID POWER DATA, even if metadata has average_watts.
    # This filters out manual entries or corrupt uploads with "ghost" power values.
    avg_p = np.mean(watts) if watts else 0.0
    
    np_val = calculate_np(watts)
    
    # Calculate absolute peak (1-sec max) from stream
    peak_1s = float(np.max(watts)) if watts else 0.0
    
    # Update back to activity object if max_watts is empty
    if peak_1s > 0 and not activity.get("max_watts"):
        db_activity = session.get(Activity, state["activity_id"])
        if db_activity:
            db_activity.max_watts = peak_1s
            session.add(db_activity)
            session.commit()
            session.refresh(db_activity)

    metrics = {
        "duration": activity.get("moving_time"),
        "distance": activity.get("distance"),
        "avg_power": avg_p,
        "normalized_power": np_val,
        "avg_hr": activity.get("average_heartrate"),
        "max_hr": activity.get("max_heartrate") or official_max_hr,
        "avg_cadence": np.mean(cadence) if cadence else None,
        "elevation_gain": activity.get("total_elevation_gain"),
        "power_hr_ratio": (avg_p / activity.get("average_heartrate")) if (avg_p and activity.get("average_heartrate")) else None,
        "hr_drift": calculate_hr_drift(watts, hr),
        "power_fade": calculate_power_fade(watts),
        "variability_index": (np_val / avg_p) if (np_val and avg_p) else None,
        "peak_5s": get_best_power(watts, 5),
        "peak_30s": get_best_power(watts, 30),
        "peak_1min": get_best_power(watts, 60),
        "peak_5min": get_best_power(watts, 300),
        "peak_10min": get_best_power(watts, 600),
        "peak_20min": get_best_power(watts, 1200),
        "peak_60min": get_best_power(watts, 3600),
    }
    
    # Simple FTP estimate (95% of peak 20min)
    if metrics["peak_20min"]:
        metrics["ftp_estimate"] = metrics["peak_20min"] * 0.95
        
    # Relative Power Metrics (W/kg)
    weight = activity.get("current_weight")
    if weight and weight > 0:
        if metrics["avg_power"]:
            metrics["avg_w_kg"] = metrics["avg_power"] / weight
        if metrics.get("ftp_estimate"):
            metrics["ftp_w_kg"] = metrics["ftp_estimate"] / weight
            
    # TSS Calculation (Power vs HR fallback)
    ftp = state.get("athlete_ftp") or 200
    duration_sec = metrics["duration"]
    
    if metrics["normalized_power"] and metrics["normalized_power"] > 0:
        # Power-based TSS
        np_val = metrics["normalized_power"]
        metrics["tss"] = (duration_sec * (np_val ** 2)) / ((ftp ** 2) * 3600) * 100
    elif metrics["avg_hr"] and metrics["avg_hr"] > 0:
        # HR-based fallback (hrSS / TRIMP inspired)
        # Simple approximation: (duration * intensity^2 * 100) / 3600
        # Intensity = (avg_hr - rhr) / (max_hr - rhr) OR simpler: avg_hr / max_hr
        max_hr = activity.get("max_heartrate") or 190
        intensity = metrics["avg_hr"] / max_hr
        # Multiply by a factor to align roughly with power TSS scales
        metrics["tss"] = (duration_sec / 3600) * (intensity ** 2) * 80 
    else:
        # If no data at all, set to 0.0 to prevent re-processing loop
        metrics["tss"] = 0.0
    
    # Zone Calculations
    ftp = state.get("athlete_ftp") or 200
    max_hr = activity.get("max_heartrate") or 190
    
    # Power Zones (Coggan): <55%, 75%, 90%, 105%, 120%, 150%
    p_zones = [ftp * 0.55, ftp * 0.75, ftp * 0.90, ftp * 1.05, ftp * 1.20, ftp * 1.50]
    metrics["power_zones"] = calculate_zone_distribution(watts, p_zones) if watts else {}
    
    # HR Zones: 60%, 70%, 80%, 90%
    hr_zones = [max_hr * 0.60, max_hr * 0.70, max_hr * 0.80, max_hr * 0.90]
    metrics["hr_zones"] = calculate_zone_distribution(hr, hr_zones) if hr else {}
    
    state["metrics"] = metrics
    return state

async def llm_analysis_node(state: AnalysisState) -> AnalysisState:
    if not settings.openai_api_key:
        state["error"] = "OpenAI API key missing"
        return state
    
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=settings.openai_api_key)
    structured_llm = llm.with_structured_output(AIAnalysisOutput)
    
    pacing_vi = f"{state['metrics']['variability_index']:.2f}" if state['metrics'].get('variability_index') else "N/A"
    
    prompt = f"""
    Analyze the following cycling/training session metrics and provide expert coaching feedback.
    
    Session: {state['raw_data'].get('name', 'Unknown')}
    Type: {state['raw_data'].get('type', 'Unknown')}
    Duration: {state['metrics']['duration']/60:.1f} min
    Avg Power: {state['metrics']['avg_power']:.1f}W
    Avg HR: {state['metrics'].get('avg_hr', 'N/A')} bpm
    Max HR: {state['metrics'].get('max_hr', 'N/A')} bpm
    Normalized Power: {state['metrics']['normalized_power']:.1f}W
    Variability Index: {pacing_vi}
    HR Drift: {state['metrics'].get('hr_drift', 'N/A')}%
    Power Fade: {state['metrics'].get('power_fade', 'N/A')}%
    Best 20min Power: {state['metrics'].get('best_20min_power', 'N/A')}W
    
    Focus on:
    1. Aerobic efficiency (P/HR) and decoupling (Drift).
    2. Pacing (VI).
    3. Session classification (Z2 is steady, VI < 1.05. Intervals have high VI).
    4. Weight-adjusted performance (W/kg) if weight data is present.
    5. Signs of fatigue or staleness.
    6. Suggested Strava title: Short, meaningful, and professional.
    """
    
    try:
        output = await structured_llm.ainvoke([
            SystemMessage(content="You are an elite cycling coach and exercise physiologist."),
            HumanMessage(content=prompt)
        ])
        state["ai_output"] = output
    except Exception as e:
        state["error"] = str(e)
        
    return state

# 5. Build Graph
workflow = StateGraph(AnalysisState)
workflow.add_node("extract_metrics", extract_metrics_node)
workflow.add_node("llm_analysis", llm_analysis_node)

workflow.set_entry_point("extract_metrics")
workflow.add_edge("extract_metrics", "llm_analysis")
workflow.add_edge("llm_analysis", END)

app_graph = workflow.compile()

# 6. Runner Function
async def run_activity_analysis(activity_id_db: int, session: Session, strava_client):
    statement = select(Activity).where(Activity.id == activity_id_db)
    activity = session.exec(statement).one()
    
    # Check if analysis already exists and is complete
    stmt_analysis = select(ActivityAnalysis).where(ActivityAnalysis.activity_id == activity.id)
    existing = session.exec(stmt_analysis).first()
    if existing and existing.tss is not None:
        return existing
    
    # If it exists but is incomplete, we'll delete the old one and regenerate
    if existing:
        session.delete(existing)
        session.commit()
    
    streams = await strava_client.get_activity_streams(activity.strava_id)
    
    # Get Athlete FTP
    from database import AthleteProfile
    profile_stmt = select(AthleteProfile).order_by(AthleteProfile.updated_at.desc())
    profile = session.exec(profile_stmt).first()
    athlete_ftp = profile.ftp if (profile and profile.ftp) else 200 # Fallback to 200W if missing

    # Fetch closest health data
    stmt_health = select(UserHealth).where(
        UserHealth.date <= activity.start_date
    ).order_by(UserHealth.date.desc())
    health = session.exec(stmt_health).first()

    initial_state = {
        "activity_id": activity.id,
        "raw_data": activity.model_dump(),
        "streams": streams,
        "metrics": {},
        "ai_output": None,
        "error": None,
        "athlete_ftp": athlete_ftp,
        "session": session,
        "strava_client": strava_client
    }
    
    # Inject health context into raw_data for LLM
    if health:
        initial_state["raw_data"]["current_weight"] = health.weight
        initial_state["raw_data"]["current_fat_ratio"] = health.fat_ratio
        
    result = await app_graph.ainvoke(initial_state)
    
    if result.get("error"):
        print(f"Error analyzing activity {activity.id}: {result['error']}")
        return None
        
    ai: AIAnalysisOutput = result["ai_output"]
    m = result["metrics"]
    
    analysis = ActivityAnalysis(
        activity_id=activity.id,
        summary=ai.summary,
        session_type=ai.session_type,
        session_intent_match=ai.session_intent_match,
        suggested_title=ai.suggested_title,
        duration=m["duration"],
        distance=m["distance"],
        avg_power=m["avg_power"],
        normalized_power=m["normalized_power"],
        avg_hr=m["avg_hr"],
        max_hr=m["max_hr"],
        avg_cadence=m["avg_cadence"],
        elevation_gain=m["elevation_gain"],
        power_hr_ratio=m["power_hr_ratio"],
        hr_drift=m["hr_drift"],
        power_fade=m["power_fade"],
        variability_index=m["variability_index"],
        peak_5s=m["peak_5s"],
        peak_30s=m["peak_30s"],
        peak_1min=m["peak_1min"],
        peak_5min=m["peak_5min"],
        peak_10min=m["peak_10min"],
        peak_20min=m["peak_20min"],
        peak_60min=m["peak_60min"],
        ftp_estimate=m.get("ftp_estimate"),
        avg_w_kg=m.get("avg_w_kg"),
        ftp_w_kg=m.get("ftp_w_kg"),
        tss=m.get("tss"),
        rpe=ai.rpe_estimate,
        notes=ai.fitness_notes,
        hr_zones=json.dumps(m.get("hr_zones", {})),
        power_zones=json.dumps(m.get("power_zones", {}))
    )
    
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
    
    # Update Strava Title
    if ai.suggested_title:
        try:
            await strava_client.update_activity(activity.strava_id, ai.suggested_title)
            print(f"Updated Strava Title for {activity.strava_id}: {ai.suggested_title}")
        except Exception as e:
            print(f"Failed to update Strava title: {e}")
            
    # Update AI Estimated FTP in Profile
    if analysis.peak_20min and analysis.peak_20min > 0:
        est = analysis.peak_20min * 0.95
        # Update profile with the new estimate if it's "better" or just keep it current
        # For now, let's keep the best estimate of the last 30 days or simply update to the latest valid one
        profile_stmt = select(AthleteProfile).order_by(AthleteProfile.updated_at.desc())
        prof = session.exec(profile_stmt).first()
        if prof:
            # We only overwrite if it's a "meaningful" effort, or if we want it to be a rolling average
            # Let's just track the highest estimate for now as a "Peak AI FTP"
            if not prof.ai_estimated_ftp or est > prof.ai_estimated_ftp:
                prof.ai_estimated_ftp = est
                prof.updated_at = datetime.utcnow()
                session.add(prof)
                session.commit()

    return analysis

async def update_daily_training_load(session: Session):
    """Recalculate CTL, ATL, TSB for all days based on ActivityAnalysis TSS."""
    # 1. Get all activities with TSS, sorted by date
    from database import Activity, ActivityAnalysis, DailyTrainingLoad
    
    statement = select(Activity, ActivityAnalysis).join(ActivityAnalysis).order_by(Activity.start_date)
    results = session.exec(statement).all()
    
    if not results:
        return
    
    # Group TSS by day
    daily_tss = {}
    for activity, analysis in results:
        day = activity.start_date.date()
        daily_tss[day] = daily_tss.get(day, 0) + (analysis.tss or 0)
        
    # Get date range
    start_date = min(daily_tss.keys())
    end_date = datetime.utcnow().date()
    
    # Rolling metrics
    ctl = 0.0
    atl = 0.0
    ctl_prev = 0.0
    atl_prev = 0.0
    
    import datetime as dt
    current_date = start_date
    
    while current_date <= end_date:
        tss = daily_tss.get(current_date, 0)
        
        # Form (TSB) is usually CTL_yesterday - ATL_yesterday
        tsb = ctl_prev - atl_prev
        
        ctl = ctl + (tss - ctl) / 42
        atl = atl + (tss - atl) / 7
        
        # Update DB
        dt_val = dt.datetime.combine(current_date, dt.time.min)
        stmt = select(DailyTrainingLoad).where(DailyTrainingLoad.date == dt_val)
        record = session.exec(stmt).first()
        
        if not record:
            record = DailyTrainingLoad(date=dt_val, ctl=ctl, atl=atl, tsb=tsb, total_tss=tss)
            session.add(record)
        else:
            record.ctl = ctl
            record.atl = atl
            record.tsb = tsb
            record.total_tss = tss
            
        ctl_prev = ctl
        atl_prev = atl
        current_date += dt.timedelta(days=1)
        
    session.commit()
