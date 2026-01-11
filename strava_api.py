import time
from datetime import datetime
import httpx
from typing import List, Optional
from database import StravaToken, Activity, ActivityAnalysis
from sqlmodel import Session, select, or_
from config import settings

class StravaClient:

    class StravaReauthRequired(Exception):
        """Raised when a 401 Unauthorized error occurs, indicating re-authorization is needed."""
        pass

    def __init__(self, session: Session):
        self.session = session
        self.base_url = "https://www.strava.com/api/v3"

    def has_token(self) -> bool:
        statement = select(StravaToken).order_by(StravaToken.id.desc())
        token = self.session.exec(statement).first()
        return token is not None

    def get_auth_url(self) -> str:
        return (
            f"https://www.strava.com/oauth/authorize?"
            f"client_id={settings.strava_client_id}&"
            f"response_type=code&"
            f"redirect_uri={settings.strava_redirect_uri}&"
            f"scope=read,activity:read_all,profile:read_all&"
            f"approval_prompt=force"
        )

    async def exchange_token(self, code: str) -> StravaToken:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://www.strava.com/oauth/token",
                data={
                    "client_id": settings.strava_client_id,
                    "client_secret": settings.strava_client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                },
            )
            response.raise_for_status()
            data = response.json()
            
            token = StravaToken(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                expires_at=data["expires_at"],
                athlete_id=data["athlete"]["id"]
            )
            self.session.add(token)
            self.session.commit()
            self.session.refresh(token)
            return token

    async def get_valid_token(self) -> str:
        statement = select(StravaToken).order_by(StravaToken.id.desc())
        token = self.session.exec(statement).first()
        
        if not token:
            raise Exception("No Strava token found. Please authenticate.")

        if token.expires_at < time.time():
            # Refresh token
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://www.strava.com/oauth/token",
                    data={
                        "client_id": settings.strava_client_id,
                        "client_secret": settings.strava_client_secret,
                        "refresh_token": token.refresh_token,
                        "grant_type": "refresh_token",
                    },
                )
                response.raise_for_status()
                data = response.json()
                
                token.access_token = data["access_token"]
                token.refresh_token = data["refresh_token"]
                token.expires_at = data["expires_at"]
                self.session.add(token)
                self.session.commit()
                self.session.refresh(token)
        
        return token.access_token

    async def fetch_activities(self, after: Optional[int] = None, before: Optional[int] = None) -> List[dict]:
        access_token = await self.get_valid_token()
        activities = []
        page = 1
        per_page = 200
        
        async with httpx.AsyncClient() as client:
            while True:
                params = {"page": page, "per_page": per_page}
                if after:
                    params["after"] = after
                if before:
                    params["before"] = before
                
                print(f"Fetching activities page {page}... (per_page={per_page})")
                try:
                    response = await client.get(
                        f"{self.base_url}/athlete/activities",
                        headers={"Authorization": f"Bearer {access_token}"},
                        params=params
                    )
                    if response.status_code != 200:
                        print(f"Strava API Error: {response.status_code} - {response.text}")
                    response.raise_for_status()
                    data = response.json()
                    
                    if not data:
                        print("No more data from Strava.")
                        break
                    
                    print(f"Received {len(data)} activities.")
                    activities.extend(data)
                    if len(data) < per_page:
                        print("Reached end of history.")
                        break
                    page += 1
                except httpx.HTTPStatusError as e:
                    print(f"HTTP Error fetching activities: {e.response.status_code} - {e.response.text}")
                    raise Exception(f"Strava API error on page {page}: {e.response.text or e}")
                except Exception as e:
                    print(f"Unexpected error on page {page}: {e}")
                    raise
                
        return activities

    async def get_athlete(self) -> dict:
        access_token = await self.get_valid_token()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/athlete",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status()
            return response.json()

    def sync_athlete(self, athlete_data: dict):
        from database import AthleteProfile
        athlete_id = athlete_data["id"]
        statement = select(AthleteProfile).where(AthleteProfile.athlete_id == athlete_id)
        profile = self.session.exec(statement).first()
        
        if not profile:
            profile = AthleteProfile(athlete_id=athlete_id)
            self.session.add(profile)
        
        profile.ftp = athlete_data.get("ftp")
        profile.updated_at = datetime.utcnow()
        self.session.commit()
        self.session.refresh(profile)
        return profile

    def get_latest_activity_timestamp(self) -> Optional[int]:
        statement = select(Activity).order_by(Activity.start_date.desc())
        latest = self.session.exec(statement).first()
        if latest:
            return int(latest.start_date.timestamp())
        return None

    async def sync_athlete_zones(self):
        """Fetch athlete zones (FTP and Max HR) and update the AthleteProfile."""
        access_token = await self.get_valid_token()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/athlete/zones",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            if response.status_code == 401:
                raise self.StravaReauthRequired("Re-authorization required for zones scope.")
            response.raise_for_status()
            zones = response.json()
            
            # Extract FTP and Max HR
            # Heart Rate Zones structure: { "heart_rate": { "custom_zones": bool, "zones": [...] }, ... }
            # Power Zones structure: { "power": { "zones": [...] }, ... }
            hr_data = zones.get("heart_rate", {})
            hr_zones = hr_data.get("zones", [])
            # Usually the top of the last zone (Z5) is the max HR
            max_hr = hr_zones[-1].get("max") if hr_zones else None
            
            # Power zones are usually based on FTP. Strava doesn't always expose "FTP" directly in this endpoint
            # but the zones are derived from it. Usually, Z4 max or Z5 min is near FTP.
            # However, Strava usually provides the FTP in the Athlete profile or we can infer it.
            # Let's check if there's a more direct way or just take the Z4-Z5 boundary.
            pwr_data = zones.get("power", {})
            pwr_zones = pwr_data.get("zones", [])
            # Z4 is typically 91-105% of FTP. So Z4 max / 1.05 is FTP.
            ftp = None
            if len(pwr_zones) >= 4:
                 ftp = pwr_zones[3].get("max") / 1.05 # Approximate FTP from Z4 max
            
            # Update DB
            from database import AthleteProfile
            # Get athlete ID from token
            stmt = select(StravaToken).limit(1)
            token_rec = self.session.exec(stmt).first()
            if not token_rec: return
            
            profile_stmt = select(AthleteProfile).where(AthleteProfile.athlete_id == token_rec.athlete_id)
            profile = self.session.exec(profile_stmt).first()
            
            if not profile:
                profile = AthleteProfile(athlete_id=token_rec.athlete_id)
                self.session.add(profile)
            
            if ftp: profile.ftp = ftp
            if max_hr and max_hr > 0: profile.max_hr = max_hr
            profile.updated_at = datetime.utcnow()
            
            self.session.commit()
            print(f"Synced Athlete Profile: FTP={profile.ftp}, MaxHR={profile.max_hr}")

    def sync_activities(self, strava_activities: List[dict]):
        print(f"Syncing {len(strava_activities)} activities to DB...")
        added_count = 0
        updated_count = 0
        
        for activity_data in strava_activities:
            # Check if exists
            statement = select(Activity).where(Activity.strava_id == activity_data["id"])
            existing = self.session.exec(statement).first()
            
            if not existing:
                activity = Activity(
                    strava_id=activity_data["id"],
                    name=activity_data["name"],
                    distance=activity_data["distance"],
                    moving_time=activity_data["moving_time"],
                    elapsed_time=activity_data["elapsed_time"],
                    total_elevation_gain=activity_data["total_elevation_gain"],
                    type=activity_data["type"],
                    sport_type=activity_data["sport_type"],
                    start_date=datetime.fromisoformat(activity_data["start_date"].replace("Z", "+00:00")),
                    start_date_local=datetime.fromisoformat(activity_data["start_date_local"].replace("Z", "+00:00")),
                    timezone=activity_data["timezone"],
                    utc_offset=activity_data["utc_offset"],
                    average_speed=activity_data["average_speed"],
                    max_speed=activity_data["max_speed"],
                    has_heartrate=activity_data["has_heartrate"],
                    average_heartrate=activity_data.get("average_heartrate"),
                    max_heartrate=activity_data.get("max_heartrate"),
                    average_watts=activity_data.get("average_watts"),
                    max_watts=activity_data.get("max_watts"),
                )
                self.session.add(activity)
                added_count += 1
            else:
                # Backfill missing metadata for existing activities
                updated = False
                fields_to_backfill = ["max_watts", "average_watts", "average_heartrate", "max_heartrate", "total_elevation_gain"]
                for field in fields_to_backfill:
                    if getattr(existing, field) is None and activity_data.get(field):
                        setattr(existing, field, activity_data[field])
                        updated = True
                
                if updated:
                    self.session.add(existing)
                    updated_count += 1
        
        self.session.commit()
        print(f"Sync complete: {added_count} added, {updated_count} updated.")

    async def sync_all_history(self):
        """Fetch all historical activities and update missing metadata (backfill) page by page."""
        print("Starting deep historical sync (present backwards to earliest)...")
        
        # We'll page backwards from now
        before = int(time.time())
        per_page = 200
        total_fetched = 0
        
        while True:
            # fetch_activities normally pages internally, but we'll use its 'before' param
            # to do our own outer loop so we sync to DB between pages.
            # To do this, we need to pass page=1 each time and just move the 'before' pointer.
            page_data = await self.fetch_single_page(before=before, per_page=per_page)
            
            if not page_data:
                break
                
            print(f"Fetched {len(page_data)} activities (Total: {total_fetched + len(page_data)}). Syncing to DB...")
            self.sync_activities(page_data)
            total_fetched += len(page_data)
            
            # Update 'before' to the earliest activity in this batch
            earliest_ts = min(
                int(datetime.fromisoformat(a["start_date"].replace("Z", "+00:00")).timestamp())
                for a in page_data
            )
            before = earliest_ts - 1
            
            if len(page_data) < per_page:
                print("Reached end of history.")
                break
                
        print(f"Historical metadata sync complete. Total processed: {total_fetched}. Starting targeted power analysis...")
        
        # 2. Identify and analyze candidates for power records
        await self.analyze_power_candidates()
        print("Historical power analysis complete.")

    async def fetch_single_page(self, page: int = 1, per_page: int = 200, before: Optional[int] = None) -> List[dict]:
        """Fetch a single page of activities for incremental syncing."""
        access_token = await self.get_valid_token()
        params = {"page": page, "per_page": per_page}
        if before:
            params["before"] = before
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.base_url}/athlete/activities",
                    headers={"Authorization": f"Bearer {access_token}"},
                    params=params
                )
                if response.status_code != 200:
                    print(f"Strava API Error: {response.status_code} - {response.text}")
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
                raise Exception(f"Strava API error: {e.response.text or e}")

    async def get_power_stats_internal(self) -> dict:
        """Internal helper to get power bests without importing from main.py."""
        valid_types = ['Ride', 'VirtualRide', 'EBikeRide', 'Rowing', 'Workout']
        
        # 1. Analyzed Peaks
        statement = select(Activity, ActivityAnalysis).join(ActivityAnalysis).where(Activity.type.in_(valid_types))
        results = self.session.exec(statement).all()
        
        stats = {
            "peak_1min": {"val": 0, "date": None},
            "peak_5min": {"val": 0, "date": None},
            "peak_20min": {"val": 0, "date": None},
            "peak_60min": {"val": 0, "date": None},
            "max_avg_p": {"val": 0, "date": None}
        }
        
        for activity, analysis in results:
            for dur in ["peak_1min", "peak_5min", "peak_20min", "peak_60min"]:
                val = getattr(analysis, dur)
                if val and val > stats[dur]["val"]:
                    stats[dur] = {"val": val, "date": activity.start_date}
            
            if activity.moving_time >= 3600 and (analysis.avg_power or 0) > stats["max_avg_p"]["val"]:
                stats["max_avg_p"] = {"val": analysis.avg_power, "date": activity.start_date}
        
        # 2. Metadata Fallback
        # Only consider activities that have NOT been analyzed yet.
        # If analyzed, we trust the analysis (even if 0) over metadata.
        all_rides_stmt = select(Activity).where(
            Activity.type.in_(valid_types),
            Activity.moving_time >= 3600,
            Activity.average_watts != None,
            Activity.id.not_in(select(ActivityAnalysis.activity_id))
        )
        all_rides = self.session.exec(all_rides_stmt).all()
        for ride in all_rides:
            if ride.average_watts > stats["max_avg_p"]["val"]:
                stats["max_avg_p"] = {"val": ride.average_watts, "date": ride.start_date}
                
        return stats

    async def analyze_power_candidates(self):
        """Identify historical activities that are likely to break records and analyze them."""
        # Get current bests from existing analyses
        current_bests = await self.get_power_stats_internal()
        
        # Heuristics for candidates:
        # 1. Max Watts > current 1min peak
        # 2. Avg Watts > current 20min/60min peaks * 0.8 (buffer for variability)
        
        p1 = current_bests["peak_1min"]["val"]
        p20 = current_bests["peak_20min"]["val"]
        p60 = current_bests["peak_60min"]["val"]
        
        print(f"Current Records (from stats): 1m={p1:.0f}W, 20m={p20:.0f}W, 60m={p60:.0f}W")
        
        # Find candidates: Rides with power that HAVEN'T been analyzed yet
        valid_types = ['Ride', 'VirtualRide', 'EBikeRide', 'Rowing', 'Workout']
        stmt = select(Activity).where(
            Activity.type.in_(valid_types),
            Activity.average_watts != None,
            Activity.id.not_in(select(ActivityAnalysis.activity_id))
        ).where(
            or_(
                Activity.max_watts > p1, # If max_watts metadata exists
                Activity.average_watts > (p20 * 0.8 if p20 > 0 else 180), # Average power as proxy
                Activity.average_watts > (p60 * 0.8 if p60 > 0 else 170),
                Activity.average_watts > 250 # High absolute threshold
            )
        ).order_by(Activity.average_watts.desc()).limit(50) # Limit to 50 for backfilling
        
        candidates = self.session.exec(stmt).all()
        print(f"Found {len(candidates)} new record-breaking candidates.")
        
        from ai_analysis import run_activity_analysis
        count = 0
        for activity in candidates:
            try:
                print(f"Analyzing candidate: {activity.name} ({activity.start_date}) - Avg: {activity.average_watts}W")
                await run_activity_analysis(activity.id, self.session, self)
                count += 1
            except Exception as e:
                print(f"Failed to analyze candidate {activity.id}: {e}")
        
        if count > 0:
            await update_daily_training_load(self.session)

    async def get_activity_streams(self, activity_id: int) -> dict:
        """Fetch streams (time-series data) for a specific activity."""
        access_token = await self.get_valid_token()
        keys = ["time", "distance", "latlng", "altitude", "velocity_smooth", "heartrate", "cadence", "watts", "temp", "moving", "grade_smooth"]
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/activities/{activity_id}/streams",
                headers={"Authorization": f"Bearer {access_token}"},
                params={"keys": ",".join(keys), "key_by_type": True}
            )
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            return response.json()

    async def update_activity(self, activity_id: int, name: str):
        """Update an activity on Strava (e.g., set the name)."""
        access_token = await self.get_valid_token()
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{self.base_url}/activities/{activity_id}",
                headers={"Authorization": f"Bearer {access_token}"},
                json={"name": name}
            )
            response.raise_for_status()
            return response.json()
