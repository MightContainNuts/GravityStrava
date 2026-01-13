import time
from datetime import datetime
import httpx
from typing import List, Optional
from database import WithingsToken, UserHealth
from sqlmodel import Session, select
from config import settings

class WithingsClient:
    def __init__(self, session: Session):
        self.session = session
        self.base_url = "https://wbsapi.withings.net"

    def has_token(self) -> bool:
        statement = select(WithingsToken).order_by(WithingsToken.id.desc())
        token = self.session.exec(statement).first()
        return token is not None

    def get_auth_url(self) -> str:
        import urllib.parse
        encoded_uri = urllib.parse.quote(settings.withings_redirect_uri, safe='')
        url = (
            f"https://account.withings.com/oauth2_user/authorize2?"
            f"response_type=code&"
            f"client_id={settings.withings_client_id}&"
            f"scope=user.metrics&"
            f"redirect_uri={encoded_uri}&"
            f"state=withings_auth_state"
        )
        return url

    async def exchange_token(self, code: str) -> WithingsToken:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v2/oauth2",
                data={
                    "action": "requesttoken",
                    "client_id": settings.withings_client_id,
                    "client_secret": settings.withings_client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": settings.withings_redirect_uri,
                },
            )
            response.raise_for_status()
            data = response.json()
            if data.get("status") != 0:
                raise Exception(f"Withings Error: {data.get('error')}")
            
            body = data["body"]
            token = WithingsToken(
                access_token=body["access_token"],
                refresh_token=body["refresh_token"],
                expires_at=int(time.time()) + body["expires_in"],
                userid=body["userid"]
            )
            self.session.add(token)
            self.session.commit()
            self.session.refresh(token)
            return token

    async def get_valid_token(self) -> str:
        statement = select(WithingsToken).order_by(WithingsToken.id.desc())
        token = self.session.exec(statement).first()
        
        if not token:
            raise Exception("No Withings token found. Please authenticate.")

        if token.expires_at < time.time() + 60:
            # Refresh token
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/v2/oauth2",
                    data={
                        "action": "requesttoken",
                        "client_id": settings.withings_client_id,
                        "client_secret": settings.withings_client_secret,
                        "refresh_token": token.refresh_token,
                        "grant_type": "refresh_token",
                    },
                )
                response.raise_for_status()
                data = response.json()
                if data.get("status") != 0:
                     raise Exception(f"Withings Refresh Error: {data.get('error')}")
                
                body = data["body"]
                token.access_token = body["access_token"]
                token.refresh_token = body["refresh_token"]
                token.expires_at = int(time.time()) + body["expires_in"]
                self.session.add(token)
                self.session.commit()
                self.session.refresh(token)
        
        return token.access_token

    async def fetch_measures(self, startdate: Optional[int] = None) -> List[dict]:
        """Fetch measures (weight, heart rate, etc.)"""
        access_token = await self.get_valid_token()
        params = {
            "action": "getmeas",
            "meastypes": "1,11,71,73,76,8,54,123,163", # weight, pulse, fat ratio, hydration, muscle mass, VO2, etc.
            "category": 1, # real measures
        }
        if startdate:
            params["startdate"] = startdate
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/measure",
                headers={"Authorization": f"Bearer {access_token}"},
                data=params
            )
            response.raise_for_status()
            data = response.json()
            if data.get("status") != 0:
                return []
            return data["body"].get("measuregrps", [])

    def sync_measures(self, measuregrps: List[dict]):
        for group in measuregrps:
            date = datetime.fromtimestamp(group["date"])
            
            # Check if exists for this date (simpler check)
            statement = select(UserHealth).where(UserHealth.date == date)
            existing = self.session.exec(statement).first()
            
            if not existing:
                health_data = {"date": date, "source": "withings"}
                for m in group["measures"]:
                    val = m["value"] * (10 ** m["unit"])
                    m_type = m["type"]
                    if m_type == 1: health_data["weight"] = val
                    elif m_type == 71: health_data["fat_ratio"] = val
                    elif m_type == 8: health_data["fat_mass_weight"] = val # Fat mass (kg)
                    elif m_type == 76: health_data["muscle_mass"] = val
                    elif m_type == 88: health_data["bone_mass"] = val
                    elif m_type == 73: health_data["hydration"] = val
                    elif m_type == 11: health_data["pulse"] = int(val) # Heart Rate (bpm)
                    elif m_type in [123, 163]: health_data["fitness_level"] = int(val) # VO2 Max
                
                if any(k in health_data for k in ["weight", "fat_ratio", "muscle_mass", "pulse", "fitness_level"]):
                    health = UserHealth(**health_data)
                    self.session.add(health)
        
        self.session.commit()

    def get_latest_weight(self) -> Optional[float]:
        statement = select(UserHealth).where(UserHealth.weight != None).order_by(UserHealth.date.desc())
        latest = self.session.exec(statement).first()
        return latest.weight if latest else None
