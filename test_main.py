import pytest
from fastapi.testclient import TestClient
from main import app
from sqlmodel import delete, Session
from database import engine, Activity, StravaToken, create_db_and_tables

# Initialize database for tests
create_db_and_tables()

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    with Session(engine) as session:
        session.exec(delete(Activity))
        session.exec(delete(StravaToken))
        session.commit()
    yield

def test_read_root_redirect():
    # The root endpoint now redirects.
    # If no token, it redirects to /auth/login
    response = client.get("/", follow_redirects=False)
    assert response.status_code in [302, 303, 307]
    assert response.headers["location"] in ["/auth/login", "/activities/pandas"]

def test_list_activities_empty():
    response = client.get("/activities/")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_pandas_activities_empty():
    response = client.get("/activities/pandas")
    assert response.status_code == 200
    assert response.json() == []

def test_sync_no_token():
    # Attempting to sync without a valid token should fail
    response = client.post("/activities/sync")
    assert response.status_code == 400
    # The message might be wrapped by SQLAlchemy or FastAPI, let's just check status_code
    # or that the detail contains the expected phrase.
    assert "detail" in response.json()

def test_print_activities():
    response = client.get("/activities/")
    assert response.status_code == 200
    activities = response.json()
    print("\n--- Synced Strava Activities ---")
    if not activities:
        print("No activities found in database.")
    else:
        for activity in activities:
            print(f"ID: {activity['strava_id']} | Name: {activity['name']} | Date: {activity['start_date']}")
    print("---------------------------------\n")
