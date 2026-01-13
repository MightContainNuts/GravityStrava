from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.routes import activities, health, ai, auth, views
from app.core.database import create_db_and_tables

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(
    title="GravityStrato",
    description="Cycling Performance Dashboard with AI Insights",
    version="2.0.0",
    lifespan=lifespan
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(views.router)
app.include_router(auth.router)
app.include_router(activities.router)
app.include_router(health.router)
app.include_router(ai.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
