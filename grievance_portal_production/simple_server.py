import os
from dotenv import load_dotenv
load_dotenv('backend/.env')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# Simple test server without AI services
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files for frontend
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
def index():
    return {"message": "Simple test server running"}

@app.get("/admin/petitions")
def test_admin_petitions(department: str):
    # Return test data including a petition with location
    return [
        {
            "_id": "test123",
            "tracking_id": "GR-2025-38BXFE",
            "name": "Test User",
            "petition_subject": "Test petition with location",
            "petition_description": "This is a test",
            "status": "pending",
            "priority": "Medium",
            "created_at": "22-Sep-2025",
            "location": {
                "latitude": 13.08268,
                "longitude": 80.270721,
                "accuracy": 10.5,
                "capture_method": "test_capture",
                "timestamp": "2025-01-21T10:30:00.000Z",
                "coordinates_string": "13.082680, 80.270721"
            }
        },
        {
            "_id": "test124",
            "tracking_id": "GR-2025-NOLOCAL",
            "name": "Another User",
            "petition_subject": "Test petition without location",
            "petition_description": "This is another test",
            "status": "pending",
            "priority": "Medium",
            "created_at": "22-Sep-2025"
            # No location data
        }
    ]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)