"""
Summarization Service - Connects to Modal SOAP RAG endpoint via SDK
Run: python3 summarize.py
"""

# Import required libraries for FastAPI web service and Modal integration
import os
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import modal

# Configure logging to track requests and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Read environment variables for service configuration (host, port, Modal app name)
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8002"))
MODAL_APP_NAME = os.getenv("MODAL_APP_NAME", "soap-summarization-medgemma-4b-test")

# Initialize FastAPI application with metadata
app = FastAPI(
    title="SOAP Summarization Service",
    description="API gateway for Modal-hosted SOAP RAG",
    version="1.0.0"
)

# Enable CORS to allow frontend requests from different domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to cache Modal connection (initialized on first request)
_summarizer = None


# Get or create Modal connection (lazy initialization for performance)
def get_summarizer():
    """Get or create Modal summarizer reference."""
    global _summarizer
    if _summarizer is None:
        logger.info(f"Connecting to Modal app: {MODAL_APP_NAME}")
        SOAPSummarizer = modal.Cls.from_name(MODAL_APP_NAME, "SOAPSummarizer")
        _summarizer = SOAPSummarizer()
    return _summarizer


# Request model: defines expected JSON payload from frontend
class SummarizeRequest(BaseModel):
    text: str
    patient_name: Optional[str] = "Patient"
    openemr_text: Optional[str] = ""


# Response model: JSON wrapper containing SOAP note as plain text (not JSON)
class SummarizeResponse(BaseModel):
    success: bool
    soap_note: Optional[str] = None  # SOAP-formatted narrative text (Subjective, Objective, Assessment, Plan)
    detected_disease: Optional[str] = None
    patient_name: Optional[str] = None
    error: Optional[str] = None


# Main endpoint: receives transcript + FHIR data, sends to Modal for processing
@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    logger.info(f"Summarize request for: {request.patient_name}")

    if not request.text or len(request.text.strip()) < 10:
        return SummarizeResponse(success=False, error="Transcription too short")

    try:
        summarizer = get_summarizer()
        result = summarizer.generate_summary.remote(
            transcript_text=request.text,
            openemr_text=request.openemr_text or "",
            patient_name=request.patient_name
        )

        logger.info(f"SOAP generation complete for: {request.patient_name}")

        return SummarizeResponse(
            success=result.get("success", False),
            soap_note=result.get("soap_note"),
            detected_disease=result.get("detected_disease"),
            patient_name=result.get("patient_name"),
            error=result.get("error")
        )

    except modal.exception.NotFoundError:
        logger.error(f"Modal app '{MODAL_APP_NAME}' not found")
        return SummarizeResponse(
            success=False,
            error="Summarization service not deployed. Run 'modal deploy modal_summarize.py' first."
        )
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return SummarizeResponse(success=False, error=str(e))


# Warmup endpoint: pre-loads Modal connection to avoid cold start on first request
@app.get("/warmup")
async def warmup():
    try:
        summarizer = get_summarizer()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# Health check endpoint: verifies service is running
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "modal_app": MODAL_APP_NAME,
        "timestamp": datetime.utcnow().isoformat(),
    }


# Entry point: starts Uvicorn server when running this file directly
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting summarization service on {HOST}:{PORT}")
    logger.info(f"Modal app: {MODAL_APP_NAME}")
    uvicorn.run(app, host=HOST, port=PORT)
