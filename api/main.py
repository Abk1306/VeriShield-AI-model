from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import os

from verishield_ai import analyze_document
from verishield_ai.scoring.risk_engine import calculate_risk_score

app = FastAPI(
    title="VeriShield AI API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "VeriShield AI",
    }


@app.get("/debug/version")
def debug_version():
    test_result = calculate_risk_score(
        0.20,
        [
            {"severity": "HIGH"},
            {"severity": "HIGH"},
            {"severity": "HIGH"},
        ],
    )

    return {
        "risk_engine_test": test_result,
        "expected_score": 32.0,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type.",
        )

    contents = await file.read()

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        result = analyze_document(temp_path)

        return result

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Required model or resource was not found.",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document analysis failed: {exc}",
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
