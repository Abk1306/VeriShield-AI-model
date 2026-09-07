from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from verishield_ai import analyze_document


app = FastAPI(
    title="VeriShield AI API",
    description="Digital document tampering detection API",
    version="1.0.0",
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "VeriShield AI",
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    extension = Path(file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension or 'unknown'}",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            shutil.copyfileobj(file.file, temp_file)

        result = analyze_document(str(temp_path))

        return result

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Required model or resource is unavailable",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document analysis failed: {exc}",
        ) from exc

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
