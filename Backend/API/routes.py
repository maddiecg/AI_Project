# routes.py
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from Backend.recommendations import recommend_from_image

app = FastAPI(title="FashionMind AI")

ALLOWED = {".jpg", ".jpeg", ".png", ".webp"}


@app.post("/api/recommend")
async def recommend(image: UploadFile = File(...), color: str | None = Form(None)):
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(400, "Please upload a JPG, PNG, or WEBP image.")

    # Your functions expect a file path, so save the upload temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        return recommend_from_image(tmp_path, color or None)
    except Exception as e:
        raise HTTPException(500, f"Could not process image: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Serve the website. This must come AFTER the API routes.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "Frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")