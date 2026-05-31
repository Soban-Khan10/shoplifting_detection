from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

try:
    from . import jobs
except ImportError:
    import jobs


app = FastAPI(title="Shoplifting Detection API")

# Friendly defaults for local frontend development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs.ensure_project_dirs()
(Path(__file__).resolve().parent / "static").mkdir(parents=True, exist_ok=True)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "shoplifting-detection-api"}


@app.post("/api/detect")
def detect(video: UploadFile = File(...)) -> dict:
    """Save an uploaded MP4, run the existing pipeline, and return clean results."""
    if not video.filename or not video.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Please upload an MP4 video.")

    job_id = jobs.new_job_id()
    video_path = jobs.save_upload_file(video, job_id)
    completed = jobs.run_detection_command(job_id, video_path)

    if completed.returncode != 0:
        summary = jobs.fail_job(
            job_id=job_id,
            message=f"Detection command failed with exit code {completed.returncode}.",
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        raise HTTPException(status_code=500, detail=summary)

    try:
        return jobs.collect_artifacts(job_id)
    except (FileNotFoundError, OSError, ValueError) as exc:
        summary = jobs.fail_job(
            job_id=job_id,
            message=f"Detection finished, but result collection failed: {exc}",
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        raise HTTPException(status_code=500, detail=summary)


@app.get("/api/results/{job_id}")
def get_results(job_id: str) -> dict:
    summary = jobs.load_job_summary(job_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return summary


@app.delete("/api/results/{job_id}")
def delete_results(job_id: str) -> dict:
    deleted = jobs.delete_job(job_id)
    if deleted is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return deleted


@app.get("/api/history")
def history() -> list[dict]:
    return jobs.list_history()


@app.get("/api/media/{job_id}/{filename}")
def media(job_id: str, filename: str) -> FileResponse:
    # Only serve files directly inside results/{job_id}; no nested paths.
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    path = jobs.RESULTS_DIR / job_id / safe_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Media file not found.")

    return FileResponse(path)
