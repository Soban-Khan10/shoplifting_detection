import json
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = REPO_ROOT / "uploads"
RESULTS_DIR = REPO_ROOT / "results"
BACKEND_OUTPUTS_DIR = REPO_ROOT / "outputs" / "backend_jobs"


def utc_now_iso() -> str:
    """Return a simple UTC timestamp for job history."""
    return datetime.now(timezone.utc).isoformat()


def new_job_id() -> str:
    """Create a short, URL-safe job id."""
    return uuid.uuid4().hex


def ensure_project_dirs() -> None:
    """Create folders used by the API if they do not already exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BACKEND_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def is_safe_job_id(job_id: str) -> bool:
    """Job ids are flat folder names; reject path traversal and empty values."""
    return bool(job_id) and Path(job_id).name == job_id


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def save_upload_file(upload_file: Any, job_id: str) -> Path:
    """Save the uploaded MP4 into uploads/{job_id}/."""
    upload_dir = UPLOADS_DIR / job_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(upload_file.filename or "uploaded_video.mp4").name
    if not original_name.lower().endswith(".mp4"):
        original_name = f"{Path(original_name).stem}.mp4"

    video_path = upload_dir / original_name
    with video_path.open("wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return video_path


def run_detection_command(job_id: str, video_path: Path) -> subprocess.CompletedProcess[str]:
    """Run the existing product pipeline without modifying its code."""
    output_dir = BACKEND_OUTPUTS_DIR / job_id / "video_detection_i3d"
    person_match_dir = BACKEND_OUTPUTS_DIR / job_id / "person_match_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    person_match_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "python",
        "run_video_detection.py",
        "--video",
        str(video_path),
        "--device",
        "cpu",
        "--stage2-checkpoint",
        "temporal_model_direct_bilstm_corrected.pth",
        "--top-k",
        "5",
        "--simulate-person-match",
        "--output-dir",
        str(output_dir),
        "--person-match-output-dir",
        str(person_match_dir),
    ]

    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def copy_if_exists(source: str | Path | None, result_dir: Path) -> str | None:
    """Copy one generated file into results/{job_id}/ and return its new filename."""
    if not source:
        return None
    source_path = Path(source)
    if not source_path.exists() or not source_path.is_file():
        return None

    destination = result_dir / source_path.name
    shutil.copy2(source_path, destination)
    return destination.name


def copy_person_match_files(result: dict[str, Any], result_dir: Path) -> dict[str, str]:
    """Copy simulated person-match JSON/image files when the pipeline created them."""
    copied: dict[str, str] = {}

    match_json_name = copy_if_exists(result.get("person_match_result_json_path"), result_dir)
    if match_json_name:
        copied["result_json"] = match_json_name

    match_dir = result.get("person_match_output_dir")
    if match_dir:
        for path in sorted(Path(match_dir).glob("*")):
            if path.is_file():
                copied[path.stem] = copy_if_exists(path, result_dir) or path.name

    return copied


def find_result_json(job_id: str) -> Path:
    """Locate the main inference JSON created by run_video_detection.py."""
    output_dir = BACKEND_OUTPUTS_DIR / job_id / "video_detection_i3d"
    result_files = sorted(output_dir.glob("*_result.json"))
    if not result_files:
        raise FileNotFoundError(f"No result JSON found in {output_dir}")
    return result_files[0]


def media_url(job_id: str, filename: str | None) -> str | None:
    if not filename:
        return None
    return f"/api/media/{job_id}/{filename}"


def normalize_response(
    job_id: str,
    result: dict[str, Any],
    copied: dict[str, str],
    simulated_match_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the clean API response returned by /api/detect and /api/results."""
    media = {
        "result_json": media_url(job_id, copied.get("result_json")),
        "score_plot": media_url(job_id, copied.get("score_plot")),
        "suspicious_clip": media_url(job_id, copied.get("suspicious_clip")),
        "key_frame": media_url(job_id, copied.get("key_frame")),
    }

    simulated_match = None
    if copied.get("person_match_result_json"):
        simulated_match = {
            "warning": result.get("person_match_warning"),
            "matched_profile": (simulated_match_data or {}).get("matched_profile"),
            "simulated_confidence": (simulated_match_data or {}).get("simulated_confidence"),
            "result_json_url": media_url(job_id, copied.get("person_match_result_json")),
            "suspect_crop_url": media_url(job_id, copied.get("suspect_crop")),
        }

    return {
        "job_id": job_id,
        "status": "completed",
        "alert": bool(result.get("alert", False)),
        "stage1_score": result.get("stage1_max_score"),
        "suspicious_timestamp": {
            "start_sec": result.get("suspicious_start_time_sec"),
            "end_sec": result.get("suspicious_end_time_sec"),
        },
        "media_urls": media,
        "top_k_windows": result.get("top_suspicious_time_windows")
        or result.get("top_suspicious_segments")
        or [],
        "simulated_match": simulated_match,
    }


def collect_artifacts(job_id: str) -> dict[str, Any]:
    """Copy generated pipeline artifacts into results/{job_id}/."""
    result_dir = RESULTS_DIR / job_id
    result_dir.mkdir(parents=True, exist_ok=True)

    result_json_path = find_result_json(job_id)
    result = read_json(result_json_path)

    copied = {
        "result_json": copy_if_exists(result_json_path, result_dir),
        "score_plot": copy_if_exists(result.get("plot_path"), result_dir),
        "suspicious_clip": copy_if_exists(result.get("suspicious_clip_path"), result_dir),
        "key_frame": copy_if_exists(result.get("suspicious_key_frame_path"), result_dir),
    }

    person_files = copy_person_match_files(result, result_dir)
    if person_files.get("result_json"):
        copied["person_match_result_json"] = person_files["result_json"]
    if person_files.get("suspect_crop"):
        copied["suspect_crop"] = person_files["suspect_crop"]

    simulated_match_data = None
    if copied.get("person_match_result_json"):
        simulated_match_data = read_json(result_dir / copied["person_match_result_json"])

    response = normalize_response(job_id, result, copied, simulated_match_data)
    response["created_at"] = utc_now_iso()
    write_json(result_dir / "summary.json", response)
    return response


def load_job_summary(job_id: str) -> dict[str, Any] | None:
    if not is_safe_job_id(job_id):
        return None
    summary_path = RESULTS_DIR / job_id / "summary.json"
    if not summary_path.exists():
        return None
    return read_json(summary_path)


def list_history() -> list[dict[str, Any]]:
    """Return previous job summaries newest first."""
    jobs = []
    for summary_path in RESULTS_DIR.glob("*/summary.json"):
        try:
            jobs.append(read_json(summary_path))
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(jobs, key=lambda item: item.get("created_at", ""), reverse=True)


def delete_job(job_id: str) -> dict[str, Any] | None:
    """Delete stored artifacts for one completed or failed API job."""
    if not is_safe_job_id(job_id):
        return None

    job_dirs = [
        RESULTS_DIR / job_id,
        UPLOADS_DIR / job_id,
        BACKEND_OUTPUTS_DIR / job_id,
    ]

    if not any(path.exists() for path in job_dirs):
        return None

    for path in job_dirs:
        if path.exists():
            shutil.rmtree(path)

    return {
        "job_id": job_id,
        "deleted": True,
        "message": "Job deleted successfully.",
    }


def fail_job(job_id: str, message: str, stdout: str = "", stderr: str = "") -> dict[str, Any]:
    """Persist a readable error summary for failed jobs."""
    summary = {
        "job_id": job_id,
        "status": "failed",
        "alert": None,
        "error": message,
        "stdout": stdout,
        "stderr": stderr,
        "created_at": utc_now_iso(),
    }
    write_json(RESULTS_DIR / job_id / "summary.json", summary)
    return summary
