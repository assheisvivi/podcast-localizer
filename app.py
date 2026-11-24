"""Short-audio review API and web frontend for the Journalism AI prototype.

Requests are stateless: recordings and draft files exist only in a temporary
directory during one request. No visitor receives the provider API keys.
"""

from __future__ import annotations

import hmac
import csv
import io
import json
import os
import tempfile
import secrets
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import pipeline


ROOT = Path(__file__).resolve().parent
MAX_AUDIO_BYTES = 4_000_000
ALLOWED_SUFFIXES = {".mp3", ".m4a", ".wav", ".ogg", ".webm"}
app = FastAPI(title="Journalism AI Review API", docs_url=None, redoc_url=None)
SESSIONS: dict[str, dict[str, object]] = {}
SAMPLE_AUDIO = ROOT / "input/audio/trump_victory_speech_65s.mp3"
SAMPLE_REVIEW = ROOT / "runs/trump_demo_local/review.csv"


def qa_flags(row: dict[str, str], glossary: list[dict[str, str]]) -> list[str]:
    import re
    source = row["english_transcript"]
    target = row["ai_translation_zh"]
    flags = []
    source_numbers = re.findall(r"\d[\d,.\-]*\d|\d", source)
    target_numbers = re.findall(r"\d[\d,.\-]*\d|\d", target)
    if source_numbers != target_numbers:
        flags.append("数字待核")
    for term in glossary:
        english = term.get("english", "").strip()
        chinese = term.get("preferred_chinese", "").strip()
        if english and chinese and english.casefold() in source.casefold() and chinese not in target:
            flags.append("术语待核")
            break
    if row.get("editor_notes", "").strip():
        flags.append("事实待核")
    if '"' in source or "“" in source or "quote" in source.casefold():
        flags.append("引语待核")
    return flags


def create_session(rows: list[dict[str, str]], glossary: list[dict[str, str]]) -> dict[str, object]:
    session_id = secrets.token_urlsafe(24)
    enriched = [{**row, "qa_flags": qa_flags(row, glossary), "error_type": ""} for row in rows]
    SESSIONS[session_id] = {"rows": enriched, "glossary": glossary}
    return {"session_id": session_id, "rows": enriched, "segment_count": len(enriched)}


def session_rows(session_id: str) -> list[dict[str, str]]:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "Review session expired. Open the sample or upload audio again.")
    return session["rows"]


class ReviewRow(BaseModel):
    segment_id: str
    start_sec: str
    end_sec: str
    duration_sec: str
    english_transcript: str
    editor_english_transcript: str = ""
    ai_translation_zh: str
    editor_translation_zh: str = ""
    review_status: str = "needs_review"
    editor_notes: str = ""
    error_type: str = ""


class SynthesizeRequest(BaseModel):
    session_id: str
    backend: str = "minimax"
    voice: str = ""
    disclosure_confirmed: bool = False


class ReviewRequest(BaseModel):
    session_id: str
    rows: list[ReviewRow] = Field(min_length=1, max_length=100)


def require_demo_access(code: Optional[str]) -> None:
    expected = os.environ.get("DEMO_ACCESS_CODE", "")
    if not expected:
        raise HTTPException(503, "The server has no access code configured.")
    if not code or not hmac.compare_digest(code, expected):
        raise HTTPException(401, "Incorrect access code.")


def public_error(exc: pipeline.PipelineError) -> HTTPException:
    # Provider errors can include request details; keep those out of responses.
    message = str(exc)
    if message.startswith(("未检测到", "OPENAI_API_KEY 看起来", "审校表", "至少有一行")):
        return HTTPException(503, "The speech or translation service is not configured.")
    return HTTPException(502, "The upstream service failed. Please try again later.")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "processing_enabled": bool(os.environ.get("DEMO_ACCESS_CODE")),
        "transcription_enabled": bool(os.environ.get("OPENAI_API_KEY", "").startswith("sk-")),
        "tts_backends": {
            "minimax": bool(os.environ.get("MINIMAX_API_KEY")),
            "openai": bool(os.environ.get("OPENAI_API_KEY", "").startswith("sk-")),
            "elevenlabs": bool(os.environ.get("ELEVENLABS_API_KEY") and os.environ.get("ELEVENLABS_VOICE_ID")),
        },
    }


@app.post("/api/prepare")
async def prepare(
    audio: UploadFile = File(...),
    context: str = Form(""),
    glossary_json: str = Form("[]"),
    authorized: bool = Form(False),
    x_demo_access_code: Optional[str] = Header(default=None),
) -> dict[str, object]:
    require_demo_access(x_demo_access_code)
    if not authorized:
        raise HTTPException(400, "Confirm that you have permission to process this audio.")
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "Upload MP3, M4A, WAV, OGG, or WebM audio.")
    if len(context) > 1000:
        raise HTTPException(400, "Editorial context must be under 1,000 characters.")
    try:
        key = pipeline.require_api_key()
    except pipeline.PipelineError as exc:
        raise public_error(exc) from exc
    try:
        glossary = json.loads(glossary_json)
        if not isinstance(glossary, list) or len(glossary) > 100 or any(not isinstance(x, dict) for x in glossary):
            raise ValueError
    except (ValueError, TypeError):
        raise HTTPException(400, "Invalid glossary format.")
    with tempfile.TemporaryDirectory(prefix="journalism-ai-web-") as directory:
        audio_path = Path(directory) / ("recording" + suffix)
        total = 0
        with audio_path.open("wb") as target:
            while chunk := await audio.read(256 * 1024):
                total += len(chunk)
                if total > MAX_AUDIO_BYTES:
                    raise HTTPException(413, "Audio exceeds 4 MB. Trim the clip first.")
                target.write(chunk)
        if total < 1024:
            raise HTTPException(400, "Audio file is empty or too small.")
        try:
            transcript = pipeline.transcribe(audio_path, "whisper-1", key)
            segments = pipeline.extract_segments(transcript)
            translation = pipeline.translate_segments(
                segments, "gpt-5-mini", glossary, key, context
            )
            rows = pipeline.build_review_rows(segments, translation)
        except pipeline.PipelineError as exc:
            raise public_error(exc) from exc
    return create_session(rows, glossary)


@app.get("/api/glossary")
def glossary(x_demo_access_code: Optional[str] = Header(default=None)) -> dict[str, object]:
    require_demo_access(x_demo_access_code)
    return {"terms": pipeline.load_glossary(ROOT / "config/glossary.csv")}


@app.get("/api/sample")
def sample(x_demo_access_code: Optional[str] = Header(default=None)) -> dict[str, object]:
    require_demo_access(x_demo_access_code)
    if not SAMPLE_AUDIO.exists() or not SAMPLE_REVIEW.exists():
        raise HTTPException(404, "The local development sample is unavailable.")
    rows = pipeline.read_review_csv(SAMPLE_REVIEW)
    for row in rows:
        row["review_status"] = "needs_review"
        row["editor_translation_zh"] = ""
    return create_session(rows, pipeline.load_glossary(ROOT / "config/glossary.csv"))


@app.get("/api/sample-audio")
def sample_audio(x_demo_access_code: Optional[str] = Header(default=None)) -> Response:
    require_demo_access(x_demo_access_code)
    if not SAMPLE_AUDIO.exists():
        raise HTTPException(404, "The local development audio is unavailable.")
    return FileResponse(SAMPLE_AUDIO, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@app.post("/api/review")
def review(request: ReviewRequest, x_demo_access_code: Optional[str] = Header(default=None)) -> dict[str, object]:
    require_demo_access(x_demo_access_code)
    originals = session_rows(request.session_id)
    if len(originals) != len(request.rows):
        raise HTTPException(400, "Review segment count does not match.")
    updated = []
    for original, model in zip(originals, request.rows):
        row = model.model_dump()
        if row["segment_id"] != original["segment_id"] or row["english_transcript"] != original["english_transcript"] or row["ai_translation_zh"] != original["ai_translation_zh"]:
            raise HTTPException(400, "The source draft changed. Review it again.")
        if row["review_status"] not in {"approved", "needs_review"}:
            raise HTTPException(400, "Invalid review status.")
        if row["review_status"] == "approved" and not row["editor_translation_zh"].strip():
            raise HTTPException(400, "Enter final Chinese text before approval.")
        updated.append({**original, **row})
    SESSIONS[request.session_id]["rows"] = updated
    return {"approved": sum(r["review_status"] == "approved" for r in updated), "total": len(updated)}


@app.get("/api/audit/{session_id}")
def audit(session_id: str, x_demo_access_code: Optional[str] = Header(default=None)) -> Response:
    require_demo_access(x_demo_access_code)
    rows = session_rows(session_id)
    fields = ["segment_id", "start_sec", "english_transcript", "editor_english_transcript", "ai_translation_zh", "editor_translation_zh", "qa_flags", "error_type", "review_status", "editor_notes"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: ";".join(row.get("qa_flags", [])) if key == "qa_flags" else row.get(key, "") for key in fields})
    return Response("\ufeff" + buffer.getvalue(), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": 'attachment; filename="human_edit_log.csv"', "Cache-Control": "no-store"})


@app.post("/api/synthesize")
def synthesize(
    request: SynthesizeRequest,
    x_demo_access_code: Optional[str] = Header(default=None),
) -> Response:
    require_demo_access(x_demo_access_code)
    rows = session_rows(request.session_id)
    if not request.disclosure_confirmed:
        raise HTTPException(400, "Confirm that the synthetic voice will be disclosed to listeners.")
    if any(row["review_status"] != "approved" or not row["editor_translation_zh"].strip() for row in rows):
        raise HTTPException(400, "Complete and approve every Chinese segment first.")
    if sum(len(row["editor_translation_zh"]) for row in rows) > 5000:
        raise HTTPException(400, "Chinese script is too long; this prototype accepts up to 5,000 characters.")
    if request.backend not in {"minimax", "openai", "elevenlabs"}:
        raise HTTPException(400, "Unsupported voice service.")
    with tempfile.TemporaryDirectory(prefix="journalism-ai-web-") as directory:
        review_path = Path(directory) / "review.csv"
        output_path = Path(directory) / "chinese_audio.mp3"
        pipeline.write_review_csv(review_path, [{key: row.get(key, "") for key in pipeline.REVIEW_FIELDS} for row in rows])
        try:
            if request.backend == "minimax":
                pipeline.synthesize_minimax(
                    review_path, output_path, "speech-2.8-hd",
                    request.voice or "male-qn-qingse", 0.96, False, True,
                    pipeline.require_env_key("MINIMAX_API_KEY"),
                )
            elif request.backend == "openai":
                pipeline.synthesize(
                    review_path, output_path, "gpt-4o-mini-tts",
                    request.voice or "cedar", False, True, pipeline.require_api_key(),
                )
            else:
                pipeline.synthesize_elevenlabs(
                    review_path, output_path, "eleven_v4",
                    os.environ.get("ELEVENLABS_VOICE_ID", ""), False, True,
                    pipeline.require_env_key("ELEVENLABS_API_KEY"),
                )
        except pipeline.PipelineError as exc:
            raise public_error(exc) from exc
        audio = output_path.read_bytes()
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'attachment; filename="chinese_audio.mp3"', "Cache-Control": "no-store"},
    )


app.mount("/", StaticFiles(directory=ROOT / "docs", html=True), name="site")
