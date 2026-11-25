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
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import pipeline
import review_store


ROOT = Path(__file__).resolve().parent
MAX_AUDIO_BYTES = 4_000_000
ALLOWED_SUFFIXES = {".mp3", ".m4a", ".wav", ".ogg", ".webm"}
@asynccontextmanager
async def lifespan(_app: FastAPI):
    review_store.ensure_ready()
    yield


app = FastAPI(title="Journalism AI Review API", docs_url=None, redoc_url=None, lifespan=lifespan)
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
    review_store.create_session(session_id, {"rows": enriched, "glossary": glossary})
    return {"session_id": session_id, "rows": enriched, "segment_count": len(enriched)}


def session_rows(session_id: str) -> list[dict[str, str]]:
    session = review_store.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Review session expired. Open the sample or upload audio again.")
    return session["rows"]


def consume_quota(action: str, environment_name: str, default: int) -> None:
    try:
        limit = max(0, int(os.environ.get(environment_name, str(default))))
    except ValueError as exc:
        raise HTTPException(503, f"Invalid {environment_name} configuration.") from exc
    if not review_store.consume_daily_quota(action, limit):
        now = datetime.now(timezone.utc)
        tomorrow = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        raise HTTPException(
            429,
            "Daily processing limit reached. Try again tomorrow (UTC).",
            headers={"Retry-After": str(max(1, int((tomorrow - now).total_seconds())))},
        )


class ReviewRow(BaseModel):
    segment_id: str = Field(max_length=32)
    start_sec: str = Field(max_length=32)
    end_sec: str = Field(max_length=32)
    duration_sec: str = Field(max_length=32)
    english_transcript: str = Field(max_length=10000)
    editor_english_transcript: str = Field(default="", max_length=10000)
    ai_translation_zh: str = Field(max_length=10000)
    editor_translation_zh: str = Field(default="", max_length=10000)
    review_status: str = "needs_review"
    editor_notes: str = Field(default="", max_length=2000)
    error_type: str = Field(default="", max_length=32)


class SynthesizeRequest(BaseModel):
    session_id: str
    backend: str = "edge"
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
    if "HTTP 402" in message:
        return HTTPException(402, "The AI account has no available credits. The classroom demo is still available.")
    if "HTTP 403" in message:
        return HTTPException(403, "The AI account cannot use the selected model. Ask the owner to check provider access.")
    if "HTTP 401" in message:
        return HTTPException(503, "The AI service key is invalid or has expired. Ask the owner to update it.")
    if message.startswith(("未检测到", "OPENAI_API_KEY 看起来", "审校表", "至少有一行")):
        return HTTPException(503, "The speech or translation service is not configured.")
    return HTTPException(502, "The upstream service failed. Please try again later.")


def ai_provider() -> str:
    selected = os.environ.get("AI_PROVIDER", "").strip().lower()
    if selected:
        return selected if selected in {"groq", "openrouter", "openai"} else "invalid"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.environ.get("OPENAI_API_KEY", "").startswith("sk-"):
        return "openai"
    return ""


@app.get("/api/health")
def health() -> dict[str, object]:
    provider = ai_provider()
    ai_key_available = bool(
        (provider == "groq" and os.environ.get("GROQ_API_KEY"))
        or (provider == "openrouter" and os.environ.get("OPENROUTER_API_KEY"))
        or (provider == "openai" and os.environ.get("OPENAI_API_KEY", "").startswith("sk-"))
    )
    return {
        "status": "ok",
        "review_storage": review_store.backend(),
        "processing_enabled": bool(os.environ.get("DEMO_ACCESS_CODE") and ai_key_available),
        "ai_provider": provider if ai_key_available else "",
        "sample_available": SAMPLE_AUDIO.exists() and SAMPLE_REVIEW.exists(),
        "transcription_enabled": ai_key_available,
        "tts_backends": {
            "edge": True,
            "minimax": bool(os.environ.get("MINIMAX_API_KEY")),
            "openai": bool(os.environ.get("OPENAI_API_KEY", "").startswith("sk-")),
            "elevenlabs": bool(os.environ.get("ELEVENLABS_API_KEY") and os.environ.get("ELEVENLABS_VOICE_ID")),
            "openrouter": bool(os.environ.get("OPENROUTER_API_KEY")),
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
    if len(glossary_json) > 20000:
        raise HTTPException(400, "Glossary is too long.")
    try:
        provider = ai_provider()
        if provider == "groq":
            key = pipeline.require_env_key("GROQ_API_KEY")
        elif provider == "openrouter":
            key = pipeline.require_env_key("OPENROUTER_API_KEY")
        elif provider == "openai":
            key = pipeline.require_api_key()
        else:
            raise pipeline.PipelineError("未检测到可用的转写与翻译服务。")
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
        consume_quota("prepare", "MAX_PREPARES_PER_DAY", 20)
        try:
            transcript = pipeline.transcribe(
                audio_path,
                (os.environ.get("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3") if provider == "groq"
                 else os.environ.get("OPENROUTER_TRANSCRIPTION_MODEL", "mistralai/voxtral-small-transcribe-3-0") if provider == "openrouter"
                 else "whisper-1"),
                key,
                (pipeline.GROQ_API_BASE if provider == "groq"
                 else pipeline.OPENROUTER_API_BASE if provider == "openrouter"
                 else pipeline.API_BASE),
            )
            segments = pipeline.extract_segments(transcript)
            if provider == "groq":
                translation = pipeline.translate_segments_groq(
                    segments, os.environ.get("GROQ_TRANSLATION_MODEL", "qwen/qwen3.8-27b"), glossary, key, context
                )
            elif provider == "openrouter":
                translation = pipeline.translate_segments_openrouter(
                    segments, os.environ.get("OPENROUTER_TRANSLATION_MODEL", "qwen/qwen3.8-2.4t-a95b"), glossary, key, context
                )
            else:
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


@app.get("/api/session/{session_id}")
def resume_session(session_id: str, x_demo_access_code: Optional[str] = Header(default=None)) -> dict[str, object]:
    require_demo_access(x_demo_access_code)
    return {"session_id": session_id, "rows": session_rows(session_id)}


@app.post("/api/review")
def review(request: ReviewRequest, x_demo_access_code: Optional[str] = Header(default=None)) -> dict[str, object]:
    require_demo_access(x_demo_access_code)
    originals = session_rows(request.session_id)
    if len(originals) != len(request.rows):
        raise HTTPException(400, "Review segment count does not match.")
    updated = []
    changes = []
    for original, model in zip(originals, request.rows):
        row = model.model_dump()
        immutable = ("segment_id", "start_sec", "end_sec", "duration_sec", "english_transcript", "ai_translation_zh")
        if any(row[key] != original[key] for key in immutable):
            raise HTTPException(400, "The source draft changed. Review it again.")
        if row["review_status"] not in {"approved", "needs_review"}:
            raise HTTPException(400, "Invalid review status.")
        if row["review_status"] == "approved" and not row["editor_translation_zh"].strip():
            raise HTTPException(400, "Enter final Chinese text before approval.")
        changed = {**original, **row}
        updated.append(changed)
        if changed != original:
            changes.append((row["segment_id"], original, changed))
    session = review_store.get_session(request.session_id)
    if session is None or not review_store.update_session(request.session_id, {**session, "rows": updated}, changes):
        raise HTTPException(404, "Review session expired. Open the sample or upload audio again.")
    return {"approved": sum(r["review_status"] == "approved" for r in updated), "total": len(updated)}


@app.get("/api/review-history/{session_id}")
def review_history(session_id: str, x_demo_access_code: Optional[str] = Header(default=None)) -> dict[str, object]:
    require_demo_access(x_demo_access_code)
    session_rows(session_id)
    return {"events": review_store.review_events(session_id)}


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
    if request.backend not in {"edge", "minimax", "openai", "elevenlabs", "openrouter"}:
        raise HTTPException(400, "Unsupported voice service.")
    if request.backend == "edge" and request.voice and request.voice not in pipeline.EDGE_VOICES:
        raise HTTPException(400, "Choose Yunxi or Xiaoxiao for the free Mandarin voice.")
    try:
        if request.backend == "edge":
            provider_key = ""
        elif request.backend == "minimax":
            provider_key = pipeline.require_env_key("MINIMAX_API_KEY")
        elif request.backend == "openai":
            provider_key = pipeline.require_api_key()
        elif request.backend == "openrouter":
            provider_key = pipeline.require_env_key("OPENROUTER_API_KEY")
        else:
            provider_key = pipeline.require_env_key("ELEVENLABS_API_KEY")
            if not os.environ.get("ELEVENLABS_VOICE_ID"):
                raise pipeline.PipelineError("未检测到 ELEVENLABS_VOICE_ID。")
    except pipeline.PipelineError as exc:
        raise public_error(exc) from exc
    consume_quota("synthesize", "MAX_SYNTHESIS_PER_DAY", 10)
    with tempfile.TemporaryDirectory(prefix="journalism-ai-web-") as directory:
        review_path = Path(directory) / "review.csv"
        output_path = Path(directory) / "chinese_audio.mp3"
        pipeline.write_review_csv(review_path, [{key: row.get(key, "") for key in pipeline.REVIEW_FIELDS} for row in rows])
        try:
            if request.backend == "edge":
                pipeline.synthesize_edge(
                    review_path, output_path,
                    request.voice or "zh-CN-YunxiNeural", False, True,
                )
            elif request.backend == "minimax":
                pipeline.synthesize_minimax(
                    review_path, output_path, "speech-2.8-hd",
                    request.voice or "male-qn-qingse", 0.96, False, True,
                    provider_key,
                )
            elif request.backend == "openai":
                pipeline.synthesize(
                    review_path, output_path, "gpt-4o-mini-tts",
                    request.voice or "cedar", False, True, provider_key,
                )
            elif request.backend == "openrouter":
                pipeline.synthesize_openrouter(
                    review_path, output_path,
                    os.environ.get("OPENROUTER_TTS_MODEL", "minimax/speech-2.8-hd"),
                    request.voice or os.environ.get("OPENROUTER_TTS_VOICE", "male-qn-qingse"),
                    False, True, provider_key,
                )
            else:
                pipeline.synthesize_elevenlabs(
                    review_path, output_path, "eleven_v4",
                    os.environ.get("ELEVENLABS_VOICE_ID", ""), False, True,
                    provider_key,
                )
        except pipeline.PipelineError as exc:
            if request.backend == "edge":
                raise HTTPException(502, "Free Mandarin MP3 voice is unavailable. Use browser preview for now.") from exc
            raise public_error(exc) from exc
        audio = output_path.read_bytes()
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Content-Disposition": 'attachment; filename="chinese_audio.mp3"', "Cache-Control": "no-store"},
    )


app.mount("/", StaticFiles(directory=ROOT / "docs", html=True), name="site")
