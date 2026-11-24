#!/usr/bin/env python3
"""One-minute, human-in-the-loop English-to-Chinese podcast dubbing MVP.

The production path deliberately stops after creating review.csv. A human editor
must review every row and set review_status to "approved" before synthesis.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


API_BASE = "https://api.openai.com/v1"
MINIMAX_TTS_URL = "https://api.minimax.io/v1/t2a_v2"
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech"
REVIEW_FIELDS = [
    "segment_id",
    "start_sec",
    "end_sec",
    "duration_sec",
    "english_transcript",
    "editor_english_transcript",
    "ai_translation_zh",
    "editor_translation_zh",
    "review_status",
    "editor_notes",
]


class PipelineError(RuntimeError):
    pass


def require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise PipelineError("未检测到 OPENAI_API_KEY。请先在环境变量中配置 API Key。")
    if not key.isascii() or not key.startswith("sk-"):
        raise PipelineError(
            "OPENAI_API_KEY 看起来是占位符而不是有效密钥；"
            "它必须是以 sk- 开头的 ASCII 字符串。"
        )
    return key


def require_env_key(name: str) -> str:
    key = os.environ.get(name, "").strip()
    if not key:
        raise PipelineError(f"未检测到 {name}。请先在环境变量中配置 API Key。")
    if not key.isascii():
        raise PipelineError(f"{name} 看起来是占位符而不是有效密钥。")
    return key


def api_request_json(endpoint: str, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        API_BASE + endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return json.loads(_open(request).decode("utf-8"))


def api_request_binary(endpoint: str, payload: dict[str, Any], api_key: str) -> bytes:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        API_BASE + endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    return _open(request)


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> bytes:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    return _open(request)


def api_request_multipart(
    endpoint: str,
    fields: list[tuple[str, str]],
    file_field: str,
    file_path: Path,
    api_key: str,
) -> dict[str, Any]:
    boundary = f"----journalism-ai-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{file_path.name}"\r\n'
            ).encode(),
            f"Content-Type: {mime_type}\r\n\r\n".encode(),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    request = Request(
        API_BASE + endpoint,
        data=b"".join(chunks),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )
    return json.loads(_open(request).decode("utf-8"))


def _open(request: Request) -> bytes:
    try:
        with urlopen(request, timeout=300) as response:
            return response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise PipelineError(f"API 请求失败（HTTP {exc.code}）：{detail}") from exc
    except URLError as exc:
        raise PipelineError(f"网络请求失败：{exc.reason}") from exc


def transcribe(audio_path: Path, model: str, api_key: str) -> dict[str, Any]:
    return api_request_multipart(
        "/audio/transcriptions",
        [
            ("model", model),
            ("language", "en"),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "segment"),
            ("timestamp_granularities[]", "word"),
        ],
        "file",
        audio_path,
        api_key,
    )


def extract_segments(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words = transcript.get("words") or []
    if words:
        sentences: list[dict[str, Any]] = []
        parts: list[str] = []
        start = 0.0
        end = 0.0
        for item in words:
            token = str(item.get("word", "")).strip()
            if not token:
                continue
            if not parts:
                start = float(item.get("start", 0))
            end = float(item.get("end", start))
            if parts and not re.match(r"^[.,!?;:%)\]}]", token) and not token.startswith("'"):
                parts.append(" ")
            parts.append(token)
            # A period in an abbreviation is not a sentence boundary.
            abbreviation = re.fullmatch(r"(?:[A-Za-z]\.){2,}|(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St)\.", token, re.I)
            if re.search(r"[.!?][\"'”’)?\]]*$", token) and not abbreviation:
                sentences.append({"segment_id": len(sentences), "start": round(start, 3), "end": round(end, 3), "text": "".join(parts)})
                parts = []
        if parts:
            sentences.append({"segment_id": len(sentences), "start": round(start, 3), "end": round(end, 3), "text": "".join(parts)})
        if sentences:
            return sentences

    raw_segments = transcript.get("segments") or []
    if not raw_segments:
        text = str(transcript.get("text", "")).strip()
        if not text:
            raise PipelineError("转写结果中没有文本。")
        return [{"segment_id": 0, "start": 0.0, "end": transcript.get("duration", 0), "text": text}]

    segments = []
    for index, item in enumerate(raw_segments):
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            {
                "segment_id": int(item.get("id", index)),
                "start": round(float(item.get("start", 0)), 3),
                "end": round(float(item.get("end", 0)), 3),
                "text": text,
            }
        )
    if not segments:
        raise PipelineError("转写结果中的所有分段均为空。")
    return segments


def load_glossary(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def translate_segments(
    segments: list[dict[str, Any]],
    model: str,
    glossary: list[dict[str, str]],
    api_key: str,
    context: str = "",
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "segment_id": {"type": "integer"},
                        "translation_zh": {"type": "string"},
                        "fact_flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["segment_id", "translation_zh", "fact_flags"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["segments"],
        "additionalProperties": False,
    }
    payload = {
        "model": model,
        "store": False,
        "instructions": (
            "你是一名国际新闻编辑和中英翻译。把英文采访或讲话忠实翻译成自然、简洁、适合播客口播的简体中文。"
            "不得补充原文没有的信息，不得淡化或强化原意；数字、日期、人名、机构名和政治口号必须准确。"
            "结合全部分段理解上下文，但逐段返回并保持 segment_id 不变。"
            "无法确认的专名不要猜测，在 fact_flags 中写明需要人工核对的项目；没有疑点时返回空数组。"
        ),
        "input": json.dumps(
            {
                "context": context,
                "glossary": glossary,
                "segments": segments,
            },
            ensure_ascii=False,
        ),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "journalism_translation",
                "strict": True,
                "schema": schema,
            }
        },
    }
    response = api_request_json("/responses", payload, api_key)
    output_text = extract_output_text(response)
    try:
        return json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise PipelineError(f"翻译接口没有返回有效 JSON：{output_text[:500]}") from exc


def extract_output_text(response: dict[str, Any]) -> str:
    direct = response.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    for item in response.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
            refusal = content.get("refusal")
            if refusal:
                raise PipelineError(f"翻译请求被模型拒绝：{refusal}")
    raise PipelineError("无法从 Responses API 返回值中找到文本。")


def build_review_rows(
    segments: list[dict[str, Any]], translation: dict[str, Any]
) -> list[dict[str, str]]:
    translations = {
        int(item["segment_id"]): item for item in translation.get("segments", [])
    }
    rows: list[dict[str, str]] = []
    for segment in segments:
        segment_id = int(segment["segment_id"])
        translated = translations.get(segment_id)
        if translated is None:
            raise PipelineError(f"中文翻译缺少 segment_id={segment_id}。")
        flags = translated.get("fact_flags", [])
        rows.append(
            {
                "segment_id": str(segment_id),
                "start_sec": f'{float(segment["start"]):.3f}',
                "end_sec": f'{float(segment["end"]):.3f}',
                "duration_sec": f'{float(segment["end"]) - float(segment["start"]):.3f}',
                "english_transcript": str(segment["text"]),
                "editor_english_transcript": "",
                "ai_translation_zh": str(translated["translation_zh"]),
                "editor_translation_zh": "",
                "review_status": "needs_review",
                "editor_notes": "；".join(str(flag) for flag in flags),
            }
        )
    return rows


def write_review_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_review_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise PipelineError("审校表为空。")
    missing = set(REVIEW_FIELDS) - set(rows[0])
    if missing:
        raise PipelineError(f"审校表缺少字段：{sorted(missing)}")
    return rows


def selected_translation(row: dict[str, str]) -> str:
    return row["editor_translation_zh"].strip() or row["ai_translation_zh"].strip()


def build_synthesis_script(
    review_path: Path,
    allow_unreviewed_demo: bool,
    include_disclosure: bool,
) -> tuple[str, list[str]]:
    rows = read_review_csv(review_path)
    pending = [
        row["segment_id"]
        for row in rows
        if row["review_status"].strip().lower() != "approved"
    ]
    if pending and not allow_unreviewed_demo:
        raise PipelineError(
            "人工审校尚未完成。请逐行填写 editor_translation_zh，"
            "并把 review_status 改为 approved。待审 segment_id：" + ", ".join(pending)
        )
    translations = [selected_translation(row) for row in rows]
    if any(not text for text in translations):
        raise PipelineError("至少有一行没有可用于合成的中文文本。")
    script = "\n".join(translations)
    if include_disclosure:
        script = "提示：以下中文配音由人工智能生成。\n" + script
    return script, pending


def save_synthesis(
    output_path: Path,
    script: str,
    audio: bytes,
    review_path: Path,
    pending: list[str],
    allow_unreviewed_demo: bool,
    include_disclosure: bool,
    audit_details: dict[str, Any],
) -> None:
    if len(audio) < 1_000:
        raise PipelineError("语音服务没有返回有效音频。")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".txt").write_text(script, encoding="utf-8")
    output_path.write_bytes(audio)
    append_audit(
        output_path.parent / "audit.jsonl",
        {
            "event": "synthesize",
            "review_file": str(review_path),
            "output_file": str(output_path),
            "human_approved": not pending,
            "demo_override": bool(pending and allow_unreviewed_demo),
            "disclosure_included": include_disclosure,
            **audit_details,
        },
    )


def synthesize(
    review_path: Path,
    output_path: Path,
    model: str,
    voice: str,
    allow_unreviewed_demo: bool,
    include_disclosure: bool,
    api_key: str,
) -> None:
    script, pending = build_synthesis_script(
        review_path, allow_unreviewed_demo, include_disclosure
    )
    audio = api_request_binary(
        "/audio/speech",
        {
            "model": model,
            "voice": voice,
            "input": script,
            "instructions": (
                "使用清晰、克制、可信的普通话新闻播报语气。保持事实性表达，不模仿任何真实人物。"
                "人名和数字要清楚，句间保留自然停顿。"
            ),
            "response_format": "mp3",
            "speed": 1.0,
        },
        api_key,
    )
    save_synthesis(
        output_path,
        script,
        audio,
        review_path,
        pending,
        allow_unreviewed_demo,
        include_disclosure,
        {
            "tts_backend": "openai",
            "tts_model": model,
            "voice": voice,
        },
    )


def synthesize_minimax(
    review_path: Path,
    output_path: Path,
    model: str,
    voice: str,
    speed: float,
    allow_unreviewed_demo: bool,
    include_disclosure: bool,
    api_key: str,
) -> None:
    script, pending = build_synthesis_script(
        review_path, allow_unreviewed_demo, include_disclosure
    )
    response_bytes = post_json(
        MINIMAX_TTS_URL,
        {
            "model": model,
            "text": script,
            "stream": False,
            "language_boost": "Chinese",
            "text_normalization_mode": "quality",
            "output_format": "hex",
            "voice_setting": {
                "voice_id": voice,
                "speed": speed,
                "vol": 1,
                "pitch": 0,
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1,
            },
        },
        {"Authorization": f"Bearer {api_key}"},
    )
    try:
        response = json.loads(response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineError("MiniMax 返回了无法解析的响应。") from exc
    base_resp = response.get("base_resp") or {}
    if base_resp.get("status_code") not in (None, 0):
        raise PipelineError(
            f"MiniMax 合成失败：{base_resp.get('status_msg', '未知错误')}"
        )
    audio_hex = (response.get("data") or {}).get("audio")
    if not isinstance(audio_hex, str):
        raise PipelineError("MiniMax 响应中没有音频数据。")
    try:
        audio = bytes.fromhex(audio_hex)
    except ValueError as exc:
        raise PipelineError("MiniMax 返回的音频不是有效的十六进制数据。") from exc
    save_synthesis(
        output_path,
        script,
        audio,
        review_path,
        pending,
        allow_unreviewed_demo,
        include_disclosure,
        {
            "tts_backend": "minimax",
            "tts_model": model,
            "voice": voice,
            "speed": speed,
        },
    )


def synthesize_elevenlabs(
    review_path: Path,
    output_path: Path,
    model: str,
    voice_id: str,
    allow_unreviewed_demo: bool,
    include_disclosure: bool,
    api_key: str,
) -> None:
    if not voice_id.strip():
        raise PipelineError(
            "ElevenLabs 需要 voice_id。请设置 ELEVENLABS_VOICE_ID 或传入 --voice-id。"
        )
    script, pending = build_synthesis_script(
        review_path, allow_unreviewed_demo, include_disclosure
    )
    audio = post_json(
        f"{ELEVENLABS_TTS_URL}/{quote(voice_id, safe='')}?output_format=mp3_44100_128",
        {"text": script, "model_id": model, "language_code": "zh"},
        {"xi-api-key": api_key},
    )
    save_synthesis(
        output_path,
        script,
        audio,
        review_path,
        pending,
        allow_unreviewed_demo,
        include_disclosure,
        {
            "tts_backend": "elevenlabs",
            "tts_model": model,
            "voice_id": voice_id,
        },
    )


def synthesize_local(
    review_path: Path,
    output_path: Path,
    voice: str,
    allow_unreviewed_demo: bool,
    include_disclosure: bool,
) -> None:
    if shutil.which("say") is None or shutil.which("ffmpeg") is None:
        raise PipelineError("本地合成需要 macOS say 和 ffmpeg。")
    script, pending = build_synthesis_script(
        review_path, allow_unreviewed_demo, include_disclosure
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".txt").write_text(script, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="journalism-ai-tts-") as temp_dir:
        aiff_path = Path(temp_dir) / "speech.aiff"
        subprocess.run(
            ["say", "-v", voice, "-o", str(aiff_path), script],
            check=True,
        )
        if not aiff_path.exists() or aiff_path.stat().st_size < 1_000:
            raise PipelineError(
                "macOS say 没有生成有效音频；请确认所选声音已安装，"
                "并允许终端访问系统语音服务。"
            )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(aiff_path),
                "-ac",
                "1",
                "-ar",
                "44100",
                "-b:a",
                "96k",
                str(output_path),
            ],
            check=True,
        )
    if not output_path.exists() or output_path.stat().st_size < 1_000:
        raise PipelineError("ffmpeg 没有生成有效的 MP3 文件。")
    append_audit(
        output_path.parent / "audit.jsonl",
        {
            "event": "synthesize_local",
            "review_file": str(review_path),
            "output_file": str(output_path),
            "tts_backend": "macos_say",
            "voice": voice,
            "human_approved": not pending,
            "demo_override": bool(pending and allow_unreviewed_demo),
            "disclosure_included": include_disclosure,
        },
    )


def append_audit(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def command_prepare(args: argparse.Namespace) -> None:
    api_key = require_api_key()
    audio_path = args.input.resolve()
    if not audio_path.exists():
        raise PipelineError(f"找不到输入音频：{audio_path}")
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    print("1/3 正在转写英文音频……", flush=True)
    transcript = transcribe(audio_path, args.transcription_model, api_key)
    transcript_path = work_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    segments = extract_segments(transcript)

    print("2/3 正在生成新闻级中文初译……", flush=True)
    glossary = load_glossary(args.glossary)
    translation = translate_segments(
        segments, args.translation_model, glossary, api_key, args.context
    )
    translation_path = work_dir / "translation_draft.json"
    translation_path.write_text(
        json.dumps(translation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("3/3 正在创建人工审校表……", flush=True)
    review_path = work_dir / "review.csv"
    write_review_csv(review_path, build_review_rows(segments, translation))
    append_audit(
        work_dir / "audit.jsonl",
        {
            "event": "prepare",
            "source_audio": str(audio_path),
            "transcription_model": args.transcription_model,
            "translation_model": args.translation_model,
            "segment_count": len(segments),
            "review_status": "needs_human_review",
        },
    )
    print(f"\n已暂停等待人工审校：{review_path}")
    print("请填写 editor_translation_zh，并把确认无误的每一行 review_status 改为 approved。")


def command_synthesize(args: argparse.Namespace) -> None:
    synthesize(
        args.review.resolve(),
        args.output.resolve(),
        args.tts_model,
        args.voice,
        args.allow_unreviewed_demo,
        not args.no_disclosure,
        require_api_key(),
    )
    print(f"中文配音已生成：{args.output.resolve()}")


def command_synthesize_local(args: argparse.Namespace) -> None:
    synthesize_local(
        args.review.resolve(),
        args.output.resolve(),
        args.voice,
        args.allow_unreviewed_demo,
        not args.no_disclosure,
    )
    print(f"本地中文配音已生成：{args.output.resolve()}")


def command_synthesize_minimax(args: argparse.Namespace) -> None:
    synthesize_minimax(
        args.review.resolve(),
        args.output.resolve(),
        args.tts_model,
        args.voice,
        args.speed,
        args.allow_unreviewed_demo,
        not args.no_disclosure,
        require_env_key("MINIMAX_API_KEY"),
    )
    print(f"MiniMax 中文配音已生成：{args.output.resolve()}")


def command_synthesize_elevenlabs(args: argparse.Namespace) -> None:
    synthesize_elevenlabs(
        args.review.resolve(),
        args.output.resolve(),
        args.tts_model,
        args.voice_id,
        args.allow_unreviewed_demo,
        not args.no_disclosure,
        require_env_key("ELEVENLABS_API_KEY"),
    )
    print(f"ElevenLabs 中文配音已生成：{args.output.resolve()}")


def command_import_draft(args: argparse.Namespace) -> None:
    transcript_path = args.transcript.resolve()
    translation_path = args.translation.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    translation = json.loads(translation_path.read_text(encoding="utf-8"))
    segments = extract_segments(transcript)
    review_path = work_dir / "review.csv"
    write_review_csv(review_path, build_review_rows(segments, translation))
    append_audit(
        work_dir / "audit.jsonl",
        {
            "event": "import_draft",
            "transcript_file": str(transcript_path),
            "translation_file": str(translation_path),
            "segment_count": len(segments),
            "review_status": "needs_human_review",
        },
    )
    print(f"已生成并暂停等待人工审校：{review_path}")


def command_status(args: argparse.Namespace) -> None:
    rows = read_review_csv(args.review.resolve())
    approved = sum(row["review_status"].strip().lower() == "approved" for row in rows)
    with_editor_text = sum(bool(row["editor_translation_zh"].strip()) for row in rows)
    print(f"总分段：{len(rows)}")
    print(f"已填写人工译文：{with_editor_text}/{len(rows)}")
    print(f"已批准：{approved}/{len(rows)}")
    print("状态：可正式合成" if approved == len(rows) else "状态：等待人工审校")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="转写、翻译并生成审校 CSV")
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("--work-dir", type=Path, required=True)
    prepare_parser.add_argument("--glossary", type=Path, default=Path("config/glossary.csv"))
    prepare_parser.add_argument("--transcription-model", default="whisper-1")
    prepare_parser.add_argument("--translation-model", default="gpt-5-mini")
    prepare_parser.add_argument("--context", default="", help="采访背景，供翻译参考")
    prepare_parser.set_defaults(func=command_prepare)

    import_parser = subparsers.add_parser(
        "import-draft", help="把本地转写与翻译草稿导入人工审校表"
    )
    import_parser.add_argument("--transcript", type=Path, required=True)
    import_parser.add_argument("--translation", type=Path, required=True)
    import_parser.add_argument("--work-dir", type=Path, required=True)
    import_parser.set_defaults(func=command_import_draft)

    synth_parser = subparsers.add_parser("synthesize", help="读取审校 CSV 并合成中文音频")
    synth_parser.add_argument("--review", type=Path, required=True)
    synth_parser.add_argument("--output", type=Path, required=True)
    synth_parser.add_argument("--tts-model", default="gpt-4o-mini-tts")
    synth_parser.add_argument("--voice", default="cedar")
    synth_parser.add_argument("--allow-unreviewed-demo", action="store_true")
    synth_parser.add_argument("--no-disclosure", action="store_true")
    synth_parser.set_defaults(func=command_synthesize)

    local_synth_parser = subparsers.add_parser(
        "synthesize-local", help="使用 macOS 系统声音生成中文演示音频"
    )
    local_synth_parser.add_argument("--review", type=Path, required=True)
    local_synth_parser.add_argument("--output", type=Path, required=True)
    local_synth_parser.add_argument("--voice", default="Reed (中文（中国大陆）)")
    local_synth_parser.add_argument("--allow-unreviewed-demo", action="store_true")
    local_synth_parser.add_argument("--no-disclosure", action="store_true")
    local_synth_parser.set_defaults(func=command_synthesize_local)

    minimax_parser = subparsers.add_parser(
        "synthesize-minimax", help="使用 MiniMax Speech 生成高质量中文配音"
    )
    minimax_parser.add_argument("--review", type=Path, required=True)
    minimax_parser.add_argument("--output", type=Path, required=True)
    minimax_parser.add_argument("--tts-model", default="speech-2.8-hd")
    minimax_parser.add_argument("--voice", default="male-qn-qingse")
    minimax_parser.add_argument("--speed", type=float, default=0.96)
    minimax_parser.add_argument("--allow-unreviewed-demo", action="store_true")
    minimax_parser.add_argument("--no-disclosure", action="store_true")
    minimax_parser.set_defaults(func=command_synthesize_minimax)

    elevenlabs_parser = subparsers.add_parser(
        "synthesize-elevenlabs", help="使用 ElevenLabs 生成高表现力中文配音"
    )
    elevenlabs_parser.add_argument("--review", type=Path, required=True)
    elevenlabs_parser.add_argument("--output", type=Path, required=True)
    elevenlabs_parser.add_argument("--tts-model", default="eleven_v4")
    elevenlabs_parser.add_argument(
        "--voice-id", default=os.environ.get("ELEVENLABS_VOICE_ID", "")
    )
    elevenlabs_parser.add_argument("--allow-unreviewed-demo", action="store_true")
    elevenlabs_parser.add_argument("--no-disclosure", action="store_true")
    elevenlabs_parser.set_defaults(func=command_synthesize_elevenlabs)

    status_parser = subparsers.add_parser("status", help="检查审校是否完成")
    status_parser.add_argument("--review", type=Path, required=True)
    status_parser.set_defaults(func=command_status)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
        return 0
    except PipelineError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
