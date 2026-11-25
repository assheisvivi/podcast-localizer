import asyncio
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

import app as web
import review_store_postgres


def review_row(status="needs_review", final=""):
    return web.ReviewRow(
        segment_id="0", start_sec="0.000", end_sec="3.000", duration_sec="3.000",
        english_transcript="Hello.", ai_translation_zh="你好。",
        editor_translation_zh=final, review_status=status,
    )


class WebApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.database_env = patch.dict(os.environ, {"SESSION_DB_PATH": os.path.join(self.directory.name, "reviews.sqlite3")})
        self.database_env.start()

    def tearDown(self):
        self.database_env.stop()
        self.directory.cleanup()

    def test_access_code_is_required(self):
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret"}):
            with self.assertRaises(HTTPException) as caught:
                web.require_demo_access("wrong")
        self.assertEqual(caught.exception.status_code, 401)

    def test_health_marks_live_upload_unconfigured_without_api_key(self):
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "OPENAI_API_KEY": "", "OPENROUTER_API_KEY": ""}):
            self.assertFalse(web.health()["processing_enabled"])

    def test_openrouter_key_marks_live_processing_configured(self):
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "OPENAI_API_KEY": "", "OPENROUTER_API_KEY": "sk-or-test"}):
            health = web.health()
        self.assertTrue(health["processing_enabled"])
        self.assertTrue(health["tts_backends"]["openrouter"])

    def test_prepare_routes_to_openrouter_when_configured(self):
        audio = UploadFile(file=io.BytesIO(b"a" * 2000), filename="clip.mp3")
        transcript = {"segments": [{"id": 0, "start": 0, "end": 3, "text": "Hello."}]}
        translation = {"segments": [{"segment_id": 0, "translation_zh": "你好。", "fact_flags": []}]}
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "OPENROUTER_API_KEY": "sk-or-test"}), \
             patch.object(web.pipeline, "transcribe", return_value=transcript) as transcribe, \
             patch.object(web.pipeline, "translate_segments_openrouter", return_value=translation) as translate:
            result = asyncio.run(web.prepare(audio=audio, context="", glossary_json="[]", authorized=True, x_demo_access_code="secret"))
        self.assertEqual(result["segment_count"], 1)
        self.assertEqual(transcribe.call_args.args[1], "mistralai/voxtral-small-transcribe-3-0")
        self.assertEqual(transcribe.call_args.args[3], web.pipeline.OPENROUTER_API_BASE)
        self.assertEqual(translate.call_args.args[1], "qwen/qwen3.8-2.4t-a95b")

    def test_prepare_prefers_groq_over_openrouter_when_both_keys_exist(self):
        audio = UploadFile(file=io.BytesIO(b"a" * 2000), filename="clip.mp3")
        transcript = {"segments": [{"id": 0, "start": 0, "end": 3, "text": "Hello."}]}
        translation = {"segments": [{"segment_id": 0, "translation_zh": "你好。", "fact_flags": []}]}
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "GROQ_API_KEY": "gsk-test", "OPENROUTER_API_KEY": "sk-or-test"}), \
             patch.object(web.pipeline, "transcribe", return_value=transcript) as transcribe, \
             patch.object(web.pipeline, "translate_segments_groq", return_value=translation) as translate:
            result = asyncio.run(web.prepare(audio=audio, context="", glossary_json="[]", authorized=True, x_demo_access_code="secret"))
            health = web.health()
        self.assertEqual(result["segment_count"], 1)
        self.assertEqual(health["ai_provider"], "groq")
        self.assertEqual(transcribe.call_args.args[1], "whisper-large-v3")
        self.assertEqual(transcribe.call_args.args[3], web.pipeline.GROQ_API_BASE)
        self.assertEqual(translate.call_args.args[1], "qwen/qwen3.8-27b")

    def test_explicit_groq_selection_does_not_fall_back_to_openrouter(self):
        with patch.dict(os.environ, {"AI_PROVIDER": "groq", "GROQ_API_KEY": "", "OPENROUTER_API_KEY": "sk-or-test", "DEMO_ACCESS_CODE": "secret"}):
            self.assertFalse(web.health()["processing_enabled"])
            self.assertEqual(web.ai_provider(), "groq")

    def test_openrouter_model_overrides(self):
        audio = UploadFile(file=io.BytesIO(b"a" * 2000), filename="clip.mp3")
        transcript = {"segments": [{"id": 0, "start": 0, "end": 3, "text": "Hello."}]}
        translation = {"segments": [{"segment_id": 0, "translation_zh": "你好。", "fact_flags": []}]}
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "OPENROUTER_API_KEY": "sk-or-test", "OPENROUTER_TRANSCRIPTION_MODEL": "vendor/stt", "OPENROUTER_TRANSLATION_MODEL": "vendor/translate"}), \
             patch.object(web.pipeline, "transcribe", return_value=transcript) as transcribe, \
             patch.object(web.pipeline, "translate_segments_openrouter", return_value=translation) as translate:
            asyncio.run(web.prepare(audio=audio, context="", glossary_json="[]", authorized=True, x_demo_access_code="secret"))
        self.assertEqual(transcribe.call_args.args[1], "vendor/stt")
        self.assertEqual(translate.call_args.args[1], "vendor/translate")

    def test_provider_credit_and_access_errors_are_safe_and_actionable(self):
        cases = [("HTTP 402", 402, "no available credits"), ("HTTP 403", 403, "cannot use"), ("HTTP 401", 503, "invalid")]
        for code, expected_status, expected_message in cases:
            error = web.public_error(web.pipeline.PipelineError(f"API 请求失败（{code}）：secret provider detail"))
            self.assertEqual(error.status_code, expected_status)
            self.assertIn(expected_message, error.detail)
            self.assertNotIn("secret provider detail", error.detail)

    def test_prepare_returns_unapproved_rows_without_persisting_audio(self):
        audio = UploadFile(file=io.BytesIO(b"a" * 2000), filename="clip.mp3")
        transcript = {"segments": [{"id": 0, "start": 0, "end": 3, "text": "Hello."}]}
        translation = {"segments": [{"segment_id": 0, "translation_zh": "你好。", "fact_flags": []}]}
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "OPENAI_API_KEY": "sk-test", "OPENROUTER_API_KEY": ""}), \
             patch.object(web.pipeline, "transcribe", return_value=transcript), \
             patch.object(web.pipeline, "translate_segments", return_value=translation) as translate:
            result = asyncio.run(web.prepare(audio=audio, context="A short interview", glossary_json="[]", authorized=True, x_demo_access_code="secret"))
        self.assertEqual(result["segment_count"], 1)
        self.assertEqual(result["rows"][0]["review_status"], "needs_review")
        self.assertTrue(result["session_id"])
        self.assertEqual(translate.call_args.args[-1], "A short interview")

    def test_prepare_rejects_oversize_audio_before_provider_call(self):
        audio = UploadFile(file=io.BytesIO(b"a" * (web.MAX_AUDIO_BYTES + 1)), filename="clip.mp3")
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "OPENAI_API_KEY": "sk-test", "OPENROUTER_API_KEY": ""}), \
             patch.object(web.pipeline, "transcribe") as transcribe:
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(web.prepare(audio=audio, context="", glossary_json="[]", authorized=True, x_demo_access_code="secret"))
        self.assertEqual(caught.exception.status_code, 413)
        transcribe.assert_not_called()

    def test_synthesis_requires_explicit_final_approval(self):
        session = web.create_session([review_row().model_dump()], [])
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret"}):
            with self.assertRaises(HTTPException) as caught:
                web.synthesize(web.SynthesizeRequest(session_id=session["session_id"], disclosure_confirmed=True), "secret")
        self.assertEqual(caught.exception.status_code, 400)

    def test_approved_rows_generate_downloadable_audio(self):
        def fake_synthesis(review_path, output_path, *args):
            output_path.write_bytes(b"ID3" + b"x" * 1200)

        session = web.create_session([review_row().model_dump()], [])
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "MINIMAX_API_KEY": "test"}), \
             patch.object(web.pipeline, "synthesize_minimax", side_effect=fake_synthesis):
            web.review(web.ReviewRequest(session_id=session["session_id"], rows=[review_row(status="approved", final="你好。")]), "secret")
            response = web.synthesize(
                web.SynthesizeRequest(session_id=session["session_id"], backend="minimax", disclosure_confirmed=True), "secret"
            )
        self.assertEqual(response.media_type, "audio/mpeg")
        self.assertTrue(response.body.startswith(b"ID3"))

    def test_approved_rows_can_use_openrouter_voice(self):
        def fake_synthesis(review_path, output_path, *args):
            output_path.write_bytes(b"ID3" + b"x" * 1200)

        session = web.create_session([review_row().model_dump()], [])
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "OPENROUTER_API_KEY": "sk-or-test"}), \
             patch.object(web.pipeline, "synthesize_openrouter", side_effect=fake_synthesis) as voice:
            web.review(web.ReviewRequest(session_id=session["session_id"], rows=[review_row(status="approved", final="你好。")]), "secret")
            response = web.synthesize(web.SynthesizeRequest(
                session_id=session["session_id"], backend="openrouter", disclosure_confirmed=True
            ), "secret")
        self.assertEqual(response.media_type, "audio/mpeg")
        self.assertEqual(voice.call_args.args[2:4], ("minimax/speech-2.8-hd", "male-qn-qingse"))

    def test_free_edge_mp3_needs_no_voice_key(self):
        def fake_synthesis(review_path, output_path, *args):
            output_path.write_bytes(b"ID3" + b"x" * 1200)

        session = web.create_session([review_row().model_dump()], [])
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "MINIMAX_API_KEY": ""}), \
             patch.object(web.pipeline, "synthesize_edge", side_effect=fake_synthesis) as voice:
            web.review(web.ReviewRequest(session_id=session["session_id"], rows=[review_row(status="approved", final="你好。")]), "secret")
            response = web.synthesize(web.SynthesizeRequest(session_id=session["session_id"], disclosure_confirmed=True), "secret")
        self.assertTrue(web.health()["tts_backends"]["edge"])
        self.assertEqual(response.media_type, "audio/mpeg")
        self.assertTrue(response.body.startswith(b"ID3"))
        self.assertEqual(voice.call_args.args[2], "zh-CN-YunxiNeural")

    def test_edge_failure_returns_browser_fallback_signal(self):
        session = web.create_session([review_row().model_dump()], [])
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret"}), \
             patch.object(web.pipeline, "synthesize_edge", side_effect=web.pipeline.PipelineError("Edge TTS unavailable.")):
            web.review(web.ReviewRequest(session_id=session["session_id"], rows=[review_row(status="approved", final="你好。")]), "secret")
            with self.assertRaises(HTTPException) as caught:
                web.synthesize(web.SynthesizeRequest(session_id=session["session_id"], disclosure_confirmed=True), "secret")
        self.assertEqual(caught.exception.status_code, 502)
        self.assertIn("browser preview", caught.exception.detail)

    def test_qa_flags_numbers_and_glossary(self):
        row = review_row().model_dump()
        row.update(english_transcript="The U.S. has 47 sanctions.", ai_translation_zh="美国有46项限制。")
        self.assertEqual(web.qa_flags(row, [{"english": "sanctions", "preferred_chinese": "制裁"}]), ["数字待核", "术语待核"])

    def test_sample_and_audit_export(self):
        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "sample.mp3"
            review_path = Path(directory) / "review.csv"
            audio_path.write_bytes(b"ID3" + b"x" * 100)
            row = review_row().model_dump()
            web.pipeline.write_review_csv(review_path, [{field: row.get(field, "") for field in web.pipeline.REVIEW_FIELDS}])
            with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret"}), \
                 patch.object(web, "SAMPLE_AUDIO", audio_path), \
                 patch.object(web, "SAMPLE_REVIEW", review_path):
                sample = web.sample("secret")
                self.assertEqual(sample["segment_count"], 1)
                self.assertTrue(all(row["review_status"] == "needs_review" for row in sample["rows"]))
                response = web.audit(sample["session_id"], "secret")
                self.assertIn("ai_translation_zh", response.body.decode("utf-8-sig"))

    def test_review_rejects_changed_ai_draft(self):
        session = web.create_session([review_row().model_dump()], [])
        changed = review_row(status="approved", final="你好。")
        changed.ai_translation_zh = "被替换的初译"
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret"}), self.assertRaises(HTTPException) as caught:
            web.review(web.ReviewRequest(session_id=session["session_id"], rows=[changed]), "secret")
        self.assertEqual(caught.exception.status_code, 400)

    def test_saved_review_and_edit_history_survive_new_database_connection(self):
        session = web.create_session([review_row().model_dump()], [])
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret"}):
            web.review(web.ReviewRequest(session_id=session["session_id"], rows=[review_row(status="approved", final="你好。")]), "secret")
            restored = web.resume_session(session["session_id"], "secret")
            history = web.review_history(session["session_id"], "secret")
        self.assertEqual(restored["rows"][0]["review_status"], "approved")
        self.assertEqual(history["events"][0]["after"]["editor_translation_zh"], "你好。")

    def test_daily_prepare_quota_blocks_provider_call(self):
        audio = UploadFile(file=io.BytesIO(b"a" * 2000), filename="clip.mp3")
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "OPENAI_API_KEY": "sk-test", "MAX_PREPARES_PER_DAY": "0"}), \
             patch.object(web.pipeline, "transcribe") as transcribe:
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(web.prepare(audio=audio, context="", glossary_json="[]", authorized=True, x_demo_access_code="secret"))
        self.assertEqual(caught.exception.status_code, 429)
        transcribe.assert_not_called()

    def test_daily_quota_is_persistent(self):
        self.assertTrue(web.review_store.consume_daily_quota("prepare", 1))
        self.assertFalse(web.review_store.consume_daily_quota("prepare", 1))

    def test_hosted_mode_requires_persistent_database(self):
        with patch.dict(os.environ, {"DATABASE_URL": "", "REQUIRE_PERSISTENT_STORE": "1"}):
            with self.assertRaisesRegex(RuntimeError, "DATABASE_URL"):
                web.review_store.ensure_ready()
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://example", "REQUIRE_PERSISTENT_STORE": "1"}):
            self.assertEqual(web.review_store.backend(), "postgres")

    def test_zero_postgres_quota_never_connects(self):
        with patch.object(review_store_postgres, "database") as database:
            self.assertFalse(review_store_postgres.consume_daily_quota("prepare", 0))
        database.assert_not_called()

    def test_daily_synthesis_quota_blocks_voice_provider(self):
        session = web.create_session([review_row().model_dump()], [])
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "MINIMAX_API_KEY": "test", "MAX_SYNTHESIS_PER_DAY": "0"}), \
             patch.object(web.pipeline, "synthesize_minimax") as voice:
            web.review(web.ReviewRequest(session_id=session["session_id"], rows=[review_row(status="approved", final="你好。")]), "secret")
            with self.assertRaises(HTTPException) as caught:
                web.synthesize(web.SynthesizeRequest(session_id=session["session_id"], backend="minimax", disclosure_confirmed=True), "secret")
        self.assertEqual(caught.exception.status_code, 429)
        voice.assert_not_called()

    def test_review_rejects_tampered_timestamps(self):
        session = web.create_session([review_row().model_dump()], [])
        changed = review_row(status="approved", final="你好。")
        changed.start_sec = "999.000"
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret"}), self.assertRaises(HTTPException) as caught:
            web.review(web.ReviewRequest(session_id=session["session_id"], rows=[changed]), "secret")
        self.assertEqual(caught.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
