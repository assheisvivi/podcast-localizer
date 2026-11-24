import asyncio
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

import app as web


def review_row(status="needs_review", final=""):
    return web.ReviewRow(
        segment_id="0", start_sec="0.000", end_sec="3.000", duration_sec="3.000",
        english_transcript="Hello.", ai_translation_zh="你好。",
        editor_translation_zh=final, review_status=status,
    )


class WebApiTests(unittest.TestCase):
    def test_access_code_is_required(self):
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret"}):
            with self.assertRaises(HTTPException) as caught:
                web.require_demo_access("wrong")
        self.assertEqual(caught.exception.status_code, 401)

    def test_prepare_returns_unapproved_rows_without_persisting_audio(self):
        audio = UploadFile(file=io.BytesIO(b"a" * 2000), filename="clip.mp3")
        transcript = {"segments": [{"id": 0, "start": 0, "end": 3, "text": "Hello."}]}
        translation = {"segments": [{"segment_id": 0, "translation_zh": "你好。", "fact_flags": []}]}
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "OPENAI_API_KEY": "sk-test"}), \
             patch.object(web.pipeline, "transcribe", return_value=transcript), \
             patch.object(web.pipeline, "translate_segments", return_value=translation) as translate:
            result = asyncio.run(web.prepare(audio=audio, context="A short interview", glossary_json="[]", authorized=True, x_demo_access_code="secret"))
        self.assertEqual(result["segment_count"], 1)
        self.assertEqual(result["rows"][0]["review_status"], "needs_review")
        self.assertTrue(result["session_id"])
        self.assertEqual(translate.call_args.args[-1], "A short interview")

    def test_prepare_rejects_oversize_audio_before_provider_call(self):
        audio = UploadFile(file=io.BytesIO(b"a" * (web.MAX_AUDIO_BYTES + 1)), filename="clip.mp3")
        with patch.dict(os.environ, {"DEMO_ACCESS_CODE": "secret", "OPENAI_API_KEY": "sk-test"}), \
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
                web.SynthesizeRequest(session_id=session["session_id"], disclosure_confirmed=True), "secret"
            )
        self.assertEqual(response.media_type, "audio/mpeg")
        self.assertTrue(response.body.startswith(b"ID3"))

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


if __name__ == "__main__":
    unittest.main()
