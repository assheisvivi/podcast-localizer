import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline


class PipelineTests(unittest.TestCase):
    def write_approved_review(self, directory: str) -> Path:
        row = {field: "" for field in pipeline.REVIEW_FIELDS}
        row.update(
            {
                "segment_id": "0",
                "ai_translation_zh": "初译。",
                "editor_translation_zh": "编辑确认后的中文。",
                "review_status": "approved",
            }
        )
        path = Path(directory) / "review.csv"
        pipeline.write_review_csv(path, [row])
        return path

    def test_extract_segments_drops_empty_segments(self):
        result = pipeline.extract_segments(
            {
                "segments": [
                    {"id": 0, "start": 0, "end": 1, "text": " Hello. "},
                    {"id": 1, "start": 1, "end": 2, "text": ""},
                ]
            }
        )
        self.assertEqual(result, [{"segment_id": 0, "start": 0.0, "end": 1.0, "text": "Hello."}])

    def test_word_timestamps_repair_pause_boundaries_and_keep_sentence_start(self):
        words = [
            ("The", 0, .2), ("U.S.", .2, .6), ("economy", .7, 1),
            ("grew.", 1, 1.3), ("It", 1.8, 2), ("slowed", 2, 2.3),
            ("later.", 2.3, 2.7),
        ]
        transcript = {"words": [{"word": word, "start": start, "end": end} for word, start, end in words],
                      "segments": [{"id": 0, "start": 0, "end": .6, "text": "The U.S."},
                                   {"id": 1, "start": .7, "end": 2.7, "text": "economy grew. It slowed later."}]}
        self.assertEqual(pipeline.extract_segments(transcript), [
            {"segment_id": 0, "start": 0.0, "end": 1.3, "text": "The U.S. economy grew."},
            {"segment_id": 1, "start": 1.8, "end": 2.7, "text": "It slowed later."},
        ])

    def test_transcribe_requests_word_timestamps(self):
        with patch.object(pipeline, "api_request_multipart", return_value={}) as request:
            pipeline.transcribe(Path("clip.mp3"), "whisper-1", "sk-test")
        self.assertIn(("timestamp_granularities[]", "word"), request.call_args.args[1])

    def test_openrouter_translation_keeps_segment_ids_and_strict_schema(self):
        segments = [{"segment_id": 3, "start": 0, "end": 2, "text": "Hello."}]
        answer = {"choices": [{"message": {"content": json.dumps({"segments": [
            {"segment_id": 3, "translation_zh": "你好。", "fact_flags": []}
        ]})}}]}
        with patch.object(pipeline, "post_json", return_value=json.dumps(answer).encode()) as request:
            result = pipeline.translate_segments_openrouter(segments, "openai/gpt-6.2-sol", [], "key")
        self.assertEqual(result["segments"][0]["translation_zh"], "你好。")
        self.assertEqual(request.call_args.args[0], pipeline.OPENROUTER_API_BASE + "/chat/completions")
        self.assertTrue(request.call_args.args[1]["provider"]["require_parameters"])
        self.assertEqual(request.call_args.args[1]["response_format"]["type"], "json_schema")

    def test_openrouter_translation_rejects_changed_segment_id(self):
        segments = [{"segment_id": 3, "start": 0, "end": 2, "text": "Hello."}]
        answer = {"choices": [{"message": {"content": json.dumps({"segments": [
            {"segment_id": 4, "translation_zh": "你好。", "fact_flags": []}
        ]})}}]}
        with patch.object(pipeline, "post_json", return_value=json.dumps(answer).encode()):
            with self.assertRaises(pipeline.PipelineError):
                pipeline.translate_segments_openrouter(segments, "openai/gpt-6.2-sol", [], "key")

    def test_build_review_rows_starts_unapproved(self):
        rows = pipeline.build_review_rows(
            [{"segment_id": 3, "start": 2, "end": 4, "text": "Example."}],
            {
                "segments": [
                    {"segment_id": 3, "translation_zh": "示例。", "fact_flags": []}
                ]
            },
        )
        self.assertEqual(rows[0]["review_status"], "needs_review")
        self.assertEqual(rows[0]["editor_translation_zh"], "")

    def test_editor_translation_has_priority(self):
        row = {"ai_translation_zh": "初译", "editor_translation_zh": "终稿"}
        self.assertEqual(pipeline.selected_translation(row), "终稿")

    def test_invalid_api_key_is_rejected_without_exposing_value(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "请填写密钥"}, clear=False):
            with self.assertRaises(pipeline.PipelineError):
                pipeline.require_api_key()

    def test_review_csv_round_trip(self):
        row = {field: "" for field in pipeline.REVIEW_FIELDS}
        row.update({"segment_id": "0", "review_status": "needs_review"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            pipeline.write_review_csv(path, [row])
            loaded = pipeline.read_review_csv(path)
        self.assertEqual(loaded[0]["segment_id"], "0")

    def test_minimax_adapter_decodes_hex_audio(self):
        fake_audio = b"ID3" + b"x" * 1_100
        response = {
            "data": {"audio": fake_audio.hex(), "status": 2},
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        with tempfile.TemporaryDirectory() as directory:
            review = self.write_approved_review(directory)
            output = Path(directory) / "minimax.mp3"
            with patch(
                "pipeline.post_json", return_value=json.dumps(response).encode("utf-8")
            ):
                pipeline.synthesize_minimax(
                    review,
                    output,
                    "speech-2.8-hd",
                    "male-qn-qingse",
                    0.96,
                    False,
                    True,
                    "key",
                )
            self.assertEqual(output.read_bytes(), fake_audio)
            self.assertIn("人工智能生成", output.with_suffix(".txt").read_text(encoding="utf-8"))

    def test_elevenlabs_adapter_writes_audio(self):
        fake_audio = b"ID3" + b"y" * 1_100
        with tempfile.TemporaryDirectory() as directory:
            review = self.write_approved_review(directory)
            output = Path(directory) / "elevenlabs.mp3"
            with patch("pipeline.post_json", return_value=fake_audio) as request:
                pipeline.synthesize_elevenlabs(
                    review, output, "eleven_v4", "voice/id", False, True, "key"
                )
            self.assertEqual(output.read_bytes(), fake_audio)
            self.assertIn("voice%2Fid", request.call_args.args[0])

    def test_openrouter_voice_generates_approved_mp3(self):
        fake_audio = b"ID3" + b"z" * 1_100
        with tempfile.TemporaryDirectory() as directory:
            review = self.write_approved_review(directory)
            output = Path(directory) / "openrouter.mp3"
            with patch.object(pipeline, "_open", return_value=fake_audio) as request:
                pipeline.synthesize_openrouter(
                    review, output, "minimax/speech-2.8-hd", "male-qn-qingse", False, True, "key"
                )
            self.assertEqual(output.read_bytes(), fake_audio)
            self.assertEqual(request.call_args.args[0].full_url, pipeline.OPENROUTER_API_BASE + "/audio/speech")

    def test_provider_key_is_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(pipeline.PipelineError):
                pipeline.require_env_key("MINIMAX_API_KEY")


if __name__ == "__main__":
    unittest.main()
