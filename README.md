# Podcast Localizer

A human-gated English-to-Mandarin podcast localization prototype for news producers. AI prepares a transcript and translation, highlights lines that need attention, and stops before voice synthesis until an editor approves every segment.

## Classroom demo

The static site in `docs/` is designed for GitHub Pages. Open `docs/index.html` locally or publish the repository with the included Pages workflow. Select **Explore classroom demo** in the review workspace. The fictional sample needs no access code, API key, or backend. You can inspect risk flags, edit the Chinese script, approve each line, hear a browser voice preview, and export an audit CSV. Browser speech is a preview only; it does not produce an MP3.

The existing local recording under `input/` is excluded from GitHub. It is for development, not a public classroom sample.

## Live local workflow

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export DEMO_ACCESS_CODE='choose-a-long-random-access-code'
# One OpenRouter key can handle transcription, translation, and voice:
export OPENROUTER_API_KEY='your-key'
# Alternatively, configure OPENAI_API_KEY for transcription/translation
# and a matching TTS provider key.
# Optional: set daily paid-call caps (defaults: 20 preparations, 10 dubs)
export MAX_PREPARES_PER_DAY=20
export MAX_SYNTHESIS_PER_DAY=10
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/app.html`. Use **Open local development sample** with the access code to review the existing clip without new transcription costs. With `OPENROUTER_API_KEY`, new uploads use Whisper transcription and GPT-6.2 Sol translation; the default OpenRouter voice is MiniMax Speech 2.8 HD. Set `OPENROUTER_TTS_MODEL` and `OPENROUTER_TTS_VOICE` together to audition another supported voice model. Alternatively, use `OPENAI_API_KEY` and a separate TTS provider. API keys stay on the server.

## Architecture and limits

- `app.py`: FastAPI endpoints, session review gate, audit export, and static pages.
- `pipeline.py`: transcription, translation, review CSV, and voice adapters.
- `review_store.py`: local SQLite or hosted PostgreSQL review sessions, edit history, and atomic daily quotas.
- `docs/`: English landing page, review workspace, and policies.
- `config/glossary.csv`: default newsroom terminology.
- `tests/`: pipeline and API checks.

The live prototype handles short, single-speaker clips. A shared access code protects paid API routes. The database saves review text and edit history for seven days; uploaded audio is discarded after processing and must be reattached locally to listen after resuming a session. Paid calls have database-backed daily caps. This remains an invited-team pilot, without individual accounts or per-user attribution. The GitHub Pages demo is static and cannot call the Python API. Do not upload confidential or uncleared recordings to third-party providers.

## Hosted pilot

`render.yaml` describes a **Free** Render Python web service. Create a Free Neon PostgreSQL project and enter its connection string as Render's secret `DATABASE_URL`; also set a long random `DEMO_ACCESS_CODE`. The deployment refuses to start without the persistent database, so review records and daily caps survive Render's free-service sleep and restarts. Add `OPENROUTER_API_KEY` in Render's Environment page to enable transcription, translation, and the default voice with one key. OpenRouter speech calls use paid credit; a free account alone does not fund them. Alternatively, add `OPENAI_API_KEY` and optional TTS provider keys. Without an AI key the classroom demo still works, but real transcription does not. Free hosting has quotas and cold starts. Never commit database URLs, access codes, or API keys to GitHub.

## GitHub Pages

The workflow at `.github/workflows/pages.yml` publishes `docs/` after pushes to `main`. In the GitHub repository, enable **Settings → Pages → Build and deployment → GitHub Actions**. Then push the source to `main`. GitHub Pages will serve the classroom demo; live API processing needs separate Python hosting.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

[中文说明](README.zh-CN.md)
