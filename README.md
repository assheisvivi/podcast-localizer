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
# Recommended: Groq handles transcription and translation.
export AI_PROVIDER='groq'
export GROQ_API_KEY='your-groq-key'
# Add one voice key for downloadable Chinese MP3 output.
export MINIMAX_API_KEY='your-minimax-key'
# Alternatively, use ELEVENLABS_API_KEY plus ELEVENLABS_VOICE_ID.
# Optional: set daily paid-call caps (defaults: 20 preparations, 10 dubs)
export MAX_PREPARES_PER_DAY=20
export MAX_SYNTHESIS_PER_DAY=10
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/app.html`. Use **Open local development sample** with the access code to review the existing clip without new transcription costs. With `AI_PROVIDER=groq` and `GROQ_API_KEY`, new uploads use Whisper Large v3 for timestamped transcription and Qwen 3.8 27B for structured Chinese translation. `GROQ_TRANSCRIPTION_MODEL` and `GROQ_TRANSLATION_MODEL` override those defaults. Groq's free tier is rate limited and access depends on your account. After approving every line, the default Edge voice generates a downloadable Mandarin MP3 without a TTS key (`zh-CN-YunxiNeural`, or enter `zh-CN-XiaoxiaoNeural`). This uses Microsoft's online speech service through the unofficial `edge-tts` library, so availability can change. If that service fails, the page offers a browser voice preview without an MP3. MiniMax, ElevenLabs, OpenRouter and OpenAI remain optional voice providers. API keys stay on the server.

## Architecture and limits

- `app.py`: FastAPI endpoints, session review gate, audit export, and static pages.
- `pipeline.py`: transcription, translation, review CSV, and voice adapters.
- `review_store.py`: local SQLite or hosted PostgreSQL review sessions, edit history, and atomic daily quotas.
- `docs/`: English landing page, review workspace, and policies.
- `config/glossary.csv`: default newsroom terminology.
- `tests/`: pipeline and API checks.

The live prototype handles short, single-speaker clips. A shared access code protects paid API routes. The database saves review text and edit history for seven days; uploaded audio is discarded after processing and must be reattached locally to listen after resuming a session. Paid calls have database-backed daily caps. This remains an invited-team pilot, without individual accounts or per-user attribution. The GitHub Pages demo is static and cannot call the Python API. Do not upload confidential or uncleared recordings to third-party providers.

## Hosted pilot

`render.yaml` describes a **Free** Render Python web service. Create a Free Neon PostgreSQL project and enter its connection string as Render's secret `DATABASE_URL`; also set a long random `DEMO_ACCESS_CODE`. The deployment refuses to start without the persistent database, so review records and daily caps survive Render's free-service sleep and restarts. Set `AI_PROVIDER=groq`, then add `GROQ_API_KEY` in Render's Environment page. This explicit provider choice prevents a fallback to an old OpenRouter key when Groq is not configured. No `MINIMAX_API_KEY` is needed for the default free Edge MP3 voice. A working Groq key and available quota are required for live processing. Free hosting has quotas and cold starts. Never commit database URLs, access codes, or API keys to GitHub.

## GitHub Pages

The workflow at `.github/workflows/pages.yml` publishes `docs/` after pushes to `main`. In the GitHub repository, enable **Settings → Pages → Build and deployment → GitHub Actions**. Then push the source to `main`. GitHub Pages will serve the classroom demo; live API processing needs separate Python hosting.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

[中文说明](README.zh-CN.md)
