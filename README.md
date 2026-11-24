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
export OPENAI_API_KEY='your-key'
# Configure a TTS provider to generate MP3 audio:
export MINIMAX_API_KEY='your-key'
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/app.html`. Use **Open local development sample** with the access code to review the existing clip without new transcription costs. Uploading new audio requires `OPENAI_API_KEY`; generating an MP3 requires a configured TTS provider. API keys stay on the server.

## Architecture and limits

- `app.py`: FastAPI endpoints, session review gate, audit export, and static pages.
- `pipeline.py`: transcription, translation, review CSV, and voice adapters.
- `docs/`: English landing page, review workspace, and policies.
- `config/glossary.csv`: default newsroom terminology.
- `tests/`: pipeline and API checks.

The live prototype handles short, single-speaker clips. Review sessions are held in server memory and disappear on restart. There is no account system, database, or production rate limiting. The GitHub Pages demo is static and cannot call the Python API. Do not upload confidential or uncleared recordings to third-party providers.

## GitHub Pages

The workflow at `.github/workflows/pages.yml` publishes `docs/` after pushes to `main`. In the GitHub repository, enable **Settings → Pages → Build and deployment → GitHub Actions**. Then push the source to `main`. GitHub Pages will serve the classroom demo; live API processing needs separate Python hosting.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

[中文说明](README.zh-CN.md)
