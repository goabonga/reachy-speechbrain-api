---
title: Reachy SpeechBrain API
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Reachy SpeechBrain API

**FastAPI-based Speaker Recognition API for Reachy robots.**

Reachy SpeechBrain API is a lightweight speaker recognition service built with **FastAPI** and **SpeechBrain**, designed to run on **Hugging Face Spaces (Docker)** or locally, and to be easily integrated with **Reachy robots** or any backend.

---

## Features

- 🎤 Speaker recognition powered by **SpeechBrain ECAPA-TDNN**
- 👤 Speaker enrollment, identification, and verification
- ⚡ FastAPI HTTP API (simple & stateless)
- 🐳 Hugging Face **Docker Space** compatible
- 🧠 CPU-friendly speaker embeddings
- 🤖 Ready to integrate with **Reachy Mini**
- 📦 Dependency management with **uv**
- 🗄️ Flexible storage: local or **Hugging Face Hub** dataset

---

## API Endpoints

### Health check
```
GET /health
```

Response:
```json
{ "status": "ok" }
```

---

### List speakers
```
GET /speakers
```

Response:
```json
{
  "speakers": ["alice", "bob", "charlie"]
}
```

---

### Enroll a speaker
```
POST /speakers/{name}/enroll
```

**Request**
- `multipart/form-data`
- Field: `file` (audio file: WAV, MP3, FLAC, etc.)

**Example**
```bash
curl -X POST \
  -F "file=@voice_sample.wav" \
  http://localhost:7860/speakers/alice/enroll
```

**Response**
```json
{
  "message": "Speaker 'alice' enrolled successfully",
  "embedding_size": 192
}
```

---

### Delete a speaker
```
DELETE /speakers/{name}
```

**Example**
```bash
curl -X DELETE http://localhost:7860/speakers/alice
```

**Response**
```json
{
  "message": "Speaker 'alice' deleted successfully"
}
```

---

### Identify speaker
```
POST /identify
```

Identifies who is speaking from the enrolled speakers.

**Request**
- `multipart/form-data`
- Field: `file` (audio file)

**Example**
```bash
curl -X POST \
  -F "file=@unknown_voice.wav" \
  http://localhost:7860/identify
```

**Response**
```json
{
  "identified": true,
  "speaker": "alice",
  "confidence": 0.85,
  "threshold": 0.25
}
```

---

### Verify speaker
```
POST /verify?name={speaker_name}
```

Verifies if the audio matches a specific speaker.

**Request**
- Query param: `name` (speaker name to verify against)
- `multipart/form-data`
- Field: `file` (audio file)

**Example**
```bash
curl -X POST \
  -F "file=@voice.wav" \
  "http://localhost:7860/verify?name=alice"
```

**Response**
```json
{
  "verified": true,
  "speaker": "alice",
  "confidence": 0.92,
  "threshold": 0.25
}
```

---

## Deployment (Hugging Face Space)

Recommended setup:

- **Space type**: `Docker`
- **Hardware**: CPU (default) or GPU
- **Exposed port**: `7860`

### Repository structure

```
reachy-speechbrain-api/
├── app.py              # FastAPI application
├── storage.py          # Storage backends (local & HuggingFace)
├── Dockerfile          # Docker image definition
├── pyproject.toml      # Project configuration and dependencies
├── uv.lock             # Lockfile for reproducible builds
├── .gitignore          # Git ignore rules
├── speakers/           # Speaker embeddings storage (created at runtime)
├── tests/              # Test suite
│   ├── __init__.py
│   ├── conftest.py     # Pytest fixtures
│   ├── test_api.py     # API tests
│   └── test_storage.py # Storage tests
└── README.md
```

Once pushed, the Space will automatically build and expose:
```
https://<username>-<space-name>.hf.space
```

---

## Docker (local run)

```bash
docker build -t reachy-speechbrain-api .
docker run -p 7860:7860 reachy-speechbrain-api
```

---

## Storage Configuration

Speaker embeddings can be stored locally or on Hugging Face Hub.

### Local storage (default)

By default, embeddings are stored in `speakers/embeddings.json`. No configuration needed.

### Hugging Face Hub storage

To persist embeddings in a Hugging Face dataset (useful for sharing between instances):

```bash
# Set environment variables
export HF_EMBEDDINGS_REPO="username/my-speaker-embeddings"
export HF_TOKEN="hf_xxxxxxxxxxxxx"  # Optional if logged in via `huggingface-cli login`

# Run the API
uv run uvicorn app:app --host 0.0.0.0 --port 7860
```

Or in Docker:
```bash
docker run -p 7860:7860 \
  -e HF_EMBEDDINGS_REPO="username/my-speaker-embeddings" \
  -e HF_TOKEN="hf_xxxxxxxxxxxxx" \
  reachy-speechbrain-api
```

The dataset will be created automatically (as private) if it doesn't exist.

---

## Dependencies

Dependencies are managed using **uv**.

Main dependencies:
- `fastapi`
- `uvicorn`
- `speechbrain` (develop branch)
- `torchaudio`
- `python-multipart`
- `requests`
- `huggingface-hub`

The lockfile (`uv.lock`) ensures reproducible builds.

---

## Development

Install dev dependencies:
```bash
uv sync --extra dev
```

### Tools

- **ruff** - Linter and formatter
- **mypy** - Static type checker
- **pytest** - Testing framework
- **pytest-cov** - Code coverage

### Run tests
```bash
uv run pytest
```

Coverage report is generated in `htmlcov/` and displayed in terminal.

### Lint and format
```bash
uv run ruff check .
uv run ruff format .
```

### Type checking
```bash
uv run mypy .
```

### Release workflow

This project uses [commitizen](https://commitizen-tools.github.io/commitizen/) for versioning and changelog generation.

To trigger a new release, push a commit to `main` with the message `chore: release a new version`:

```bash
git commit --allow-empty -m "chore: release a new version"
git push origin main
```

This will:
1. Bump the version based on conventional commits
2. Generate/update the CHANGELOG
3. Create a GitHub Release
4. Sync to Hugging Face Space

---

## Usage with Reachy

This API is designed to be called from:
- Reachy Mini
- A central VPS backend
- Another Hugging Face Space

Typical flow:
1. **Enrollment**: Record voice samples from known users and enroll them
2. **Identification**: When someone speaks, send audio to `/identify` to know who it is
3. **Verification**: Use `/verify` to confirm a claimed identity

Use cases:
- Personalized interactions based on who is speaking
- Access control for voice commands
- Multi-user conversation tracking

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
