import io
import logging
from contextlib import asynccontextmanager
from typing import cast

import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, UploadFile
from speechbrain.inference.speaker import SpeakerRecognition

from storage import StorageBackend, get_storage_backend

logger = logging.getLogger("uvicorn.error")

speaker_model: SpeakerRecognition | None = None
speaker_embeddings: dict[str, list[float]] = {}
storage: StorageBackend | None = None


def preprocess_audio(audio_bytes: bytes) -> torch.Tensor:
    """Load and preprocess audio to 16kHz mono."""
    audio_buffer = io.BytesIO(audio_bytes)
    waveform, sample_rate = torchaudio.load(audio_buffer)

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000
        )
        waveform = resampler(waveform)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return cast(torch.Tensor, waveform)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global speaker_model, speaker_embeddings, storage
    logger.info("Loading SpeechBrain Speaker Recognition model...")
    speaker_model = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
    )

    storage = get_storage_backend()
    speaker_embeddings = storage.load()

    logger.info(
        f"Model loaded successfully! {len(speaker_embeddings)} speakers registered."
    )
    yield
    logger.info("Shutting down...")


app = FastAPI(title="Reachy SpeechBrain API", lifespan=lifespan)


@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}


@app.get("/speakers")
def list_speakers():
    """List all registered speakers."""
    return {"speakers": list(speaker_embeddings.keys())}


@app.post("/speakers/{name}/enroll")
async def enroll_speaker(name: str, file: UploadFile = File(...)):
    """Enroll a new speaker by providing an audio sample."""
    if speaker_model is None:
        raise RuntimeError("Model not loaded")

    audio_bytes = await file.read()
    waveform = preprocess_audio(audio_bytes)

    # Extract embedding
    embedding = speaker_model.encode_batch(waveform)
    embedding_list = embedding.squeeze().tolist()

    # Store embedding
    speaker_embeddings[name] = embedding_list
    if storage:
        storage.save(speaker_embeddings)

    return {
        "message": f"Speaker '{name}' enrolled successfully",
        "embedding_size": len(embedding_list),
    }


@app.delete("/speakers/{name}")
def delete_speaker(name: str):
    """Delete a registered speaker."""
    if name not in speaker_embeddings:
        raise HTTPException(status_code=404, detail=f"Speaker '{name}' not found")

    del speaker_embeddings[name]
    if storage:
        storage.save(speaker_embeddings)

    return {"message": f"Speaker '{name}' deleted successfully"}


@app.post("/identify")
async def identify_speaker(file: UploadFile = File(...)):
    """Identify the speaker from an audio sample."""
    if speaker_model is None:
        raise RuntimeError("Model not loaded")

    if not speaker_embeddings:
        raise HTTPException(
            status_code=400,
            detail="No speakers enrolled. Please enroll speakers first.",
        )

    audio_bytes = await file.read()
    waveform = preprocess_audio(audio_bytes)

    # Extract embedding for input audio
    input_embedding = speaker_model.encode_batch(waveform)

    # Compare with all registered speakers
    best_match = None
    best_score = -1.0

    for name, stored_embedding in speaker_embeddings.items():
        stored_tensor = torch.tensor(stored_embedding).unsqueeze(0)
        score = speaker_model.similarity(input_embedding, stored_tensor)
        score_value = float(score.squeeze())

        if score_value > best_score:
            best_score = score_value
            best_match = name

    # Threshold for identification (ECAPA-TDNN typically uses ~0.25)
    threshold = 0.25
    identified = best_score >= threshold

    return {
        "identified": identified,
        "speaker": best_match if identified else None,
        "confidence": best_score,
        "threshold": threshold,
    }


@app.post("/verify")
async def verify_speaker(name: str, file: UploadFile = File(...)):
    """Verify if the audio matches a specific speaker."""
    if speaker_model is None:
        raise RuntimeError("Model not loaded")

    if name not in speaker_embeddings:
        raise HTTPException(status_code=404, detail=f"Speaker '{name}' not found")

    audio_bytes = await file.read()
    waveform = preprocess_audio(audio_bytes)

    # Extract embedding for input audio
    input_embedding = speaker_model.encode_batch(waveform)

    # Compare with stored embedding
    stored_tensor = torch.tensor(speaker_embeddings[name]).unsqueeze(0)
    score = speaker_model.similarity(input_embedding, stored_tensor)
    score_value = float(score.squeeze())

    threshold = 0.25
    verified = score_value >= threshold

    return {
        "verified": verified,
        "speaker": name,
        "confidence": score_value,
        "threshold": threshold,
    }
