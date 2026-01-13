import io
import wave
from unittest.mock import MagicMock, patch

import pytest
import torch

import app as app_module
from app import preprocess_audio


def create_wav_buffer(
    duration_seconds: float = 1.0, sample_rate: int = 16000
) -> io.BytesIO:
    """Create a valid WAV file buffer with silence."""
    buffer = io.BytesIO()
    n_frames = int(duration_seconds * sample_rate)

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * n_frames)

    buffer.seek(0)
    return buffer


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_speakers_empty(client):
    """Test listing speakers when none are enrolled."""
    original_embeddings = app_module.speaker_embeddings.copy()
    app_module.speaker_embeddings.clear()

    try:
        response = client.get("/speakers")
        assert response.status_code == 200
        assert response.json() == {"speakers": []}
    finally:
        app_module.speaker_embeddings.update(original_embeddings)


def test_list_speakers_with_data(client):
    """Test listing speakers with enrolled speakers."""
    original_embeddings = app_module.speaker_embeddings.copy()
    app_module.speaker_embeddings["alice"] = [0.1] * 192
    app_module.speaker_embeddings["bob"] = [0.2] * 192

    try:
        response = client.get("/speakers")
        assert response.status_code == 200
        data = response.json()
        assert set(data["speakers"]) == {"alice", "bob"}
    finally:
        app_module.speaker_embeddings.clear()
        app_module.speaker_embeddings.update(original_embeddings)


def test_enroll_speaker(client):
    """Test enrolling a new speaker."""
    original_model = app_module.speaker_model
    original_embeddings = app_module.speaker_embeddings.copy()

    mock_model = MagicMock()
    mock_model.encode_batch.return_value = torch.randn(1, 192)
    app_module.speaker_model = mock_model
    app_module.speaker_embeddings.clear()

    try:
        buffer = create_wav_buffer()

        with patch("app.torchaudio.load") as mock_load:
            mock_load.return_value = (torch.zeros(1, 16000), 16000)

            mock_storage = MagicMock()
            original_storage = app_module.storage
            app_module.storage = mock_storage

            try:
                response = client.post(
                    "/speakers/alice/enroll",
                    files={"file": ("test.wav", buffer, "audio/wav")},
                )

                assert response.status_code == 200
                data = response.json()
                assert "alice" in data["message"]
                assert data["embedding_size"] == 192
                assert "alice" in app_module.speaker_embeddings
            finally:
                app_module.storage = original_storage
    finally:
        app_module.speaker_model = original_model
        app_module.speaker_embeddings.clear()
        app_module.speaker_embeddings.update(original_embeddings)


def test_enroll_speaker_model_not_loaded(client):
    """Test enrolling when model is not loaded."""
    original_model = app_module.speaker_model
    app_module.speaker_model = None

    try:
        buffer = create_wav_buffer()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            client.post(
                "/speakers/alice/enroll",
                files={"file": ("test.wav", buffer, "audio/wav")},
            )
    finally:
        app_module.speaker_model = original_model


def test_delete_speaker(client):
    """Test deleting a speaker."""
    original_embeddings = app_module.speaker_embeddings.copy()
    original_storage = app_module.storage
    app_module.speaker_embeddings["alice"] = [0.1] * 192

    mock_storage = MagicMock()
    app_module.storage = mock_storage

    try:
        response = client.delete("/speakers/alice")

        assert response.status_code == 200
        assert "alice" not in app_module.speaker_embeddings
    finally:
        app_module.speaker_embeddings.clear()
        app_module.speaker_embeddings.update(original_embeddings)
        app_module.storage = original_storage


def test_delete_speaker_not_found(client):
    """Test deleting a non-existent speaker."""
    response = client.delete("/speakers/unknown")
    assert response.status_code == 404


def test_identify_speaker(client):
    """Test identifying a speaker."""
    original_model = app_module.speaker_model
    original_embeddings = app_module.speaker_embeddings.copy()

    mock_model = MagicMock()
    mock_model.encode_batch.return_value = torch.randn(1, 192)
    mock_model.similarity.return_value = torch.tensor([[0.85]])
    app_module.speaker_model = mock_model

    app_module.speaker_embeddings.clear()
    app_module.speaker_embeddings["alice"] = [0.1] * 192

    try:
        buffer = create_wav_buffer()

        with patch("app.torchaudio.load") as mock_load:
            mock_load.return_value = (torch.zeros(1, 16000), 16000)

            response = client.post(
                "/identify", files={"file": ("test.wav", buffer, "audio/wav")}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["identified"] is True
            assert data["speaker"] == "alice"
            assert "confidence" in data
            assert "threshold" in data
    finally:
        app_module.speaker_model = original_model
        app_module.speaker_embeddings.clear()
        app_module.speaker_embeddings.update(original_embeddings)


def test_identify_speaker_no_match(client):
    """Test identification when no speaker matches."""
    original_model = app_module.speaker_model
    original_embeddings = app_module.speaker_embeddings.copy()

    mock_model = MagicMock()
    mock_model.encode_batch.return_value = torch.randn(1, 192)
    mock_model.similarity.return_value = torch.tensor([[0.1]])  # Below threshold
    app_module.speaker_model = mock_model

    app_module.speaker_embeddings.clear()
    app_module.speaker_embeddings["alice"] = [0.1] * 192

    try:
        buffer = create_wav_buffer()

        with patch("app.torchaudio.load") as mock_load:
            mock_load.return_value = (torch.zeros(1, 16000), 16000)

            response = client.post(
                "/identify", files={"file": ("test.wav", buffer, "audio/wav")}
            )

            assert response.status_code == 200
            data = response.json()
            assert data["identified"] is False
            assert data["speaker"] is None
    finally:
        app_module.speaker_model = original_model
        app_module.speaker_embeddings.clear()
        app_module.speaker_embeddings.update(original_embeddings)


def test_identify_no_speakers_enrolled(client):
    """Test identification when no speakers are enrolled."""
    original_model = app_module.speaker_model
    original_embeddings = app_module.speaker_embeddings.copy()

    mock_model = MagicMock()
    app_module.speaker_model = mock_model
    app_module.speaker_embeddings.clear()

    try:
        buffer = create_wav_buffer()

        response = client.post(
            "/identify", files={"file": ("test.wav", buffer, "audio/wav")}
        )

        assert response.status_code == 400
        assert "No speakers enrolled" in response.json()["detail"]
    finally:
        app_module.speaker_model = original_model
        app_module.speaker_embeddings.clear()
        app_module.speaker_embeddings.update(original_embeddings)


def test_verify_speaker(client):
    """Test verifying a specific speaker."""
    original_model = app_module.speaker_model
    original_embeddings = app_module.speaker_embeddings.copy()

    mock_model = MagicMock()
    mock_model.encode_batch.return_value = torch.randn(1, 192)
    mock_model.similarity.return_value = torch.tensor([[0.85]])
    app_module.speaker_model = mock_model

    app_module.speaker_embeddings.clear()
    app_module.speaker_embeddings["alice"] = [0.1] * 192

    try:
        buffer = create_wav_buffer()

        with patch("app.torchaudio.load") as mock_load:
            mock_load.return_value = (torch.zeros(1, 16000), 16000)

            response = client.post(
                "/verify",
                params={"name": "alice"},
                files={"file": ("test.wav", buffer, "audio/wav")},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["verified"] is True
            assert data["speaker"] == "alice"
            assert "confidence" in data
    finally:
        app_module.speaker_model = original_model
        app_module.speaker_embeddings.clear()
        app_module.speaker_embeddings.update(original_embeddings)


def test_verify_speaker_not_found(client):
    """Test verifying a non-existent speaker."""
    original_model = app_module.speaker_model
    original_embeddings = app_module.speaker_embeddings.copy()

    mock_model = MagicMock()
    app_module.speaker_model = mock_model
    app_module.speaker_embeddings.clear()

    try:
        buffer = create_wav_buffer()

        response = client.post(
            "/verify",
            params={"name": "unknown"},
            files={"file": ("test.wav", buffer, "audio/wav")},
        )

        assert response.status_code == 404
    finally:
        app_module.speaker_model = original_model
        app_module.speaker_embeddings.clear()
        app_module.speaker_embeddings.update(original_embeddings)


def test_verify_speaker_failed(client):
    """Test verification when speaker doesn't match."""
    original_model = app_module.speaker_model
    original_embeddings = app_module.speaker_embeddings.copy()

    mock_model = MagicMock()
    mock_model.encode_batch.return_value = torch.randn(1, 192)
    mock_model.similarity.return_value = torch.tensor([[0.1]])  # Below threshold
    app_module.speaker_model = mock_model

    app_module.speaker_embeddings.clear()
    app_module.speaker_embeddings["alice"] = [0.1] * 192

    try:
        buffer = create_wav_buffer()

        with patch("app.torchaudio.load") as mock_load:
            mock_load.return_value = (torch.zeros(1, 16000), 16000)

            response = client.post(
                "/verify",
                params={"name": "alice"},
                files={"file": ("test.wav", buffer, "audio/wav")},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["verified"] is False
    finally:
        app_module.speaker_model = original_model
        app_module.speaker_embeddings.clear()
        app_module.speaker_embeddings.update(original_embeddings)


def test_identify_model_not_loaded(client):
    """Test identify when model is not loaded."""
    original_model = app_module.speaker_model
    original_embeddings = app_module.speaker_embeddings.copy()

    app_module.speaker_model = None
    app_module.speaker_embeddings["alice"] = [0.1] * 192

    try:
        buffer = create_wav_buffer()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            client.post("/identify", files={"file": ("test.wav", buffer, "audio/wav")})
    finally:
        app_module.speaker_model = original_model
        app_module.speaker_embeddings.clear()
        app_module.speaker_embeddings.update(original_embeddings)


def test_verify_model_not_loaded(client):
    """Test verify when model is not loaded."""
    original_model = app_module.speaker_model
    original_embeddings = app_module.speaker_embeddings.copy()

    app_module.speaker_model = None
    app_module.speaker_embeddings["alice"] = [0.1] * 192

    try:
        buffer = create_wav_buffer()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            client.post(
                "/verify",
                params={"name": "alice"},
                files={"file": ("test.wav", buffer, "audio/wav")},
            )
    finally:
        app_module.speaker_model = original_model
        app_module.speaker_embeddings.clear()
        app_module.speaker_embeddings.update(original_embeddings)


def test_preprocess_audio_with_resampling():
    """Test audio preprocessing with resampling from 44.1kHz to 16kHz."""
    with patch("app.torchaudio.load") as mock_load:
        mock_load.return_value = (torch.zeros(1, 44100), 44100)

        with patch("app.torchaudio.transforms.Resample") as mock_resample:
            mock_resampler = MagicMock()
            mock_resampler.return_value = torch.zeros(1, 16000)
            mock_resample.return_value = mock_resampler

            result = preprocess_audio(b"fake_audio_data")

            mock_resample.assert_called_once_with(orig_freq=44100, new_freq=16000)
            assert result.shape == (1, 16000)


def test_preprocess_audio_stereo_to_mono():
    """Test audio preprocessing converting stereo to mono."""
    with patch("app.torchaudio.load") as mock_load:
        # Stereo audio (2 channels)
        mock_load.return_value = (torch.zeros(2, 16000), 16000)

        result = preprocess_audio(b"fake_audio_data")

        # Should be converted to mono (1 channel)
        assert result.shape[0] == 1


def test_preprocess_audio_no_changes_needed():
    """Test audio preprocessing when no resampling or mono conversion needed."""
    with patch("app.torchaudio.load") as mock_load:
        mock_load.return_value = (torch.zeros(1, 16000), 16000)

        result = preprocess_audio(b"fake_audio_data")

        assert result.shape == (1, 16000)
