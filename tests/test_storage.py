import json
import os
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from storage import (
    HuggingFaceStorage,
    LocalStorage,
    get_storage_backend,
)


def _create_repo_not_found_error():
    """Create a RepositoryNotFoundError with required response argument."""
    from huggingface_hub.utils import RepositoryNotFoundError

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.headers = {}
    return RepositoryNotFoundError("Not found", response=mock_response)


class TestLocalStorage:
    """Tests for LocalStorage backend."""

    def test_load_file_exists(self):
        """Test loading embeddings when file exists."""
        test_data = {"alice": [0.1] * 192}
        storage = LocalStorage()

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=json.dumps(test_data))):
                result = storage.load()
                assert result == test_data

    def test_load_file_not_exists(self):
        """Test loading embeddings when file doesn't exist."""
        storage = LocalStorage()

        with patch.object(Path, "exists", return_value=False):
            result = storage.load()
            assert result == {}

    def test_save(self):
        """Test saving embeddings to file."""
        test_data = {"alice": [0.1] * 192}
        storage = LocalStorage()

        with patch.object(Path, "mkdir") as mock_mkdir:
            with patch("builtins.open", mock_open()) as mock_file:
                storage.save(test_data)

                mock_mkdir.assert_called_once_with(exist_ok=True)
                mock_file.assert_called_once()


class TestHuggingFaceStorage:
    """Tests for HuggingFaceStorage backend."""

    def test_init_with_token(self):
        """Test initialization with explicit token."""
        storage = HuggingFaceStorage(repo_id="user/repo", token="test_token")
        assert storage.repo_id == "user/repo"
        assert storage.token == "test_token"

    def test_init_with_env_token(self):
        """Test initialization with token from environment."""
        with patch.dict(os.environ, {"HF_TOKEN": "env_token"}):
            storage = HuggingFaceStorage(repo_id="user/repo")
            assert storage.token == "env_token"

    def test_load_success(self):
        """Test loading embeddings from HuggingFace."""
        test_data = {"alice": [0.1] * 192}
        storage = HuggingFaceStorage(repo_id="user/repo", token="token")

        with patch("storage.hf_hub_download") as mock_download:
            mock_download.return_value = "/tmp/embeddings.json"
            with patch("builtins.open", mock_open(read_data=json.dumps(test_data))):
                result = storage.load()
                assert result == test_data

    def test_load_entry_not_found(self):
        """Test loading when embeddings file doesn't exist on HF."""
        from huggingface_hub.utils import EntryNotFoundError

        storage = HuggingFaceStorage(repo_id="user/repo", token="token")

        with patch("storage.hf_hub_download") as mock_download:
            mock_download.side_effect = EntryNotFoundError("Not found")
            result = storage.load()
            assert result == {}

    def test_load_repo_not_found(self):
        """Test loading when repository doesn't exist."""
        storage = HuggingFaceStorage(repo_id="user/repo", token="token")

        with patch("storage.hf_hub_download") as mock_download:
            mock_download.side_effect = _create_repo_not_found_error()
            result = storage.load()
            assert result == {}

    def test_load_fallback_to_local_cache(self):
        """Test loading falls back to local cache on error."""
        test_data = {"alice": [0.1] * 192}
        storage = HuggingFaceStorage(repo_id="user/repo", token="token")

        with patch("storage.hf_hub_download") as mock_download:
            mock_download.side_effect = Exception("Network error")
            with patch.object(Path, "exists", return_value=True):
                with patch("builtins.open", mock_open(read_data=json.dumps(test_data))):
                    result = storage.load()
                    assert result == test_data

    def test_load_fallback_no_cache(self):
        """Test loading returns empty when no cache and HF fails."""
        storage = HuggingFaceStorage(repo_id="user/repo", token="token")

        with patch("storage.hf_hub_download") as mock_download:
            mock_download.side_effect = Exception("Network error")
            with patch.object(Path, "exists", return_value=False):
                result = storage.load()
                assert result == {}

    def test_save_success(self):
        """Test saving embeddings to HuggingFace."""
        test_data = {"alice": [0.1] * 192}
        storage = HuggingFaceStorage(repo_id="user/repo", token="token")

        with patch.object(Path, "mkdir"):
            with patch("builtins.open", mock_open()):
                with patch.object(storage.api, "upload_file") as mock_upload:
                    storage.save(test_data)
                    mock_upload.assert_called_once()

    def test_save_creates_repo_if_not_exists(self):
        """Test saving creates repo if it doesn't exist."""
        test_data = {"alice": [0.1] * 192}
        storage = HuggingFaceStorage(repo_id="user/repo", token="token")

        with patch.object(Path, "mkdir"):
            with patch("builtins.open", mock_open()):
                with patch.object(storage.api, "upload_file") as mock_upload:
                    mock_upload.side_effect = [
                        _create_repo_not_found_error(),
                        None,
                    ]
                    with patch.object(storage.api, "create_repo") as mock_create:
                        storage.save(test_data)
                        mock_create.assert_called_once()

    def test_save_raises_on_error(self):
        """Test saving raises exception on persistent error."""
        test_data = {"alice": [0.1] * 192}
        storage = HuggingFaceStorage(repo_id="user/repo", token="token")

        with patch.object(Path, "mkdir"):
            with patch("builtins.open", mock_open()):
                with patch.object(storage.api, "upload_file") as mock_upload:
                    mock_upload.side_effect = Exception("Persistent error")
                    with pytest.raises(Exception, match="Persistent error"):
                        storage.save(test_data)


class TestGetStorageBackend:
    """Tests for get_storage_backend factory function."""

    def test_returns_local_storage_by_default(self):
        """Test returns LocalStorage when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove HF_EMBEDDINGS_REPO if it exists
            os.environ.pop("HF_EMBEDDINGS_REPO", None)
            backend = get_storage_backend()
            assert isinstance(backend, LocalStorage)

    def test_returns_hf_storage_when_configured(self):
        """Test returns HuggingFaceStorage when HF_EMBEDDINGS_REPO is set."""
        with patch.dict(os.environ, {"HF_EMBEDDINGS_REPO": "user/repo"}):
            backend = get_storage_backend()
            assert isinstance(backend, HuggingFaceStorage)
            assert backend.repo_id == "user/repo"
