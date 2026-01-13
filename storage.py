"""Storage backends for speaker embeddings."""

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, cast

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

logger = logging.getLogger("uvicorn.error")

EMBEDDINGS_FILENAME = "embeddings.json"
EmbeddingsDict = dict[str, list[float]]


def _load_json(f: Any) -> EmbeddingsDict:
    """Load JSON and cast to embeddings dict type."""
    return cast(EmbeddingsDict, json.load(f))


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def load(self) -> dict[str, list[float]]:
        """Load speaker embeddings."""
        pass

    @abstractmethod
    def save(self, embeddings: dict[str, list[float]]) -> None:
        """Save speaker embeddings."""
        pass


class LocalStorage(StorageBackend):
    """Local file storage backend."""

    def __init__(self, storage_dir: Path = Path("speakers")):
        self.storage_dir = storage_dir
        self.storage_file = storage_dir / EMBEDDINGS_FILENAME

    def load(self) -> dict[str, list[float]]:
        """Load speaker embeddings from local file."""
        if self.storage_file.exists():
            with open(self.storage_file) as f:
                return _load_json(f)
        return {}

    def save(self, embeddings: dict[str, list[float]]) -> None:
        """Save speaker embeddings to local file."""
        self.storage_dir.mkdir(exist_ok=True)
        with open(self.storage_file, "w") as f:
            json.dump(embeddings, f)


class HuggingFaceStorage(StorageBackend):
    """Hugging Face Hub dataset storage backend."""

    def __init__(self, repo_id: str, token: str | None = None):
        """
        Initialize HuggingFace storage.

        Args:
            repo_id: The HuggingFace dataset repository ID
                (e.g., "username/dataset-name")
            token: HuggingFace API token. If None, uses HF_TOKEN env var
                or cached login.
        """
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token)
        self._local_cache = Path("speakers") / EMBEDDINGS_FILENAME

    def load(self) -> dict[str, list[float]]:
        """Load speaker embeddings from HuggingFace dataset."""
        try:
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=EMBEDDINGS_FILENAME,
                repo_type="dataset",
                token=self.token,
            )
            with open(local_path) as f:
                return _load_json(f)
        except (EntryNotFoundError, RepositoryNotFoundError):
            logger.info(
                f"No embeddings found in {self.repo_id}, starting with empty database"
            )
            return {}
        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace: {e}, using local cache")
            if self._local_cache.exists():
                with open(self._local_cache) as f:
                    return _load_json(f)
            return {}

    def save(self, embeddings: dict[str, list[float]]) -> None:
        """Save speaker embeddings to HuggingFace dataset."""
        # Save locally first
        self._local_cache.parent.mkdir(exist_ok=True)
        with open(self._local_cache, "w") as f:
            json.dump(embeddings, f)

        # Upload to HuggingFace
        try:
            self.api.upload_file(
                path_or_fileobj=str(self._local_cache),
                path_in_repo=EMBEDDINGS_FILENAME,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message="Update speaker embeddings",
            )
            logger.info(f"Embeddings saved to {self.repo_id}")
        except RepositoryNotFoundError:
            logger.info(f"Creating dataset repository {self.repo_id}")
            self.api.create_repo(
                repo_id=self.repo_id,
                repo_type="dataset",
                private=True,
            )
            self.api.upload_file(
                path_or_fileobj=str(self._local_cache),
                path_in_repo=EMBEDDINGS_FILENAME,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message="Initial speaker embeddings",
            )
            logger.info(f"Created and saved embeddings to {self.repo_id}")
        except Exception as e:
            logger.error(f"Failed to save to HuggingFace: {e}")
            raise


def get_storage_backend() -> StorageBackend:
    """
    Get the appropriate storage backend based on environment variables.

    Environment variables:
        HF_EMBEDDINGS_REPO: HuggingFace dataset repo ID for storing embeddings
        HF_TOKEN: HuggingFace API token (optional if logged in via CLI)

    Returns:
        LocalStorage if HF_EMBEDDINGS_REPO is not set, otherwise HuggingFaceStorage
    """
    hf_repo = os.environ.get("HF_EMBEDDINGS_REPO")

    if hf_repo:
        logger.info(f"Using HuggingFace storage: {hf_repo}")
        return HuggingFaceStorage(repo_id=hf_repo)

    logger.info("Using local storage")
    return LocalStorage()
