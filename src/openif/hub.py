"""The single supported model and its authenticated-by-digest download path."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from httpx import HTTPError
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from .compatibility import _decode_verified


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    architecture: str
    extractor_version: str
    filename: str
    revision: str
    sha256: str
    dimensions: int
    label_count: int


NASUBI = ModelSpec(
    model_id="openif/gpr-1-nasubi-massive",
    architecture="gpr-1-nasubi",
    extractor_version="1",
    filename="gpr-1-nasubi-massive.joblib",
    revision="2ca6313cc646d79fe4320b340633886a91c9986e",
    sha256="b93e3c138ccffa17518c07f04ce64aea8f7012e35daf8a1d57bca81088c455d1",
    dimensions=269_048,
    label_count=60,
)


def load_model(
    model_id: str,
    revision: str | None,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> dict:
    if model_id != NASUBI.model_id:
        raise ValueError(f"Unknown model. The supported model is {NASUBI.model_id!r}.")
    if revision is not None and revision != NASUBI.revision:
        raise ValueError(f"Unverified revision. Use {NASUBI.revision!r}.")
    try:
        path = hf_hub_download(
            repo_id=model_id,
            filename=NASUBI.filename,
            revision=NASUBI.revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    except LocalEntryNotFoundError as error:
        raise OSError(
            "Nasubi is unavailable in this cache. Check cache_dir and run "
            "from_pretrained once with network access."
        ) from error
    except (OSError, HTTPError, HfHubHTTPError) as error:
        raise OSError(
            "Cannot retrieve Nasubi. Check cache permissions, Hub access and "
            "network connectivity."
        ) from error

    # Decode exactly the immutable bytes that passed verification, including on cache hits.
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != NASUBI.sha256:
        raise ValueError(
            "Nasubi SHA-256 mismatch. Remove the damaged cached artifact "
            "and download the pinned revision again."
        )
    return _decode_verified(data)
