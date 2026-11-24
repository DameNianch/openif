from unittest.mock import Mock

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from openif import Classifier, hub


@pytest.fixture
def decoder(monkeypatch):
    decoder = Mock(side_effect=AssertionError("Unexpected deserialization"))
    monkeypatch.setattr(hub, "_decode_verified", decoder)
    return decoder


@pytest.mark.parametrize(
    "model_id,revision", [("third-party/model", None), (hub.NASUBI.model_id, "main")]
)
def test_unknown_source_stops_before_network(monkeypatch, decoder, model_id, revision):
    monkeypatch.setattr(
        hub, "hf_hub_download", Mock(side_effect=AssertionError("Unexpected download"))
    )
    with pytest.raises(ValueError):
        Classifier.from_pretrained(model_id, revision=revision)
    decoder.assert_not_called()


@pytest.mark.parametrize("offline", [False, True])
def test_damaged_artifact_is_rejected(monkeypatch, decoder, tmp_path, offline):
    artifact = tmp_path / "model"
    artifact.write_bytes(b"damaged")
    download = Mock(return_value=artifact)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    with pytest.raises(ValueError, match="SHA-256"):
        Classifier.from_pretrained(
            hub.NASUBI.model_id, cache_dir=tmp_path, local_files_only=offline
        )
    download.assert_called_once_with(
        repo_id=hub.NASUBI.model_id,
        filename=hub.NASUBI.filename,
        revision=hub.NASUBI.revision,
        cache_dir=tmp_path,
        local_files_only=offline,
    )
    decoder.assert_not_called()


@pytest.mark.parametrize(
    "error", [LocalEntryNotFoundError("missing"), PermissionError("cache")]
)
def test_unavailable_cache(monkeypatch, decoder, error):
    monkeypatch.setattr(hub, "hf_hub_download", Mock(side_effect=error))
    with pytest.raises(OSError, match="cache"):
        Classifier.from_pretrained(hub.NASUBI.model_id, local_files_only=True)
    decoder.assert_not_called()
