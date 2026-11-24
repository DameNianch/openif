import os
import sys
from types import ModuleType

import numpy as np
import pytest
from openif.hub import NASUBI

from openif import Classifier


@pytest.fixture(scope="module")
def model():
    if not os.environ.get("OPENIF_TEST_MODEL"):
        pytest.skip("Set OPENIF_TEST_MODEL=1 to exercise the official Hub artifact.")
    return Classifier.from_pretrained(NASUBI.model_id)


def test_predictions_and_order(model):
    inputs = ["明日の天気を教えて", "朝7時にアラームを設定して", "音楽を再生して"]
    expected = ["weather_query", "alarm_set", "play_music"]
    assert model.predict(inputs) == expected
    assert [model.predict(text) for text in inputs] == expected
    repeated = (inputs * 44)[:130]
    assert model.predict(repeated) == (expected * 44)[:130]
    assert model.labels == tuple(sorted(set(model.labels)))
    assert len(model.labels) == 60
    assert model.model_id == NASUBI.model_id
    assert model.extractor_version == NASUBI.extractor_version
    batch = model.decision_function(inputs)
    assert batch.shape == (3, 60) and batch.dtype == np.float32
    for text, margins in zip(inputs, batch):
        np.testing.assert_array_equal(margins, model.decision_function(text))
    assert [model.labels[i] for i in batch.argmax(axis=1)] == expected


@pytest.mark.parametrize("value", [None, b"string", ("日本語",), ["あ", 1]])
def test_rejects_types(model, value):
    with pytest.raises(TypeError):
        model.predict(value)


def test_empty_unknown_long(model):
    assert model.predict([]) == []
    assert model.decision_function([]).shape == (0, 60)
    for text in ["", "　\t", "𰻞🪿ﾃｽﾄ", "猫" * 2049]:
        assert model.predict(text) == model.predict([text])[0]
    with pytest.raises(ValueError, match="too long"):
        model.predict("猫" * 52000)


def test_scoped_pickle_reference(model, monkeypatch):
    unrelated = ModuleType("sparse_tokens")
    monkeypatch.setitem(sys.modules, "sparse_tokens", unrelated)
    original_path = sys.path.copy()
    cached = Classifier.from_pretrained(model.model_id, local_files_only=True)
    assert cached.predict("音楽を再生して") == "play_music"
    assert sys.path == original_path
    assert sys.modules["sparse_tokens"] is unrelated
