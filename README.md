# OpenIF

OpenIF (Open Intelligence Function) classifies Japanese utterances into the 60 intents of MASSIVE 1.1. GPR-1 Nasubi combines Japanese morphological analysis, lexical rules, 24 sparse feature blocks and a learned linear head. Inference runs locally on a CPU.

GPR stands for *Glorified Predefined Rules*. Nasubi takes its name from the eggplant in the Japanese saying about auspicious first dreams. Our contribution to Artificial Ifelligence is *Intelligence is temporary. Conditions are forever.*

## Install from source

From the root of this checkout, use a Python 3.12 virtual environment. The tested platform is Linux x86-64. The dictionaries are substantial downloads on first installation.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Predict

```python
from openif import Classifier

model_id = "openif/gpr-1-nasubi-massive"
model = Classifier.from_pretrained(model_id)

model.predict("明日の天気を教えて")                        # 'weather_query'
model.predict(["朝7時にアラームを設定して", "音楽を再生して"])  # ['alarm_set', 'play_music']

model.labels              # tuple of the 60 intent names in score order
model.model_id            # 'openif/gpr-1-nasubi-massive'
model.architecture        # 'gpr-1-nasubi'
model.extractor_version   # '1'
model.decision_function("音楽を再生して").shape  # (60,)
```

A string returns one label. A list returns labels in order, including duplicates. An empty list returns `[]`. Empty, whitespace-only and unfamiliar text receive an intent from the same closed set. The model has no abstention class. A list of inputs produces a score matrix of shape `(n, 60)`. Scores are uncalibrated margins. Invalid input types raise `TypeError`. The extractor preserves the full utterance. Inputs beyond Sudachi's supported byte length raise `ValueError`.

## Download once, run offline

The first load downloads the 15.98 MiB joblib file into Hugging Face's cache. Subsequent loads can use the same cache with the network disabled. Supply the same `cache_dir` on both calls when configuring a custom location.

```python
model = Classifier.from_pretrained(model_id, local_files_only=True)
```

OpenIF accepts the official model at a pinned Hub commit. It verifies the SHA-256 of the exact bytes passed to the joblib decoder on every load. Pickle-based formats can execute code during decoding, so the loader rejects unknown model IDs, revisions and mismatched hashes. A missing offline cache raises an error with instructions to populate it.

[Model and weights on Hugging Face](https://huggingface.co/openif/gpr-1-nasubi-massive)

## Accuracy and scope

| MASSIVE 1.1 ja-JP split | Correct | Accuracy | Macro-F1 (60 intents) |
|---|---:|---:|---:|
| Validation, verified with this package | 1,759 / 2,033 | 86.52% | 84.37% |
| Test, previously reported independent scoring | 2,538 / 2,974 | 85.34% | 81.21% |

The macro average includes all 60 intents and assigns zero to undefined F1. Validation has a development history, and the project has previously seen aggregate test results. These figures describe this benchmark. Accuracy on out-of-domain requests requires separate evaluation.

With the model cached on Linux x86-64 (AMD EPYC 9V74, CPython 3.12.14, one BLAS thread), single-utterance latency was 3.99 ms p50 and 4.90 ms p95. A batch of 128 took 45.54 ms p50. From process launch through the first prediction took 1.55 s, and peak RSS in the single-utterance run was 570 MiB. Results are medians across three processes, each scoring 128 individual utterances or 16 batches. Dictionary files account for approximately 456 MiB on disk.

## Source Characters

The four extractor modules contain **10,163 Source Characters** across 331 lines, with 12 `if` or `elif` statements. The JSON lexicon adds 10 rules with 111 topic, cue and veto entries.

Run `python scripts/source_characters.py` from the source checkout for per-file counts. It counts UTF-8 decoded code points, including comments, whitespace and newlines, in the Nasubi extractor, morphology, projection and rule modules. Lines use `splitlines()`. The branch count includes Python AST `If` nodes (including `elif`). The script reports the JSON lexicon separately. Dependencies, the head, API, loader and tests are outside the source count.

The classifier projects onto 269,048 dimensions. Its head contains 16,142,880 learned coefficients and 60 biases, in addition to the fitted vectorizers. Source Characters measure the extractor implementation.

## Distribution terms

The package and trained-model licenses are under review. OpenIF is an independent project. The source distribution includes data and dictionary notices in `NOTICE.md` and the first-release procedure in `docs/release.md`.

## License

Licensing terms are currently under review. Commercial use and redistribution permissions have not yet been established.