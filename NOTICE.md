# Notices

The GPR-1 Nasubi MASSIVE model was trained on the Japanese (ja-JP) portion of MASSIVE 1.1. Amazon.com, Inc. or its affiliates hold copyright in MASSIVE and provide it under CC BY 4.0. MASSIVE localizes and translates SLURP text, also available under CC BY 4.0. OpenIF uses the utterances and intent annotations for training and evaluation. The fitted vocabulary, IDF and classifier coefficients are separate artifacts distributed from the model repository.

- [Amazon MASSIVE copyright notice](https://github.com/alexa/massive/blob/main/NOTICE.md)
- [SLURP text data license](https://github.com/pswietojanski/slurp/blob/master/LICENSE.txt)
- [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

The wheel contains OpenIF's inference code, semantic categories and lexical rule table. The two tokenizers and their dictionaries are installed from their own Python distributions, which include the relevant licenses and notices.

| Distribution | Upstream notices |
|---|---|
| SudachiPy and SudachiDict-core | Apache-2.0. The dictionary also credits UniDic and NEologd resources. See the [dictionary notices](https://github.com/WorksApplications/SudachiDict#license) and [SudachiPy](https://github.com/WorksApplications/sudachi.rs). |
| fugashi | Its wrapper and bundled MeCab carry separate MIT and BSD notices. See [fugashi](https://github.com/polm/fugashi). |
| unidic-lite | The Python package and UniDic 2.1.2 dictionary carry separate notices. See [unidic-lite](https://github.com/polm/unidic-lite#license). |

Distribution terms for OpenIF's own code, rule tables and trained artifacts are under review by the maintainer. Retain upstream notices when redistributing their resources, including in installers or images.
