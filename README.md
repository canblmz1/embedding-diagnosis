# embedding-diagnosis 0.1.0a1 — developer handoff

Maintained by **Cuma Can Bilmez (Can)** — [@canblmz1](https://github.com/canblmz1).

**Developer alpha.** Start with the disposable synthetic example and read the
evidence/observer requirements before connecting an application.

This small Python component reports `match`, `mismatch` or `unknown` for selected
stored document vectors in an initialized LightRAG Nano application. It uses
the application's document embedding callable and preserves the accepted
0.1.0a1 wheel, source archive and full example without rebuilding them.

## Start here

1. Follow [QUICKSTART_EN.md](QUICKSTART_EN.md) from this directory. It contains
   the complete Linux/Python 3.12 installation commands and model download step.
2. Run [examples/lightrag_local.py](examples/lightrag_local.py) with a new,
   disposable output directory. The accepted expected result is **2 match,
   0 mismatch, 2 unknown**.
3. Read [OBSERVER_AND_SCOPE.md](OBSERVER_AND_SCOPE.md) before connecting an
   application or designing a negative control. It explains the actual-input
   evidence, native readback, cosine metric, observer and lifecycle boundary.
4. Record your experience in [FEEDBACK.md](FEEDBACK.md). Demo execution,
   application integration and reuse intention are separate outcomes.

The source distribution is
[dist/embedding_diagnosis-0.1.0a1.tar.gz](dist/embedding_diagnosis-0.1.0a1.tar.gz).
The installable wheel is
[dist/embedding_diagnosis-0.1.0a1-py3-none-any.whl](dist/embedding_diagnosis-0.1.0a1-py3-none-any.whl).
All runtime pins and the model manifest/downloader are under `integration/`.
No model weights or environment are bundled.

## Fixed starting point

| Component | Accepted version or revision |
| --- | --- |
| Python / platform | Python 3.12.14, Linux |
| Torch | 2.5.1+cpu |
| Transformers | 4.57.3 |
| NanoVectorDB | 0.0.4.3 |
| LightRAG | `aaebfe1bb8d55deb6a4304d357a3bf4c92fcd40a` (metadata 1.5.8) |
| Model | `sentence-transformers/paraphrase-MiniLM-L3-v2` |
| Model revision | `4ca70771034acceecb2e72475f72050fcdde4ddc` |
| Example cosine threshold | `0.99999`, specific to this calibrated path |

The existing downloader requires `--download` to use the fixed manifest.
Do not run it with Python `-O`, because its integrity checks use assertions.
The eight selected model files total 70,272,180 bytes; dependencies are additional.

## Existing evidence and current trial status

The accepted local example and separate stored-vector control were run on
31 October 2026. Their model/native execution was not repeated to assemble this
handoff. Packaging checks verify identity and references only.

[EXPECTED_OUTCOMES.md](EXPECTED_OUTCOMES.md) explains those results and links a
compact, unchanged result record from the controlled stored-vector intervention.
That intervention is not a model change or an embedding-path-change test.

Design feedback and interest in a local trial were received on 2 November.
The interested developer explicitly had not run the alpha. No independent
installation, application integration or repeat use is claimed.

## License and attribution

The authorized project contributions are provided under [Apache-2.0](LICENSE),
Copyright 2026 Cuma Can Bilmez. [LICENSE_SCOPE.md](LICENSE_SCOPE.md) identifies
the covered contributions and exact unchanged distributions. Third-party terms
and attributions remain applicable; the model's license is separate. Keep
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [MODEL_NOTICE.md](MODEL_NOTICE.md),
the license/scope files and the complete `licenses/` texts with distributions.
