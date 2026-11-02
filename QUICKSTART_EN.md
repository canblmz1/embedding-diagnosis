# embedding-diagnosis 0.1.0a1 — first developer pilot

Read-only diagnosis of **selected stored document embeddings** in an initialized LightRAG Nano application. The report explains `match`, `mismatch`, or `unknown` for each selected native ID.

**License:** authorized project contributions are provided under Apache-2.0; see `LICENSE` and `LICENSE_SCOPE.md`. Third-party and model terms remain separate. Design feedback and interest have been received; no independent installation or integration result has been reported. See `OBSERVER_AND_SCOPE.md` for exact evidence and lifecycle limits.

## 1. Run the complete example

Use Linux, Python 3.12 and Git. The accepted environment used Python 3.12.14 and CPU Torch; Windows/GPU execution was not tested. From the extracted pilot directory:

```bash
sha256sum -c CHECKSUMS.sha256
python3.12 -m venv .venv
.venv/bin/python -m pip install dist/embedding_diagnosis-0.1.0a1-py3-none-any.whl
.venv/bin/python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r integration/requirements-example.txt
.venv/bin/python -m pip install --no-deps git+https://github.com/HKUDS/LightRAG.git@aaebfe1bb8d55deb6a4304d357a3bf4c92fcd40a
.venv/bin/python -m pip check
.venv/bin/python integration/download_model.py --download
.venv/bin/python -I examples/lightrag_local.py --model-dir models/paraphrase-MiniLM-L3-v2 --work-dir runs/first_pilot
```

`runs/first_pilot` must not exist. The unchanged example creates a separate native index, writes four generic texts, closes the writer, reopens the index and attaches the diagnostic API. It does not open your application index. It requires no paid API; its unconfigured LLM callable fails if invoked, and the example does not run LLM extraction/generation.

The downloader uses the pinned manifest in `integration/model_source.json`. **Keep `--download`**: running this historical tool without the flag refreshes its manifest from the current upstream model. Do not use Python `-O`; the historical downloader's integrity checks use assertions. An existing copy of the exact model can be passed through `--model-dir`, skipping the download.

Model: `sentence-transformers/paraphrase-MiniLM-L3-v2`, revision `4ca70771034acceecb2e72475f72050fcdde4ddc`. Model/tokenizer/config files total 70,272,180 bytes; Torch/native dependencies add installation size. No model weights are bundled. See `MODEL_NOTICE.md` for source and license details.

Expected example result: **2 match, 0 mismatch, 2 unknown**. The two unknown records have no recorded historical embedding input; this is intentional. Outputs are `report.json`, `summary.txt`, `calibration.json` and `usage_record.json` under the new run directory. The example checks index-file bytes around diagnosis after reader initialization; lock files, filesystem metadata, initialization/adoption/migration and later finalization are outside that assertion.

## 2. Attach to your own application

Provide initialized `rag.chunks_vdb`, a small list of native chunk IDs and an explicit cosine threshold. Numeric decisions also need:

- Trustworthy original-input evidence keyed by native ID: actual historical encoder input and its hash, source-text hash, document role, preprocessing description and a reference to the recorded observation.
- An observer attached to the **same document embedding callable your application uses**. The supplied `LocalTransformerEncoder` provides a local BERT forward observer; another provider requires application-specific observation code.

In your existing async application:

```python
import json
from pathlib import Path
from embedding_diagnosis import diagnose, summarize
from embedding_diagnosis.lightrag import LightRAGAdapter

adapter = LightRAGAdapter(
    rag.chunks_vdb,
    observer=app_encoder,
    provenance="observed-inputs.json",
)
report = await diagnose(adapter, selected_ids, threshold=my_threshold)
Path("diagnosis.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(summarize(report))
```

The adapter calls the existing `storage.embedding_func(texts, context="document")`. Missing/malformed historical evidence leaves readable records `unknown` without embedding those records. Without an observer, replay is not attempted. Today's configuration or stored text alone is not proof of the historical encoder input. Do not reuse the example's evidence for your own IDs. A Python call or remote API request alone cannot prove uncached encoder execution.

The core wheel only requires NumPy. The example's 70 runtime pins, CPU Torch and pinned LightRAG are separate application dependencies; none were removed to shrink this bundle. The original source distribution is also provided unchanged for inspection.

## 3. Interpret the result and report the effort

| Result | Meaning |
| --- | --- |
| `match` | This selected record agrees under the declared document contract and tolerance. |
| `mismatch` | This selected stored vector was not reproduced under that contract. It does not identify a model change or certify index-wide corruption. |
| `unknown` | Required evidence, usable data, call association or execution observation is missing. Read `reason` and the observation explanation. |

The example threshold `0.99999` belongs to the accepted CPU/MiniLM/mean-pooling/Nano Float16 path, supported by two held-out calibration texts. It is not a universal threshold. Keep your own justified tolerance explicit.

Supported native source: LightRAG `aaebfe1bb8d55deb6a4304d357a3bf4c92fcd40a` (metadata version 1.5.8), NanoVectorDB 0.0.4.3. Finish writes and reopen a quiescent index; no concurrent writer or pending/unsaved upsert/delete is supported. The adapter reads five private Nano fields: `_pending_upserts`, `_unsaved_upserts`, `_pending_deletes`, `_unsaved_deletes`, `_client_dirty`. Missing guards are not silently accepted. Only a single ordered provider call with attributable encoder events is supported.

Complete `FEEDBACK.md`: record **demo execution**, **your own application integration**, and **reuse intent** separately. Report active setup/wiring time and help needed, if available. Full diagnostic JSON contains text and vectors; sanitized counts, reason strings and connection snippets are sufficient for initial feedback.
