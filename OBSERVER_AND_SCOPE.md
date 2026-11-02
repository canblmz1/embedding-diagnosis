# Observer, input evidence and comparison boundary

This describes the unchanged 0.1.0a1 implementation. The complete runnable
application remains `examples/lightrag_local.py`; the snippets below explain
its connection points rather than replace its initialization and cleanup.

## Same document callable, same observer object

```python
from embedding_diagnosis.local_encoder import LocalTransformerEncoder
from embedding_diagnosis.lightrag import LightRAGAdapter
from embedding_diagnosis import diagnose
from lightrag.utils import EmbeddingFunc

encoder = LocalTransformerEncoder(model_dir, max_length=128)
embedding = EmbeddingFunc(
    embedding_dim=encoder.embedding_dim,
    max_token_size=128,
    func=encoder,
    model_name=("sentence-transformers/paraphrase-MiniLM-L3-v2@"
                "4ca70771034acceecb2e72475f72050fcdde4ddc"),
    supports_asymmetric=True,
)
# Your LightRAG application is constructed with embedding_func=embedding.
# After writes are flushed and a quiescent reader is initialized:
adapter = LightRAGAdapter(
    reader.chunks_vdb,
    observer=encoder,
    provenance="input_evidence.json",
)
report = await diagnose(adapter, selected_ids, threshold=my_threshold)
```

The adapter invokes `storage.embedding_func(texts, context="document")`.
Replay sends the stored source text through that callable. It then compares
the observed actual input with the retained historical actual input. Do not
send an already-prefixed historical input through a prefixing provider again.

The supplied callable loads a local BERT-family model, uses CPU inference and
attention-mask mean pooling, and returns ordered Float32 rows. It observes the
completion of `model.encoder` through a forward hook. A Python wrapper entry or
an API response is insufficient. There is no embedding-output cache in this
callable, and the example does not clear caches to manufacture freshness.

## Historical evidence must already be trustworthy

The example captures evidence prospectively around one native
`ainsert_custom_kg` chunk write at a time. It requires one provider call, one
event and input position zero, then binds the native ID through readback.

`provenance` accepts a native-ID-to-proof mapping or a JSON file containing it.
For each proof the core checks the current source-text hash, retained actual
`embedding_input` and its hash, `role="document"`,
`reconstruction_verified=True`, and nonempty `preprocessing`/`evidence_ref`.
These are evidence declarations from the application. Do not set the verified
flag without the underlying write-time evidence.

The example additionally retains native ID, write/provider call IDs, input
position, tokenized row and encoder-configuration digest. Configuration records
include model-file hashes, maximum length, prefixes, pooling, device and dtype.
The manifest fixes the model/tokenizer revision. The core does not independently
authenticate historical evidence, dereference `evidence_ref`, or require the
historical and current configuration digests to be identical. A hash cannot
reconstruct input that was never retained.

The run emits `input_evidence.json`, `write_observations.json` and
`encoder_configuration.json`, alongside calibration and report outputs. These
files describe that synthetic run; do not transplant its evidence onto other IDs.
An existing stored chunk or today's configuration alone is not proof of the
original encoder input.

## Observer contract and association

A custom observer exposes synchronous `capture()` as a context manager yielding
an `Observation` with `call_id`, `provider_calls`, `events` and `issues` fields.
The adapter accepts one
completed provider call with the captured scope ID and correct input/output
counts. It requires one attributable event for each input position, with the
scope/provider IDs, document role, actual-input digest, observed execution and
an output hash matching the returned row as little-endian Float32.

Duplicate text values are not record identities. Scope/call IDs and positions
bind observations to outputs. The supplied observer uses ContextVars and one
ordered encoder forward per call. **General split batches, arbitrary threading,
reordering and Ollama encoder observation are not supported by this alpha's
demonstrated contract.** Do not synthesize successful events for another provider.

## Which stored vector is compared?

The adapter calls native `get_vectors_by_ids`. In the pinned LightRAG path,
provider output is stored in each record's `vector` payload as Float16 compressed
with zlib/base64, before Nano's client upsert. Readback decodes that payload to
Float32. Diagnosis does not read Nano's normalized ANN search matrix.

Fresh output is the unnormalized Float32 attention-mask mean-pooled vector.
The shared comparator computes cosine with Float64 dot products and norms.
The accepted calibration includes the native Float16 storage round trip.

Consequences for experiments:

- Positive magnitude-only changes can still match: cosine ignores positive scale.
- Changing the actual input string, such as its prefix, returns
  `unknown / replayed_input_differs_from_verified_original` before numerical
  comparison. It does not automatically produce mismatch.
- A direction-changing output under the same verified actual input may produce
  mismatch. The existing stored-vector intervention did not test that path change.
- Dimension disagreement and numeric drift have different reason codes; neither
  establishes root cause or retrieval quality.

The `0.99999` example threshold is supported by two separate calibration texts
for the pinned CPU/MiniLM/Nano path. It is not a universal threshold.

## Read-only interval

The accepted example initializes the reader, takes before-hashes, runs
`diagnose`, takes after-hashes, and finalizes the reader later. The hashes cover
index-file bytes excluding names ending in `.lock`; they do not compare file
permissions, timestamps or other filesystem metadata.

**The unchanged-index assertion covers diagnosis after successful initialization.**
It does not establish read-only initialization, adoption, migration, finalization,
lock handling or concurrent writes. Begin with the example's disposable synthetic
index. A native lifecycle-wide invariant is a separate check, not an existing claim.

The Nano adapter inspects `_pending_upserts`, `_unsaved_upserts`,
`_pending_deletes`, `_unsaved_deletes` and `_client_dirty`. Missing guards or an
unsettled storage are not silently treated as safe.

## Reasons to inspect

| Situation | Reason |
| --- | --- |
| Missing/unverifiable historical input | `original_embedding_input_unverified` |
| Native read or guard failure | `storage_read_error` |
| Missing/ambiguous native ID | `stored_id_missing_or_ambiguous` |
| Provider failure or invalid output shape/count | `document_embedding_error` |
| Unverified encoder execution | `fresh_backend_execution_unverified` |
| Unverified document role | `document_role_unverified` |
| Actual replay input differs from historical input | `replayed_input_differs_from_verified_original` |
| Different vector dimensions after evidence checks | `vector_dimension_mismatch` |
| Cosine below declared threshold | `not_reproduced_under_document_contract` |
| Cosine within declared threshold | `cosine_within_declared_tolerance` |

Also inspect `fresh_evidence.observation_issue`. Association problems can surface
as a role reason because role is checked before freshness. Missing historical
evidence or an unavailable observer does not trigger unnecessary replay.
