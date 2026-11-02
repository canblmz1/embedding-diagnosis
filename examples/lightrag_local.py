"""One complete, controlled native LightRAG usage example with a trained model.

Run the installed package, not a research harness. The supplied model directory
is local and already downloaded. A new work directory is required so existing
application data cannot be overwritten. Two older records deliberately have no
write-time evidence; two later writes explicitly opt into prospective capture.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import threading

import numpy as np
from transformers import AutoTokenizer
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id, sanitize_text_for_encoding

from embedding_diagnosis import diagnose, summarize
from embedding_diagnosis.lightrag import LightRAGAdapter
from embedding_diagnosis.local_encoder import LocalTransformerEncoder


MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L3-v2@4ca70771034acceecb2e72475f72050fcdde4ddc"
THRESHOLD = 0.99999
CALIBRATION_TEXTS = ["Rainfall is measured in millimetres.",
                     "A compiler translates source code into executable instructions."]
OLDER_TEXTS = ["The local library opens at nine o'clock on weekday mornings.",
               "An electric motor converts electrical energy into mechanical motion."]
OBSERVED_TEXTS = ["The museum preserves photographs and handwritten letters.",
                  "A database transaction groups several operations into one logical unit."]


class ApplicationTokenizer:
    """Real HF tokenizer through LightRAG's documented tokenizer interface."""

    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True,
                                                       trust_remote_code=False)
        self.lock = threading.Lock()

    def encode(self, text):
        with self.lock:
            return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        with self.lock:
            return self.tokenizer.decode(tokens)

    def __deepcopy__(self, memo):
        return self


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file() and not p.name.endswith(".lock")}


def delta(before, after):
    return {key: after[key] - value for key, value in before.items()}


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def llm_not_configured(*args, **kwargs):
    raise RuntimeError("This chunk-storage example does not configure LLM extraction or generation.")


async def run(model_dir, work_dir):
    work_dir.mkdir(parents=True, exist_ok=False)
    index_dir = work_dir / "index"
    protocol = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                "controlled_example": True, "independent_user_index": False,
                "model_name": MODEL_NAME, "threshold_fixed_before_calls": THRESHOLD,
                "calibration_texts": CALIBRATION_TEXTS,
                "expected": {"older_without_proof": "unknown", "prospectively_observed": "match"},
                "scope": "Exact CPU float32 local MiniLM document path and Nano Float16 readback; no universal tolerance or retrieval-quality claim."}
    write_json(work_dir / "protocol.json", protocol)
    encoder = LocalTransformerEncoder(model_dir, max_length=128)
    # Explicit CPU-only calibration setting; no throughput claim is made.
    encoder.torch.set_num_threads(1)
    tokenizer = Tokenizer(MODEL_NAME, ApplicationTokenizer(str(model_dir)))

    def application():
        return LightRAG(
            working_dir=str(index_dir), workspace="local_alpha", tokenizer=tokenizer,
            llm_model_func=llm_not_configured,
            embedding_func=EmbeddingFunc(embedding_dim=encoder.embedding_dim, func=encoder,
                model_name=MODEL_NAME, supports_asymmetric=True, max_token_size=128),
            embedding_batch_num=16,
        )

    writer = None
    reader = None
    phases = {}
    try:
        writer = application()
        await writer.initialize_storages()
        before = encoder.counts()
        calibration_calls = []
        outputs = []
        for _ in range(2):
            with encoder.capture() as observed:
                outputs.append(await writer.chunks_vdb.embedding_func(CALIBRATION_TEXTS, context="document"))
            calibration_calls.append(asdict(observed))
        repeated = [cosine(a, b) for a, b in zip(outputs[0], outputs[1], strict=True)]
        quantized = [cosine(v, v.astype(np.float16).astype(np.float32)) for v in outputs[0]]
        calibration = {"threshold": THRESHOLD, "selected_before_calibration": True,
                       "repeat_cosines": repeated, "float16_roundtrip_cosines": quantized,
                       "maximum_observed_cosine_loss": max(1.0 - min(repeated), 1.0 - min(quantized)),
                       "tolerance_cosine_loss_budget": 1.0 - THRESHOLD,
                       "scope": "Two held-out calibration texts, repeated same CPU encoder and native Float16 storage transform; not cross-model calibration.",
                       "observations": calibration_calls}
        write_json(work_dir / "calibration.json", calibration)
        if min(repeated + quantized) < THRESHOLD:
            raise RuntimeError("This environment exceeds the predeclared example tolerance; do not silently relax it.")
        phases["calibration"] = delta(before, encoder.counts())

        before = encoder.counts()
        await writer.ainsert_custom_kg({"chunks": [
            {"content": text, "source_id": f"older-source-{i}", "file_path": f"older-{i}.txt"}
            for i, text in enumerate(OLDER_TEXTS)]})
        phases["earlier_native_writes_without_capture"] = delta(before, encoder.counts())

        before = encoder.counts()
        provenance = {}
        observed_writes = []
        for i, text in enumerate(OBSERVED_TEXTS):
            stored_text = sanitize_text_for_encoding(text)
            native_id = compute_mdhash_id(stored_text, prefix="chunk-")
            with encoder.capture() as observed:
                await writer.ainsert_custom_kg({"chunks": [{"content": text,
                    "source_id": f"observed-source-{i}", "file_path": f"observed-{i}.txt"}]})
            # One known native write, one input position. Text is not used to
            # find an event or distinguish one record from another.
            if (len(observed.provider_calls) != 1 or len(observed.events) != 1
                    or observed.events[0]["input_position"] != 0
                    or observed.provider_calls[0]["input_count"] != 1):
                raise RuntimeError("Cannot establish this single-record native write's input position.")
            event = observed.events[0]
            native_records = await writer.chunks_vdb.get_by_ids([native_id])
            if len(native_records) != 1 or native_records[0]["content"] != stored_text:
                raise RuntimeError("Native write/read identity does not match the declared record.")
            provenance[native_id] = {
                "source_text_sha256": hashlib.sha256(stored_text.encode()).hexdigest(),
                "embedding_input": event["embedding_input"], "input_sha256": event["input_sha256"],
                "role": "document", "reconstruction_verified": True,
                "preprocessing": "Observed native custom-KG sanitization and ordered local tokenizer/mean-pooling document call; no prefix in this example.",
                "evidence_ref": f"write_observations.json#{event['event_id']}",
                "write_call_id": observed.call_id, "provider_call_id": event["provider_call_id"],
                "input_position": 0, "native_record_id": native_id,
                "encoder_configuration_sha256": encoder.configuration_sha256,
                "tokenized_row": event["tokenized_row"],
            }
            observed_writes.append({"native_record_id": native_id, "single_record_write": True,
                                    "observation": asdict(observed)})
        phases["prospective_native_writes_with_capture"] = delta(before, encoder.counts())
        write_json(work_dir / "write_observations.json", observed_writes)
        write_json(work_dir / "input_evidence.json", provenance)
        write_json(work_dir / "encoder_configuration.json", encoder.configuration)
        await writer.finalize_storages()
        writer = None

        reader = application()
        await reader.initialize_storages()
        ids = [compute_mdhash_id(sanitize_text_for_encoding(text), prefix="chunk-")
               for text in OLDER_TEXTS + OBSERVED_TEXTS]
        native_before = file_hashes(index_dir)
        before = encoder.counts()
        adapter = LightRAGAdapter(reader.chunks_vdb, observer=encoder,
                                  provenance=work_dir / "input_evidence.json")
        # The actual installed API. No experiment classes or paths participate.
        with encoder.torch.profiler.profile(activities=[encoder.torch.profiler.ProfilerActivity.CPU]) as cpu_trace:
            report = await diagnose(adapter, ids, threshold=THRESHOLD)
        phases["diagnosis"] = delta(before, encoder.counts())
        native_after = file_hashes(index_dir)
        write_json(work_dir / "report.json", report)
        (work_dir / "summary.txt").write_text(summarize(report) + "\n", encoding="utf-8")
        operators = {e.key: e.count for e in cpu_trace.key_averages()
                     if e.key in {"aten::linear", "aten::matmul", "aten::addmm", "aten::mm", "aten::bmm"}}
        usage = {"controlled_usage_example": True, "existing_user_index": False,
                 "native_write_api": "LightRAG.ainsert_custom_kg(chunks only)",
                 "diagnostic_api": "diagnose(LightRAGAdapter(initialized_rag.chunks_vdb,...), ids, threshold=...)",
                 "historical_subset_ids": ids[:2], "prospective_subset_ids": ids[2:],
                 "counts_by_phase": phases, "total_encoder_counts": encoder.counts(),
                 "diagnosis_cpu_tensor_operator_counts": operators,
                 "index_hashes_before": native_before, "index_hashes_after": native_after,
                 "index_unchanged_during_diagnosis": native_before == native_after,
                 "native_private_fields_read": list(adapter._GUARDS),
                 "observer_scope": "Local completed BERT encoder forward, per-position token tensors and returned vector digest; no remote-cache claim.",
                 "prospective_evidence_requirement": "Only writes explicitly captured at the time supply original-input proof; today's settings are not historical proof.",
                 "cpu_threads": encoder.torch.get_num_threads(),
                 "llm_boundary": "Native SDK requires a callable; this example supplies a fail-loud unconfigured function. Chunk insertion/diagnosis do not invoke it.",
                 "package_versions": {name: importlib.metadata.version(name)
                                      for name in ["embedding-diagnosis", "lightrag-hku", "nano-vectordb", "numpy", "torch", "transformers"]},
                 "adapter_observations": adapter.observations}
        write_json(work_dir / "usage_record.json", usage)
        if native_before != native_after:
            raise RuntimeError("The index changed during diagnosis.")
        if [row["status"] for row in report["samples"]] != ["unknown", "unknown", "match", "match"]:
            raise RuntimeError("The controlled example did not produce its predeclared outcomes; inspect report.json.")
        if phases["diagnosis"] != {"provider_calls": 1, "encoder_forwards": 1, "encoded_texts": 2} or not operators:
            raise RuntimeError("Expected local model execution evidence is missing.")
        print(summarize(report))
        print(json.dumps({"report": str(work_dir / "report.json"), "index_unchanged": True,
                          "diagnostic_counts": phases["diagnosis"]}, indent=2))
    finally:
        if writer is not None:
            await writer.finalize_storages()
        if reader is not None:
            await reader.finalize_storages()
        encoder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.model_dir.resolve(), args.work_dir.resolve()))


if __name__ == "__main__":
    main()
