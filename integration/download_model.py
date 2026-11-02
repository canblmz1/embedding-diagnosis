"""Acquire the official pinned, Apache-2.0 MiniLM model; no alternate hosts."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.txt", "sentence_bert_config.json", "README.md"]


def get(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    target = ROOT / "models/paraphrase-MiniLM-L3-v2"
    evidence = ROOT / "integration/model_source.json"
    if args.download:
        info = json.loads(evidence.read_text())
    else:
        url = f"https://huggingface.co/api/models/{MODEL_ID}?blobs=true"
        raw = get(url)
        model = json.loads(raw)
        assert model["cardData"]["license"] == "apache-2.0"
        siblings = {item["rfilename"]: item for item in model["siblings"]}
        selected = []
        for name in FILES:
            f = siblings[name]
            selected.append({"name": name, "size": f.get("size"), "git_blob": f.get("blobId"),
                             "lfs_sha256": f.get("lfs", {}).get("sha256")})
        info = {"model_id": MODEL_ID, "revision": model["sha"], "license": "apache-2.0",
                "official_model_card": f"https://huggingface.co/{MODEL_ID}/blob/{model['sha']}/README.md",
                "metadata_url": url, "selected_files": selected,
                "planned_total_bytes": sum(f["size"] for f in selected),
                "trained_model": True, "cpu_execution": True,
                "local_inventory": "No model weights found in the task workspace or standard /root/.cache/huggingface; prior indexes contain deterministic research vectors, not a user-trained-model index."}
        evidence.write_text(json.dumps(info, indent=2) + "\n")
        (ROOT / "integration/model_metadata_raw.json").write_bytes(raw)
    if args.download:
        target.mkdir(parents=True, exist_ok=True)
        for f in info["selected_files"]:
            path = target / f["name"]
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{info['revision']}/{f['name']}"
            data = get(url)
            assert len(data) == f["size"], f["name"]
            sha = hashlib.sha256(data).hexdigest()
            if f["lfs_sha256"]:
                assert sha == f["lfs_sha256"], f["name"]
            else:
                blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                assert blob == f["git_blob"], f["name"]
            path.write_bytes(data)
            f["downloaded_sha256"] = sha
            f["source_url"] = url
        (ROOT / "integration/model_download.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
