# Model source and notices

The local example uses **sentence-transformers/paraphrase-MiniLM-L3-v2** at the
fixed revision **`4ca70771034acceecb2e72475f72050fcdde4ddc`**.

- Official repository: <https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2>
- Fixed model card: <https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/blob/4ca70771034acceecb2e72475f72050fcdde4ddc/README.md>
- Publisher-declared license: **Apache-2.0**. A copy of that license text is in
  [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
- Model-card credit: trained by Sentence-Transformers. The card references
  Nils Reimers and Iryna Gurevych, *Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks* (2019), <https://arxiv.org/abs/1908.10084>.

The model's license is separate from the adopted license for authorized
`embedding-diagnosis` project contributions. The project's Apache-2.0 grant was
separately approved by its owner; it is not inferred from the model's terms.

## What the downloader retrieves

Model weights are **not included** in the Python wheel, source distribution or
small pilot archive. The supplied download tool retrieves eight files from the
fixed official revision: `config.json`, `model.safetensors`, `tokenizer.json`,
`tokenizer_config.json`, `special_tokens_map.json`, `vocab.txt`,
`sentence_bert_config.json`, and the original `README.md` model card.

The safetensors weights are **69,569,488 bytes**. All eight selected files total
**70,272,180 bytes**; that excludes Torch and other installed dependencies.
`integration/model_source.json` records their sizes and official Git/LFS
identities. The downloader verifies those identities and retains the model card
alongside the downloaded files.

Weights SHA-256:

```text
cf1e4e2d420c664973037c3c73125d7a8fc69952495093ef8f50596f8943a433
```

The fixed model card read during this review is 3,828 bytes, SHA-256
`367a3b024bae0609cd4c9c355d4ef968a974c03d9bde297550508472e5dc0db9`.

This pilot supplies the CPU tokenizer/AutoModel/attention-mask mean-pooling path
and a completed-encoder observation boundary. It does not retrain or change the
model. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the pooling-example
attribution and adaptation description.

If model files are redistributed later, accompany them with the applicable
Apache-2.0 license and preserve their existing attribution/notice material. The
reviewed fixed repository/model-file selection did not supply a separate
`NOTICE` file; do not invent one on behalf of the publisher. Future model
revisions or different models need their own source and terms review.
