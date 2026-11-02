# Third-party notices and license status

This developer handoff contains the unchanged `embedding-diagnosis 0.1.0a1`
wheel and source distribution. The authorized project contributions are licensed
under Apache-2.0 by Cuma Can Bilmez; see `LICENSE` and `LICENSE_SCOPE.md` for
coverage. The following third-party terms and attributions remain applicable.
They do not establish ownership of the project's other contributions.

## Bundled comparator: embedguard attribution

The Python/NumPy `embedding_diagnosis/_comparison.py` function was carried forward
from the project's earlier `shared/numpy_baseline.py` and
`experiment/native_baseline.py`. That earlier helper was developed and compared
against **embedguard v0.1.0**, commit
`659d837849a86aad9bce86ca34f66e6a15cb522c`. Its probe mapping, missing/dimension
cases, threshold decision and result schema closely follow embedguard's
`checkDrift` interface. We conservatively retain the upstream MIT attribution
for this Python adaptation; we do not claim a clean-room implementation.

- Upstream: <https://github.com/jayblast-spec/embedguard>
- Reviewed source: <https://github.com/jayblast-spec/embedguard/blob/659d837849a86aad9bce86ca34f66e6a15cb522c/src/checks.ts>
- Related cosine source: <https://github.com/jayblast-spec/embedguard/blob/659d837849a86aad9bce86ca34f66e6a15cb522c/src/similarity.ts>
- Copyright notice: **Copyright (c) 2026 jayblast-spec**.
- License: MIT. The complete original notice is in
  [licenses/embedguard-MIT.txt](licenses/embedguard-MIT.txt).
- Adaptation: Python/NumPy instead of TypeScript; the old research-only runner
  was removed from the package. The numerical function was retained in the
  accepted alpha. No embedguard TypeScript runtime is bundled.

Keep that MIT copyright and permission notice with copies or substantial
portions of this comparator, including when distributing it within the wheel
or source distribution.

## Local encoder and model-card example

`embedding_diagnosis/local_encoder.py` uses the attention-mask mean-pooling
recipe shown in the official `sentence-transformers/paraphrase-MiniLM-L3-v2`
model card. The tensor expression closely follows that example. We retain this
attribution conservatively rather than claim that all implementation expression
is original. The reviewed model card declares **Apache-2.0** and credits
Sentence-Transformers for training the model.

- Pinned model/card revision: `4ca70771034acceecb2e72475f72050fcdde4ddc`.
- Source: <https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/blob/4ca70771034acceecb2e72475f72050fcdde4ddc/README.md>
- Adaptation notice: the alpha embeds the pooling expression in a CPU,
  `inference_mode` callable using `last_hidden_state`; it adds local-file loading,
  explicit maximum length, ordered inputs, output checks, and completed-encoder
  observation. It is not the model publisher's original example or an endorsed
  implementation.
- Apache License 2.0 text: [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
- Model source, download scope and distinct model terms: [MODEL_NOTICE.md](MODEL_NOTICE.md).

The wheel and sdist remain byte-identical historical artifacts. Their authorized
project contributions are covered by the accompanying grant in `LICENSE_SCOPE.md`;
the notices above identify the adapted components and preserve their original
terms. Keep the distributions together with the adopted license and notice
materials. No third-party endorsement or reassignment of rights is implied.

## Separately installed dependencies

The pilot does **not** bundle LightRAG, NanoVectorDB, NumPy, Torch, Transformers,
their dependency wheels, or model weights. They are installed or downloaded
separately using the supplied pins/tool and remain governed by their own terms.
The following direct-component references are informational, not a claim that
their licenses license `embedding-diagnosis`:

| Component used by the example | Reviewed license/source |
| --- | --- |
| NumPy | BSD-3-Clause; <https://numpy.org/doc/stable/license.html> |
| LightRAG, pinned commit `aaebfe1bb8d55deb6a4304d357a3bf4c92fcd40a` | MIT, Copyright (c) 2025 LightRAG Team; <https://github.com/HKUDS/LightRAG/blob/aaebfe1bb8d55deb6a4304d357a3bf4c92fcd40a/LICENSE> |
| NanoVectorDB | MIT; <https://github.com/gusye1234/nano-vectordb/blob/main/LICENSE> |
| Transformers 4.57.3 | Apache-2.0; <https://github.com/huggingface/transformers/blob/v4.57.3/LICENSE> |
| Torch 2.5.1 | BSD-style license with additional bundled-component notices; <https://github.com/pytorch/pytorch/blob/v2.5.1/LICENSE> and <https://github.com/pytorch/pytorch/blob/v2.5.1/NOTICE> |

Installed dependency distributions contain their respective license and notice
files. The runtime pin list also includes transitive dependencies; this short
table is not a license inventory for redistributing a complete Python
environment. If such an environment is later redistributed, retain and review
the licenses/notices of those actual redistributed artifacts.

No endorsement by a third-party author or project is claimed.
