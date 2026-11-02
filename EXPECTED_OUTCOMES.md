# Expected outcomes and historical control

These are records of accepted work from 31 October 2026, not a new test run or
an independent developer result.

| Existing check | match | mismatch | unknown |
| --- | ---: | ---: | ---: |
| Complete local example | 2 | 0 | 2 |
| Separate controlled stored-vector intervention | 1 | 1 | 0 |

The local example writes four synthetic texts. Two have recorded write-time
input evidence; two intentionally do not. The latter are unknown with
`original_embedding_input_unverified` and are not unnecessarily embedded.
For diagnosis the accepted example observed one provider call, one completed
encoder forward and two encoded texts.

The separate negative control copied the index, changed one logical record's
stored vector by `v -> -v` without changing its 384 dimensions, and retained
valid historical input and current encoder evidence. Both persisted
representations of that same logical vector were adjusted consistently. Native
controls were not disabled, and the original index was preserved.

The unchanged result record is
[evidence/controlled_stored_vector_result.json](evidence/controlled_stored_vector_result.json).
At threshold `0.99999`, its control cosine was `0.9999999811521038` and its
intervened-record cosine was `-0.9999999782285408`.

This is a **controlled stored-vector mismatch**. It is not a real model change,
an embedding-path-change experiment or an independent field incident. The file
is supplied as historical result evidence; this handoff does not add a new
negative-control runner or claim a new run.

Your proposed same-dimension path-change check is distinct. In particular,
positive output scaling may still match under cosine, and changing the actual
input string produces unknown when it no longer equals the verified historical
input. See `OBSERVER_AND_SCOPE.md` for those gates and the lifecycle boundary.
