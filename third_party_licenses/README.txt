Third-party notices for identified inherited components only
Verified: 2026-10-23. Official source URLs and archived-file SHA256 values are in SOURCES.json. License text files are copied from the stated upstream sources; trailing newline/line endings may be normalized. Original copyright/source headers remain in code.

Component mapping:
- DETR / Deformable DETR: Apache-2.0; inherited backbone, position encoding, matcher, utilities and CUDA attention; preserve applicable original file headers.
- MonoDETR: MIT; inherited method/code acknowledgement and model lineage.
- CaDDN: Apache-2.0; inherited depth predictor loss/balancer.
- Microsoft UniLM Diff-Transformer: MIT; local multihead differential-attention related structures are modified derivatives.
- FlashAttention v2.3.6: BSD-3-Clause; corresponding Tri Dao rotary code. AUTHORS is included. Local integration is modified.
- Kornia v0.6.11: Apache-2.0; focal loss/one_hot reference implementation. The local header also references zhezh/focalloss, which has no explicit repository license found. Confirm the actual local inheritance chain; this archive does not relicense that repository.

These notices do not assert that all local code is byte-identical to an upstream version. The inherited framework integration includes modifications whose complete authorship/version history remains to be recorded. The primary Mono3DVG and Mono3DVLT repositories did not expose a project-wide LICENSE/COPYING/NOTICE or README grant in the inspected official trees. Their absence is a remaining clearance issue for adopted parts, not cured by these component texts.

The user has already confirmed the senior's permission for competition, source publication and weights sharing. Record exact author/version/scope and which upstream-derived portions it covers; do not request the same permission again. No project-wide MIT/Apache/BSD license has been assigned by this staging directory. Data and pretrained weight rights are separate.

Local release changes: README/current workflow docs updated; legacy private paths replaced; unsafe legacy split generator retired; old lists moved into legacy/splits. Core full30 candidate runtime files remain unchanged from the tested WORK snapshot. See RELEASE_STATUS.txt and submission_docs/02_THIRD_PARTY_PROVENANCE.json.
