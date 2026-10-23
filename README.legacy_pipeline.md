# Archived legacy pipeline

This repository retains inherited train.py/test.py, tools/batch_infer.py,
tools/export_vlsot.py, tools/export_onnx.py, tools/smoke_infer.py and run_smoke.sh
for source provenance and historical diagnostics. They use the old output
encoding and must not be mixed with the current camera_geometric_center_v2
weights. Old 1029-key exports, world/native trial packages and the severe
localization anomaly of the unaligned inherited checkpoint are historical
observations, not the current candidate's evaluation.

Use README.md for the full30 v2 candidate, fixed official TRAIN split, matching
03_weights, numeric-prior postprocessing and 04_ONNX. Root legacy ID lists have
been moved to legacy/splits/. utils/split.py has been retired so importing it
cannot overwrite the current lists. Private absolute paths were replaced only
in this release copy. The inherited architecture README remains separately in
README.upstream.md; its installation, data and pretrained links are upstream
history, not the current reproduction instructions.

The legacy report generator tools/build_report.py and requirements-report.txt
are historical utilities. Do not use their hardcoded warning or old numbers in
a current report. The current five-page PPT is supplied separately.

The historical target-consistency probe is retired in this public staging copy; its embedded real TRAIN target/output fixture is omitted. Geometry unit tests use explicit synthetic values.
