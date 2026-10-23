"""Retired historical target-consistency probe.

Release staging 2026-10-23: the original probe embedded a real TRAIN target
and old checkpoint outputs. Those values are intentionally not redistributed.
Use tests/test_camera_targets_v2.py for synthetic geometry roundtrips, and
use tools/evaluate_official_v2.py with legitimately acquired TRAIN data for
measured validation. No target values or replacement measurements are invented.
"""

def main():
    raise SystemExit("Archived probe; see synthetic geometry tests and current v2 validation instructions in README.md.")

if __name__ == "__main__":
    main()
