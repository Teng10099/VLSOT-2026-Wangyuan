r"""Audit clip provenance without reading target fields or loading images/models.

Example (run from the project root)::

    python tools/audit_training_inputs.py --train-split source/Rope3D/train.txt \
        --video-root /data/V2X-Seq-SPD/infrastructure-side/video \
        --official-test-split source/Rope3D/test.txt --output training_audit.json

Only ``videoID`` and ``frame0..29[2]`` (image paths) are used.  Frame arrays may
contain just their first three entries; GT boxes, dimensions and coordinates
are neither required nor interpreted.  Image identity is the case-sensitive
basename, matching the original loader's image-root/basename join.  This audit
does not verify image availability, split officialness, labels or model quality.

Two clips are inseparable for validation when they share an input image OR
belong to the same ``sequence_NNNN`` recording directory.  Unknown provenance
is explicitly reported.  The command never writes or chooses an actual split.
An invalid audit exits 2; a valid audit with insufficient independent groups
still exits 0 and explains the limitation in its JSON report.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Set


SCHEMA_VERSION = 1
FRAME_COUNT = 30
ID_PATTERN = re.compile(r"^[0-9]{6}$")
SCENE_PATTERN = re.compile(r"^sequence_[0-9]+$", re.IGNORECASE)


def read_split(path: Path) -> List[str]:
    """Read unique six-digit IDs, preserving leading zeros; reject empty input."""
    if not path.is_file():
        raise ValueError("missing split file: {}".format(path))
    ids = []
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        clip_id = line.strip()
        if not clip_id:
            continue
        if clip_id.lower().endswith(".json"):
            clip_id = clip_id[:-5]
        if not ID_PATTERN.fullmatch(clip_id):
            raise ValueError("invalid six-digit clip ID at {}:{}: {!r}".format(
                path, line_no, clip_id))
        if clip_id in seen:
            raise ValueError("duplicate clip ID {!r} in {}".format(clip_id, path))
        ids.append(clip_id)
        seen.add(clip_id)
    if not ids:
        raise ValueError("empty split: {}".format(path))
    return ids


def _unique_object(pairs: List[Any]) -> Dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: {!r}".format(key))
        result[key] = value
    return result


def _image_basename(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("image path must be a nonempty string")
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    if name in ("", ".", ".."):
        raise ValueError("image path has an invalid basename: {!r}".format(value))
    return name


def _video_id(value: Any) -> str:
    if isinstance(value, bool):
        raise ValueError("videoID must be a six-digit ID or its integer value")
    if isinstance(value, int) and 0 <= value <= 999999:
        return "{:06d}".format(value)
    if isinstance(value, str) and value.isascii() and value.isdigit() and len(value) <= 6:
        return value.zfill(6)
    raise ValueError("videoID must be a six-digit ID or its integer value")


def _read_clip(path: Path, clip_id: str) -> Dict[str, Any]:
    """Extract metadata only; no numerical target field is accessed."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    except (OSError, ValueError) as exc:
        raise ValueError("cannot read {}: {}".format(path, exc)) from exc
    if isinstance(raw, list):
        if len(raw) != 1:
            raise ValueError("ambiguous annotation list in {}: expected one object".format(path))
        raw = raw[0]
    if not isinstance(raw, dict):
        raise ValueError("clip JSON must contain one object: {}".format(path))
    if "videoID" not in raw:
        raise ValueError("missing videoID in {}".format(path))
    if _video_id(raw["videoID"]) != clip_id:
        raise ValueError("videoID does not match indexed clip ID {} in {}".format(clip_id, path))
    frame_names = []
    for index in range(FRAME_COUNT):
        key = "frame{}".format(index)
        frame = raw.get(key)
        if not isinstance(frame, list) or len(frame) < 3:
            raise ValueError("missing or invalid {} image path in {}".format(key, path))
        try:
            frame_names.append(_image_basename(frame[2]))
        except ValueError as exc:
            raise ValueError("{} in {}: {}".format(key, path, exc)) from exc
    if len(set(frame_names)) != len(frame_names):
        raise ValueError("duplicate image paths within clip {}".format(clip_id))
    # Inspect the lexical source path: resolving symlinks first could lose the
    # supplied recording directory's provenance.
    scenes = {part.lower() for part in path.parts if SCENE_PATTERN.fullmatch(part)}
    if len(scenes) > 1:
        raise ValueError("ambiguous recording directories for {}: {}".format(
            path, sorted(scenes)))
    return {"clip_id": clip_id, "json_path": str(path.resolve()),
            "scene": next(iter(scenes)) if scenes else None,
            "image_names": frame_names}


def _index_clips(video_root: Path, requested: Set[str]) -> Dict[str, Dict[str, Any]]:
    if not video_root.is_dir():
        raise ValueError("missing video root directory: {}".format(video_root))
    matches = defaultdict(list)
    # Do not index one guessed sequence directory.  Duplicate requested names
    # anywhere in the supplied tree are errors, even when one looks convenient.
    for path in sorted(video_root.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".json" and path.stem in requested:
            matches[path.stem].append(path)
    missing = sorted(requested - set(matches))
    ambiguous = {key: [str(path) for path in value]
                 for key, value in sorted(matches.items()) if len(value) != 1}
    if missing or ambiguous:
        raise ValueError("JSON index rejected: missing IDs {}; ambiguous IDs {}".format(
            missing, ambiguous))
    resolved = {}
    clips = {}
    for clip_id in sorted(requested):
        path = matches[clip_id][0]
        canonical = str(path.resolve())
        if canonical in resolved:
            raise ValueError("duplicate canonical JSON path for IDs {} and {}: {}".format(
                resolved[canonical], clip_id, canonical))
        resolved[canonical] = clip_id
        clips[clip_id] = _read_clip(path, clip_id)
    return clips


class _UnionFind:
    def __init__(self, ids: List[str]) -> None:
        self.parent = {clip_id: clip_id for clip_id in ids}

    def find(self, clip_id: str) -> str:
        while self.parent[clip_id] != clip_id:
            self.parent[clip_id] = self.parent[self.parent[clip_id]]
            clip_id = self.parent[clip_id]
        return clip_id

    def join(self, left: str, right: str) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _components(clips: List[Dict[str, Any]], include_scene: bool) -> List[Dict[str, Any]]:
    union = _UnionFind([clip["clip_id"] for clip in clips])
    first_image = {}
    first_scene = {}
    for clip in clips:
        clip_id = clip["clip_id"]
        for name in clip["image_names"]:
            if name in first_image:
                union.join(clip_id, first_image[name])
            else:
                first_image[name] = clip_id
        if include_scene and clip["scene"] is not None:
            scene = clip["scene"]
            if scene in first_scene:
                union.join(clip_id, first_scene[scene])
            else:
                first_scene[scene] = clip_id
    groups = defaultdict(list)
    for clip in clips:
        groups[union.find(clip["clip_id"])].append(clip)
    results = []
    for _, members in sorted(groups.items()):
        results.append({"clip_ids": sorted(clip["clip_id"] for clip in members),
                        "clip_count": len(members),
                        "unique_frame_count": len({name for clip in members
                                                   for name in clip["image_names"]}),
                        "scene_ids": sorted({clip["scene"] for clip in members
                                             if clip["scene"] is not None})})
    return results


def _summarize(clips: List[Dict[str, Any]]) -> Dict[str, Any]:
    scenes = defaultdict(list)
    for clip in clips:
        scenes[clip["scene"]].append(clip)
    per_scene = []
    for scene, members in sorted(scenes.items(), key=lambda pair: pair[0] or ""):
        per_scene.append({"scene": scene, "clip_count": len(members),
                          "unique_frame_count": len({name for clip in members
                                                     for name in clip["image_names"]}),
                          "shared_frame_components": _components(members, False)})
    return {"clip_count": len(clips),
            "unique_frame_count": len({name for clip in clips for name in clip["image_names"]}),
            "scenes": per_scene,
            "shared_frame_components": _components(clips, False),
            "leakage_groups": _components(clips, True),
            "unknown_scene_clip_ids": sorted(clip["clip_id"] for clip in clips
                                             if clip["scene"] is None)}


def audit_inputs(train_split: Path, video_root: Path,
                 official_test_split: Optional[Path] = None) -> Dict[str, Any]:
    """Return deterministic metadata/provenance statistics; invalid inputs raise ValueError."""
    train_split, video_root = Path(train_split), Path(video_root)
    train_ids = read_split(train_split)
    test_ids = read_split(Path(official_test_split)) if official_test_split is not None else []
    indexed = _index_clips(video_root, set(train_ids) | set(test_ids))
    training = [indexed[clip_id] for clip_id in sorted(train_ids)]
    testing = [indexed[clip_id] for clip_id in sorted(test_ids)]
    summary = _summarize(training)
    train_frames = {name for clip in training for name in clip["image_names"]}
    test_frames = {name for clip in testing for name in clip["image_names"]}
    shared_frames = train_frames & test_frames
    id_overlap = sorted(set(train_ids) & set(test_ids))
    checked = official_test_split is not None
    test_clean = not (id_overlap or shared_frames) if checked else None
    group_count = len(summary["leakage_groups"])
    scene_complete = not summary["unknown_scene_clip_ids"]
    can_split = group_count >= 2
    if not can_split:
        status = "insufficient_independent_groups"
        reason = ("Only one recording/shared-frame group is available; independent validation "
                  "of generalization is not possible. Do not use a random frame or clip split.")
    elif not scene_complete:
        status = "unknown_recording_provenance"
        reason = ("At least two metadata groups exist, but some original recording directories "
                  "are unknown; scene-independent validation is not established.")
    elif test_clean is False:
        status = "official_test_overlap"
        reason = "Training inputs overlap the supplied official test split; resolve provenance first."
    elif not checked:
        status = "official_test_not_checked"
        reason = ("At least two disjoint recording/shared-frame groups exist; official-test "
                  "overlap has not been checked. The supplied training split's officialness is unverified.")
    else:
        status = "groups_available"
        reason = ("At least two disjoint recording/shared-frame groups exist and no supplied "
                  "official-test ID/image overlap was found. Keep each group wholly on one side "
                  "of a training-only train/validation split; officialness is not inferred from filenames.")
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {"train_split": str(train_split.resolve()),
                   "video_root": str(video_root.resolve()),
                   "official_test_split": str(Path(official_test_split).resolve()) if checked else None},
        "rules": {"frame_count_per_clip": FRAME_COUNT,
                  "fields_used": ["videoID", "frame0..29[2]"],
                  "image_identity": "case-sensitive basename, matching the original image loader",
                  "scene_identity": "sequence_NNNN directory in the clip JSON path",
                  "leakage_group": "transitive union of shared images OR the same known recording",
                  "image_files_checked": False,
                  "split_officialness_verified": False},
        "clip_index": [indexed[clip_id] for clip_id in sorted(indexed)],
        "training": summary,
        "official_test": _summarize(testing) if checked else None,
        "official_test_overlap": {
            "checked": checked, "id_overlap": id_overlap, "frame_overlap": sorted(shared_frames),
            "train_clip_ids": sorted(clip["clip_id"] for clip in training
                                     if shared_frames.intersection(clip["image_names"])),
            "test_clip_ids": sorted(clip["clip_id"] for clip in testing
                                    if shared_frames.intersection(clip["image_names"]))},
        "safe_validation": {"group_count": group_count, "can_split_by_group": can_split,
                            "scene_identity_complete": scene_complete,
                            "official_test_clean": test_clean,
                            "ready_for_independent_validation": can_split and scene_complete and test_clean is True,
                            "status": status, "reason": reason}}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-split", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--official-test-split", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON report path; an existing file is never overwritten")
    args = parser.parse_args(argv)
    try:
        report = audit_inputs(args.train_split, args.video_root, args.official_test_split)
        # Exclusive creation protects prior reports and accidental input paths.
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        print("audit rejected: {}".format(exc), file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output.resolve()),
                      "training_clips": report["training"]["clip_count"],
                      "unique_images": report["training"]["unique_frame_count"],
                      "safe_validation": report["safe_validation"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
