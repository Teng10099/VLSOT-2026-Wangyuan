"""Read-only, model-independent checks for one legacy Mono3DVLT sequence.

Run from the project root, for example::

    python tools/dataset_probe.py --config configs/mono3dvg.yaml \
        --form 1 --split test --sequence 033447 \
        --video-root /data/V2X-Seq-SPD/infrastructure-side/video

The original loader's form selects a *cross-dataset experiment*: form 1 trains
on Rcooper and tests on Rope3D; form 2 does the reverse.  This tool never imports
the original Dataset, BERT, torch, a checkpoint, or model code.  It does not
produce predictions.  Optional --output writes only the diagnostic manifest.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


STYLES = ("YOUTH", "ELDER", "POLICE", "ENGINEER", "CHILD", "TEACHER")
SOURCES = ("Rcooper", "Rope3D")


def active_sources(form: str, split: str) -> tuple[str, ...]:
    """Match the unmodified loader's split-dependent, string-valued form."""
    if form not in ("1", "2", "3"):
        raise ValueError("form must be the string '1', '2', or '3'")
    if split not in ("train", "val", "test"):
        raise ValueError("split must be train, val, or test")
    if form == "3":
        return SOURCES if split == "train" else SOURCES[::-1]
    if split == "train":
        return ("Rcooper",) if form == "1" else ("Rope3D",)
    return ("Rope3D",) if form == "1" else ("Rcooper",)


def portable_basename(value: Any) -> str:
    """Resolve a JSON's Windows/POSIX filename without trusting its old root."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected a nonempty image/calibration path string")
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    if name in ("", ".", ".."):
        raise ValueError(f"invalid basename: {value!r}")
    return name


def read_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("reading YAML requires PyYAML (pip install pyyaml)") from exc
        raw = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or not isinstance(raw.get("dataset"), dict):
        raise ValueError("config must contain a dataset mapping")
    return raw


def source_path(dataset: dict[str, Any], key: str, source: str) -> Path:
    value = dataset.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"dataset.{key} must be [Rcooper_path, Rope3D_path]")
    selected = value[SOURCES.index(source)]
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError(f"dataset.{key} has an invalid {source} path")
    return Path(selected).expanduser()


def read_split(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
           if line.strip()]
    ids = [item[:-5] if item.lower().endswith(".json") else item for item in ids]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate sequence IDs in {path}")
    if any(portable_basename(item) != item for item in ids):
        raise ValueError(f"split list must contain sequence IDs, not paths: {path}")
    return ids


def recursive_json_index(root: Path) -> dict[str, list[Path]]:
    """Index basename IDs, detecting collisions rather than selecting one."""
    if not root.is_dir():
        raise ValueError(f"video root does not exist or is not a directory: {root}")
    found: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*.json")):
        if path.is_file():
            found.setdefault(path.stem, []).append(path)
    return found


def projection_matrix(path: Path, source: str) -> list[list[float]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    key, expected = ("cam_K", 9) if source == "Rcooper" else ("P", 12)
    values = data.get(key) if isinstance(data, dict) else None
    if not isinstance(values, list):
        raise ValueError(f"calibration {path} is missing list {key!r}")
    # Permit both flattened matrices and explicit rows.
    if values and isinstance(values[0], list):
        values = [number for row in values for number in row]
    if len(values) != expected:
        raise ValueError(f"calibration {path}: {key} needs {expected} numbers")
    numbers = [float(number) for number in values]
    if not all(math.isfinite(number) for number in numbers):
        raise ValueError(f"calibration {path} contains nonfinite values")
    if source == "Rcooper":
        return [numbers[0:3] + [0.0], numbers[3:6] + [0.0], numbers[6:9] + [0.0]]
    return [numbers[0:4], numbers[4:8], numbers[8:12]]


def image_size(path: Path) -> list[int] | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    with Image.open(path) as image:
        # Header verification catches corrupt images without creating tensors.
        size = list(image.size)
        image.verify()
    return size


def probe(args: argparse.Namespace) -> dict[str, Any]:
    cfg = read_config(args.config)
    dataset = cfg["dataset"]
    allowed = active_sources(args.form, args.split)
    source = args.source or allowed[0]
    if source not in allowed:
        raise ValueError(f"form {args.form}, split {args.split} selects {allowed}, not {source}")
    root_value = args.root_dir or dataset.get("root_dir")
    if not root_value:
        raise ValueError("dataset.root_dir is missing")
    root_dir = Path(root_value).expanduser()
    image_root = args.image_root or source_path(dataset, "root_img_dir", source)
    calib_root = args.calib_root or source_path(dataset, "calib_path", source)
    video_key = "Rcooper_vedio" if source == "Rcooper" else "Rope3D_vedio"
    video_value = args.video_root or dataset.get(video_key)
    if not video_value:
        raise ValueError(f"dataset.{video_key} is missing")
    video_root = Path(video_value).expanduser()
    split_paths = {name: root_dir / name / (args.split + ".txt") for name in SOURCES}
    ids = read_split(split_paths[source]) if split_paths[source].is_file() else []
    warnings = [f"legacy loader unconditionally opens missing split: {path}"
                for path in split_paths.values() if not path.is_file()]
    if args.sequence_json:
        sequence_path = args.sequence_json
        selected = args.sequence or sequence_path.stem
    else:
        if not ids:
            raise ValueError(f"selected split is missing or empty: {split_paths[source]}")
        selected = args.sequence or ids[0]
        selected = selected[:-5] if selected.lower().endswith(".json") else selected
        if selected not in ids:
            raise ValueError(f"sequence {selected!r} is not in {split_paths[source]}")
        index = recursive_json_index(video_root)
        matches = index.get(selected, [])
        if len(matches) != 1:
            raise ValueError(f"sequence {selected!r} needs exactly one JSON under {video_root}; "
                             f"found {len(matches)}: {[str(path) for path in matches]}")
        sequence_path = matches[0]
        missing_ids = [name for name in ids if name not in index]
        duplicates = {name: [str(path) for path in index[name]]
                      for name in ids if len(index.get(name, [])) > 1}
        if missing_ids:
            warnings.append(f"other split IDs missing from video tree: {missing_ids}")
        if duplicates:
            warnings.append(f"other split IDs have ambiguous JSON basenames: {duplicates}")
        direct_path = video_root / (selected + ".json")
        if sequence_path != direct_path:
            warnings.append("sequence JSON requires recursive lookup; original loader's direct "
                            "join(video_root, sequence_id+'.json') will not find it")
    raw = json.loads(sequence_path.read_text(encoding="utf-8-sig"))
    if source == "Rope3D":
        if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
            raise ValueError("Rope3D sequence must be a nonempty list of annotation objects")
        if len(raw) != 1:
            warnings.append(f"sequence has {len(raw)} annotations; original loader uses only [0]")
        annotation = raw[0]
        description = annotation.get("description")
        selected_style = None
    else:
        if not isinstance(raw, dict):
            raise ValueError("Rcooper sequence must be a JSON object")
        annotation = raw
        descriptions = annotation.get("description")
        if not isinstance(descriptions, dict) or not descriptions:
            raise ValueError("Rcooper description must map styles to strings")
        selected_style = args.style or next(iter(descriptions))
        if selected_style not in descriptions:
            raise ValueError(f"style {selected_style!r} missing; available: {list(descriptions)}")
        description = descriptions[selected_style]
        if set(descriptions) != set(STYLES[:len(descriptions)]):
            warnings.append("style names differ from the original loader's hard-coded style order")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("selected description must be a nonempty string")
    raw_video_id = str(annotation.get("videoID", selected))
    legacy_video_id = source + raw_video_id + (selected_style or "")
    category = str(annotation.get("type", "")).upper()
    if category and category not in ("CAR", "TRUCK", "BUS", "VAN"):
        warnings.append(f"original loader filters or cannot encode category {category!r}")
    frames: list[dict[str, Any]] = []
    errors: list[str] = []
    sizes: set[tuple[int, ...]] = set()
    for frame_index in range(30):
        key = f"frame{frame_index}"
        frame = annotation.get(key)
        if not isinstance(frame, (list, tuple)) or len(frame) < 3:
            errors.append(f"{key} must contain [exist_flag, calibration_path, image_path, ...]")
            continue
        image_path = image_root / portable_basename(frame[2])
        calibration_path = calib_root / portable_basename(frame[1])
        result: dict[str, Any] = {
            "frame_index": frame_index,
            "output_frame_index": frame_index + 1,
            "image": str(image_path),
            "calibration": str(calibration_path),
            "image_exists": image_path.is_file(),
            "calibration_exists": calibration_path.is_file(),
            "legacy_gt_fields_present": len(frame) >= 18,
            "exist_flag": frame[0],
        }
        for kind, path in (("image", image_path), ("calibration", calibration_path)):
            if not path.is_file():
                errors.append(f"{key} missing {kind}: {path}")
        if image_path.is_file():
            try:
                size = image_size(image_path)
                result["image_size_wh"] = size
                if size:
                    sizes.add(tuple(size))
            except Exception as exc:
                errors.append(f"{key} image is unreadable: {exc}")
        if calibration_path.is_file():
            try:
                result["projection_matrix"] = projection_matrix(calibration_path, source)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                errors.append(f"{key} invalid calibration: {exc}")
        frames.append(result)
    if len(sizes) > 1:
        warnings.append("frame image dimensions vary; original tester reuses frame0 image_size")
    matrices = {json.dumps(frame["projection_matrix"]) for frame in frames
                if "projection_matrix" in frame}
    if len(matrices) > 1:
        warnings.append("frame calibration matrices vary; original Dataset returns only the "
                        "last frame's matrix and original tester reuses it for all frames")
    if any(not frame["legacy_gt_fields_present"] for frame in frames):
        warnings.append("GT-free frames cannot be loaded by the original Dataset: it reads "
                        "bbox/size/position/angle through index 17 even in split=test")
    warnings.append("calibration matrices are reported as supplied; this tool does not infer "
                    "whether object coordinates are camera, lidar, or world coordinates")
    return {
        "ok": not errors,
        "config": str(args.config),
        "form": args.form,
        "split": args.split,
        "active_sources": list(allowed),
        "source": source,
        "split_file": str(split_paths[source]),
        "split_count": len(ids),
        "sequence_id": selected,
        "sequence_json": str(sequence_path),
        "raw_video_id": raw_video_id,
        "legacy_video_id": legacy_video_id,
        "legacy_returned_video_id": legacy_video_id[-6:],
        "description_style": selected_style,
        "description": description,
        "distinct_frame_projection_matrices": len(matrices),
        "preprocessing": {
            "resolution_wh": [1920, 1080],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "affine": "legacy get_affine_transform(center=image_size/2, scale=image_size, rot=0)",
        },
        "frames": frames,
        "warnings": warnings,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/mono3dvg.yaml"))
    parser.add_argument("--form", choices=("1", "2", "3"), default="1")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--source", choices=SOURCES)
    parser.add_argument("--sequence", help="literal split ID, preserving any leading zeros")
    parser.add_argument("--sequence-json", type=Path, help="probe an explicit JSON without split lookup")
    parser.add_argument("--style", help="Rcooper description style; defaults to the first actual key")
    parser.add_argument("--root-dir", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--calib-root", type=Path)
    parser.add_argument("--video-root", type=Path, help="recursively indexed video root, not one sequence folder")
    parser.add_argument("--output", type=Path, help="optional new diagnostic JSON; existing files are not overwritten")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = probe(args)
    except (ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
        report = {"ok": False, "errors": [str(exc)]}
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
        except OSError as exc:
            print(f"Cannot write diagnostic output: {exc}")
            return 2
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
