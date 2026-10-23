"""Conservative numeric priors from an explicitly initial text description.

This module has no dataset, calibration, model, or third-party dependencies.
The range is an initial text anchor, not a constant ground truth for all frames.
Azimuth is deliberately unused: its coordinate convention is not established.
"""

import copy
import math
import re


_NUMBER = r"(?<![\w.,+\-])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])"
_METERS = r"(?:meters?|metres?)\b"
_INITIAL = re.compile(
    r"\b(?:initially|at\s+first|at\s+the\s+(?:start|beginning)|"
    r"in\s+the\s+(?:first|initial)\s+frame|begins?\s+(?:its\s+)?journey)\b",
    re.IGNORECASE,
)
_LATER = re.compile(
    r"\b(?:as\s+(?:the\s+)?(?:video|footage|clip|sequence|scene)\s+"
    r"(?:progresses|continues|unfolds|advances)|"
    r"(?:towards?|by|at|near)\s+(?:the\s+)?(?:end|conclusion)|"
    r"later(?:\s+on)?|subsequently|afterwards|thereafter|now|then|"
    r"during\s+(?:the\s+)?(?:video|footage|clip|sequence)|"
    r"(?:in|at)\s+(?:the\s+)?(?:next|following|final|last|middle)\s+"
    r"(?:frame|stage|part)|over\s+time|"
    r"throughout\s+(?:the\s+)?(?:video|footage|clip|sequence))\b",
    re.IGNORECASE,
)
_RANGE = re.compile(
    r"\bdistance\s+of\s+(?P<distance>" + _NUMBER + r")\s*" + _METERS
    + r"|(?P<away>" + _NUMBER + r")\s*" + _METERS + r"\s+(?:away|from)\b",
    re.IGNORECASE,
)
_DIMENSION_WORDS = {
    "height": r"(?:height|tall)",
    "length": r"(?:length|long)",
    "width": r"(?:width|wide)",
}
_DIMENSIONS = {
    name: re.compile(
        r"(?P<after>" + _NUMBER + r")\s*" + _METERS
        + r"\s+(?:in\s+)?" + words + r"\b|\b" + name
        + r"\s+of\s+(?P<before>" + _NUMBER + r")\s*" + _METERS,
        re.IGNORECASE,
    )
    for name, words in _DIMENSION_WORDS.items()
}
_QUALIFIED_PREFIX = re.compile(
    r"(?:\b(?:not|never|about|approximately|roughly|nearly|over|under)|"
    r"\b(?:at\s+least|at\s+most|up\s+to|more\s+than|less\s+than)|"
    r"\u00b1|\+/-|[\u2010-\u2015\u2212])\s*(?:at\s+a\s+)?$|"
    r"\b(?:between|from)\s+" + _NUMBER + r"(?:\s*" + _METERS
    + r")?\s+(?:and|to)\s*$|"
    + _NUMBER + r"\s*(?:or|to|-|\u00b1|\+/-)\s*$",
    re.IGNORECASE,
)


def _field(matches, text):
    candidates = []
    for match in matches:
        token = next(value for value in match.groupdict().values() if value is not None)
        number = float(token)
        candidates.append({
            "value": number if math.isfinite(number) else None,
            "numeric_token": token,
            "qualified": bool(_QUALIFIED_PREFIX.search(text[:match.start()])),
            "span": [match.start(), match.end()],
            "snippet": text[match.start():match.end()],
        })
    if not candidates:
        status, value = "missing", None
    elif any(c["value"] is None or c["value"] <= 0 for c in candidates):
        status, value = "invalid", None
    elif any(c["qualified"] for c in candidates):
        status, value = "ambiguous", None
    elif len({c["value"] for c in candidates}) != 1:
        status, value = "ambiguous", None
    else:
        status, value = "valid", candidates[0]["value"]
    return {"status": status, "value": value, "candidates": candidates}


def parse_initial_priors(text):
    """Return initial range and dimensions (L, W, H), with auditable spans.

    Only the first paragraph before an explicit later-time marker is examined.
    An explicit initial anchor is required for the range. Dimensions describe
    the initial subject and may be extracted without a time word. Conflicting,
    incomplete, qualified, nonpositive, or nonfinite fields are never adopted. Repeated
    candidates with exactly the same value are allowed and remain in evidence.
    Spans are character offsets into the original input string.
    """
    result = {
        "range_m": None,
        "dimensions_lwh": None,
        "status": "missing",
        "ambiguity": {"range": False, "dimensions": False},
        "initial_span": [0, 0],
        "initial_snippet": "",
        "initial_anchor": None,
        "scope": "initial_only",
        "fields": {},
    }
    if not isinstance(text, str):
        result["status"] = "invalid_input"
        return result

    paragraph_end = re.search(r"\r?\n[ \t]*\r?\n", text)
    end = paragraph_end.start() if paragraph_end else len(text)
    later = _LATER.search(text, 0, end)
    if later:
        end = later.start()
    initial = text[:end]
    result["initial_span"] = [0, end]
    result["initial_snippet"] = initial
    anchor = _INITIAL.search(initial)
    if anchor:
        result["initial_anchor"] = {
            "span": [anchor.start(), anchor.end()],
            "snippet": anchor.group(),
        }

    range_field = _field(_RANGE.finditer(initial), initial)
    if range_field["status"] == "valid" and not anchor:
        range_field["status"], range_field["value"] = "unanchored", None
    fields = {"range": range_field}
    fields.update({name: _field(pattern.finditer(initial), initial)
                   for name, pattern in _DIMENSIONS.items()})
    result["fields"] = fields
    result["range_m"] = range_field["value"]
    dims = [fields[name] for name in ("length", "width", "height")]
    if all(field["status"] == "valid" for field in dims):
        result["dimensions_lwh"] = [field["value"] for field in dims]
    result["ambiguity"] = {
        "range": range_field["status"] == "ambiguous",
        "dimensions": any(field["status"] == "ambiguous" for field in dims),
    }
    available = (result["range_m"] is not None, result["dimensions_lwh"] is not None)
    if all(available):
        result["status"] = "valid"
    elif any(available):
        result["status"] = "partial"
    elif any(result["ambiguity"].values()):
        result["status"] = "ambiguous"
    elif any(field["status"] == "invalid" for field in fields.values()):
        result["status"] = "invalid"
    return result


def _copied_vector(value):
    try:
        return copy.deepcopy(list(value))
    except TypeError:
        return copy.deepcopy(value)


def _finite_vector(value):
    try:
        if len(value) != 3 or any(isinstance(v, (bool, str, bytes)) for v in value):
            return None
        numbers = [float(v) for v in value]
        return numbers if all(math.isfinite(v) for v in numbers) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _positive_number(value):
    if isinstance(value, (bool, str, bytes)):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def _apply_vector_numeric_priors(native_xyz, dimensions_lwh, priors, *,
                                use_range=True, use_dimensions=False):
    """Copy predictions and optionally apply valid parsed priors.

    Range changes only the norm of native_xyz, preserving its direction. No
    coordinate conversion or temporal extrapolation is performed. This function
    does not mutate its inputs, and returns reasons for every applied/fallback
    field. Parsed evidence is required; a hand-written number alone is ignored.
    """
    result = {
        "native_xyz": _copied_vector(native_xyz),
        "dimensions_lwh": _copied_vector(dimensions_lwh),
        "range_applied": False,
        "dimensions_applied": False,
        "reasons": [],
    }
    priors = priors if isinstance(priors, dict) else {}
    fields = priors.get("fields")
    fields = fields if isinstance(fields, dict) else {}
    range_field = fields.get("range")
    range_field = range_field if isinstance(range_field, dict) else {}
    distance = _positive_number(priors.get("range_m"))
    xyz = _finite_vector(result["native_xyz"])
    if not use_range:
        result["reasons"].append("range: disabled")
    elif distance is None or range_field.get("status") != "valid":
        result["reasons"].append("range: no valid initial range (%s)" %
                                 range_field.get("status", "missing"))
    elif xyz is None:
        result["reasons"].append("range: native_xyz must contain three finite numbers")
    else:
        norm = math.hypot(*xyz)
        if not math.isfinite(norm) or norm == 0:
            result["reasons"].append("range: native_xyz has zero or nonfinite norm")
        else:
            scaled = [v / norm * distance for v in xyz]
            if all(math.isfinite(v) for v in scaled) and math.hypot(*scaled) > 0:
                result["native_xyz"] = scaled
                result["range_applied"] = True
                result["reasons"].append("range: applied initial text range; direction preserved")
            else:
                result["reasons"].append("range: scaling produced invalid coordinates")

    parsed_dims = _finite_vector(priors.get("dimensions_lwh"))
    original_dims = _finite_vector(result["dimensions_lwh"])
    dim_fields = [fields.get(name) for name in ("length", "width", "height")]
    valid_dim_fields = all(isinstance(f, dict) and f.get("status") == "valid"
                           for f in dim_fields)
    if not use_dimensions:
        result["reasons"].append("dimensions: disabled")
    elif parsed_dims is None or any(v <= 0 for v in parsed_dims) or not valid_dim_fields:
        result["reasons"].append("dimensions: no valid complete LWH text prior")
    elif original_dims is None:
        result["reasons"].append("dimensions: input must contain three finite numbers")
    else:
        result["dimensions_lwh"] = parsed_dims
        result["dimensions_applied"] = True
        result["reasons"].append("dimensions: applied semantic LWH text prior")
    return result


# Both spellings refer to the same conservative parser.
parse_initial_numeric_priors = parse_initial_priors


def apply_numeric_priors(decoded, prior, *, use_range=True, use_dimensions=True):
    """Return a fresh decoded prediction with synchronized native XYZ and LWH.

    ``decoded`` follows infer_official_v2.decode_prediction. Applied fields are
    synchronized into values_lwhxyz=[L,W,H,X,Y,Z], dimensions_lwh and
    label_position_native_xyz. Details are in result['numeric_priors'].

    Scaling native XYZ invalidates derived camera geometry. Without an explicit
    recalculation by the caller, these fields are removed and camera metrics
    marked unavailable. A dimensions-only change preserves camera geometry.
    This function is frame agnostic: the caller must choose when an initial
    anchor is applicable and must not assume it describes later motion.
    """
    if not isinstance(decoded, dict):
        raise TypeError("decoded must be a prediction dictionary")
    result = copy.deepcopy(decoded)
    xyz = _copied_vector(decoded.get("label_position_native_xyz"))
    dims = _copied_vector(decoded.get("dimensions_lwh"))
    if not isinstance(xyz, list) or len(xyz) != 3 or not isinstance(dims, list) or len(dims) != 3:
        result["numeric_priors"] = {
            "range_applied": False, "dimensions_applied": False,
            "reasons": ["decoded native XYZ and LWH must each contain three values"],
            "scope": "initial_only",
        }
        return result
    applied = _apply_vector_numeric_priors(xyz, dims, prior, use_range=use_range,
                                          use_dimensions=use_dimensions)
    result["label_position_native_xyz"] = applied["native_xyz"]
    result["dimensions_lwh"] = applied["dimensions_lwh"]
    result["values_lwhxyz"] = (copy.deepcopy(applied["dimensions_lwh"])
                                + copy.deepcopy(applied["native_xyz"]))
    result["numeric_priors"] = {
        key: applied[key] for key in ("range_applied", "dimensions_applied", "reasons")
    }
    result["numeric_priors"]["scope"] = "initial_only"
    if applied["range_applied"]:
        for key in ("camera_center_xyz", "camera_z", "normalized_center_xy"):
            result.pop(key, None)
        result["camera_metrics_available"] = False
    else:
        valid_camera = (_finite_vector(result.get("camera_center_xyz")) is not None
                        and _positive_number(result.get("camera_z")) is not None)
        result["camera_metrics_available"] = bool(
            result.get("camera_metrics_available", valid_camera) and valid_camera)
    return result
