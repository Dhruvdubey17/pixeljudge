"""Loading subjective datasets and assembling the feature table.

The primary dataset is the **Netflix Public Dataset (NFLX-P)**. It is the right
choice for this project for one specific reason: it is genuinely full-reference.
A handful of pristine reference clips, each degraded by compression and scaling
(the same two distortions our own pipeline applies), with a human DMOS per
degraded version. It is also what VMAF itself was validated on.

The popular large datasets (KoNViD-1k, YouTube-UGC, LIVE-VQC, LSVQ) cannot be used
here at all: they are "in the wild" clips with no pristine original, so there is
nothing to compute PSNR/SSIM/VMAF *against*. That is a property of the data, not a
limitation of the code.

The dataset ships as a Python file listing its reference and distorted videos with
their DMOS. We read it with :mod:`ast` and ``literal_eval`` rather than importing
it: the labels are data, and executing a downloaded file to read a list of numbers
would be a poor trade.

The video files themselves are distributed separately from the label file. When
they are present, :func:`build_feature_table` measures them; when they are not,
that is reported as a missing prerequisite rather than papered over.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pandas as pd

from ..errors import DatasetError
from ..logging_conf import get_logger

log = get_logger(__name__)

# The columns everything downstream relies on.
LABEL_COLUMN = "mos"
GROUP_COLUMN = "content"
# Objective features, in the order they are reported.
DEFAULT_FEATURES = ("vmaf", "psnr_y", "float_ssim", "float_ms_ssim")


def parse_vmaf_dataset_file(path: Path | str) -> dict[str, list[dict[str, Any]]]:
    """Extract ``ref_videos`` / ``dis_videos`` from a vmaf-style dataset file.

    Parsed with :mod:`ast`, so a malformed or hostile file produces a
    :class:`DatasetError` instead of running.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"dataset descriptor not found: {path}")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise DatasetError(f"{path} is not a parseable Python file: {exc}") from exc

    wanted = {"ref_videos", "dis_videos"}
    # Simple top-level assignments first: the real NFLX-P file defines ref_dir and
    # dis_dir, then builds every path as `ref_dir + '/clip.yuv'`.
    namespace: dict[str, Any] = {}
    found: dict[str, list[dict[str, Any]]] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if not names:
            continue
        try:
            value = _safe_eval(node.value, namespace)
        except DatasetError as exc:
            if any(name in wanted for name in names):
                raise DatasetError(f"{path}: cannot read {names[0]} safely: {exc}") from exc
            continue  # an assignment we do not need; skip it rather than fail
        for name in names:
            namespace[name] = value
            if name in wanted:
                if not isinstance(value, list):
                    raise DatasetError(f"{path}: {name} is not a list")
                found[name] = [dict(item) for item in value]

    missing = wanted - set(found)
    if missing:
        raise DatasetError(f"{path} does not define {sorted(missing)}")
    return found


def _safe_eval(node: ast.expr, namespace: dict[str, Any]) -> Any:
    """Evaluate the small expression language a dataset descriptor actually uses.

    ``ast.literal_eval`` is not quite enough: the real file writes
    ``'path': ref_dir + '/BigBuckBunny_25fps.yuv'``, which is a name lookup plus a
    concatenation, not a literal. So this handles literals, lists/tuples/dicts,
    references to names already assigned in the same file, and ``+``/``-`` on
    strings and numbers.

    Everything else - calls, attribute access, imports, comprehensions - raises.
    That keeps the guarantee that matters: reading a downloaded label file never
    executes it.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_safe_eval(item, namespace) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(item, namespace) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _safe_eval(key, namespace): _safe_eval(value, namespace)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }
    if isinstance(node, ast.Name):
        if node.id not in namespace:
            raise DatasetError(f"reference to undefined name {node.id!r}")
        return namespace[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand, namespace)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub):
        left, right = _safe_eval(node.left, namespace), _safe_eval(node.right, namespace)
        if isinstance(node.op, ast.Add):
            return left + right
        return left - right
    raise DatasetError(f"unsupported expression {type(node).__name__} in dataset descriptor")


def load_nflx_labels(path: Path | str, *, include_hidden_reference: bool = False) -> pd.DataFrame:
    """Label table for NFLX-P: one row per distorted clip.

    Three things worth knowing about this dataset's conventions:

    **Score direction.** NFLX-P's ``dmos`` field runs from bad to excellent (its
    reference clips sit at 100), so higher already means better and no flip is
    needed. Other vmaf-format datasets store a *degradation* where higher is worse.
    Getting this backwards would invert every correlation in the report while still
    producing plausible-looking numbers, which is why it is stated here rather than
    assumed.

    **Hidden references.** The distorted list includes each pristine reference
    scored against itself (DMOS 100). Those rows are dropped by default: PSNR of a
    clip against itself is infinite (libvmaf reports its 60 dB ceiling), so they are
    degenerate points that flatter any correlation. Pass
    ``include_hidden_reference=True`` to keep them.

    **Encoding parameters in the filenames.** NFLX-P encodes them into the file
    name as ``{content}_{expert score}_{height}_{bitrate kbps}.yuv``, so the
    resolution, the target bitrate and a separate expert opinion score are all
    recoverable from the label file alone, without the videos. They are parsed into
    columns here because they cost nothing and are useful for sanity checks.
    """
    payload = parse_vmaf_dataset_file(path)
    references = {int(ref["content_id"]): ref for ref in payload["ref_videos"]}

    rows: list[dict[str, Any]] = []
    for entry in payload["dis_videos"]:
        try:
            content_id = int(entry["content_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DatasetError(f"distorted entry without a usable content_id: {entry}") from exc
        reference = references.get(content_id)
        if reference is None:
            raise DatasetError(f"distorted entry references unknown content_id {content_id}")

        score = _first_present(entry, ("dmos", "groundtruth", "mos"))
        if score is None:
            raise DatasetError(f"distorted entry {entry.get('asset_id')} has no subjective score")
        reference_path = str(reference.get("path", ""))
        distorted_path = str(entry.get("path", ""))
        rows.append(
            {
                "content_id": content_id,
                GROUP_COLUMN: str(reference.get("content_name", f"content_{content_id}")),
                "asset_id": entry.get("asset_id"),
                "dmos": float(score),
                "reference_path": reference_path,
                "distorted_path": distorted_path,
                "is_hidden_reference": distorted_path == reference_path,
                **_parse_nflx_filename(distorted_path),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise DatasetError(f"{path} contained no distorted videos")
    table[LABEL_COLUMN] = table["dmos"]

    hidden = int(table["is_hidden_reference"].sum())
    if hidden and not include_hidden_reference:
        table = table[~table["is_hidden_reference"]].reset_index(drop=True)
        log.info("dropped %d hidden reference row(s) (a clip scored against itself)", hidden)

    log.info(
        "loaded %d labelled clips over %d contents from %s",
        len(table),
        table[GROUP_COLUMN].nunique(),
        Path(path).name,
    )
    return table


def _parse_nflx_filename(path: str) -> dict[str, Any]:
    """Pull ``{content}_{expert score}_{height}_{bitrate}`` out of a filename.

    Returns empty values rather than raising when the pattern does not match: the
    reference clips are named ``{content}_{fps}fps.yuv`` and other vmaf-format
    datasets use their own conventions.
    """
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) < 4:
        return {"expert_score": None, "height": None, "target_bitrate_kbps": None}
    try:
        expert, height, bitrate = (int(parts[-3]), int(parts[-2]), int(parts[-1]))
    except ValueError:
        return {"expert_score": None, "height": None, "target_bitrate_kbps": None}
    return {"expert_score": expert, "height": height, "target_bitrate_kbps": bitrate}


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            try:
                return float(mapping[key])
            except (TypeError, ValueError):
                continue
    return None


def build_feature_table(
    labels: pd.DataFrame,
    measure: Callable[[Path, Path], Mapping[str, float]],
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    """Join measured objective metrics onto the label table.

    ``measure`` is injected rather than imported so this function can be tested
    without ffmpeg, and so a cached measurement can be replayed instead of
    recomputed. Missing media is collected and reported once at the end: finding
    out on clip 3 of 70 that the videos are not downloaded is more useful than
    finding out on clip 1 and dying.
    """
    root = Path(root) if root is not None else None
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for raw_record in labels.to_dict(orient="records"):
        record = {str(key): value for key, value in raw_record.items()}
        reference = _resolve(str(record["reference_path"]), root)
        distorted = _resolve(str(record["distorted_path"]), root)
        if not reference.exists() or not distorted.exists():
            missing.append(distorted.name or str(distorted))
            continue
        measured = measure(distorted, reference)
        rows.append({**record, **measured})

    if missing:
        log.warning(
            "%d of %d clips had no media on disk (first few: %s)",
            len(missing),
            len(labels),
            ", ".join(missing[:3]),
        )
    if not rows:
        raise DatasetError(
            "no clip in the dataset had both its reference and distorted file on disk. "
            "The NFLX-P label file ships without the videos: see DATA_CARD.md for how "
            "to obtain them, and point --media-root at the directory that holds them."
        )
    return pd.DataFrame(rows)


def _resolve(path: str, root: Path | None) -> Path:
    """Find a dataset clip on *this* machine.

    Dataset descriptors carry absolute paths from whoever built them
    (``/Users/someone/vmaf/resource/yuv/...``), which of course do not exist here.
    So: use the recorded path if it happens to resolve, otherwise look for the same
    filename under our own media root. Trusting the recorded path alone would make
    every local checkout look like it had no media.
    """
    candidate = Path(path)
    if candidate.exists() or root is None:
        return candidate
    return root / candidate.name


def build_livenflx_features(
    release_dir: Path | str,
    *,
    scope: str = "compression_only",
    metrics: Sequence[str] | None = None,
    poolings: Sequence[str] = ("mean", "hmean"),
) -> pd.DataFrame:
    """Feature table for LIVE-Netflix: read the release, then pool to one row per clip.

    The counterpart to :func:`build_feature_table`, which measures NFLX-P's videos
    with ffmpeg. This one needs no media and no ffmpeg, because the release ships
    the per-frame quality vectors alongside the subjective scores - which is the
    whole reason it can close this out while NFLX-P's access grant is pending.

    The label it carries is ``subj_score``, not ``mos``: LIVE-Netflix Z-scores its
    ratings per session and per subject, so the values are centred on zero rather
    than on a 1-5 opinion scale. See :mod:`~pixeljudge.model.livenflx`.
    """
    from .features import pool_to_features
    from .livenflx import ConditionScope, load_livenflx

    long_table = load_livenflx(release_dir, scope=cast("ConditionScope", scope), metrics=metrics)
    return pool_to_features(long_table, poolings=poolings)


def load_feature_table(
    path: Path | str, *, features: Sequence[str] = DEFAULT_FEATURES
) -> pd.DataFrame:
    """Read a cached feature table and check it has what the model needs."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"feature table not found: {path}")
    table = pd.read_csv(path)
    require_columns(table, [*features, LABEL_COLUMN, GROUP_COLUMN], source=str(path))
    return table


def require_columns(table: pd.DataFrame, columns: Sequence[str], *, source: str = "table") -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise DatasetError(
            f"{source} is missing required column(s) {missing}; present: {sorted(table.columns)}"
        )


def content_groups(table: pd.DataFrame, group_column: str = GROUP_COLUMN) -> pd.Series:
    """The grouping vector used for content-based cross-validation.

    This is the anti-leakage centrepiece. Each source clip appears many times in a
    subjective dataset (once per distortion level), so a random row split puts the
    *same content* in train and test. A model can then score well by recognising
    the content instead of judging its quality, and the reported correlation is
    inflated. Grouping by source clip keeps every distortion of one clip on one
    side of the split.
    """
    require_columns(table, [group_column])
    return table[group_column].astype(str)


def describe(table: pd.DataFrame, *, label_column: str = LABEL_COLUMN) -> dict[str, Any]:
    """Small summary used in logs and the model card."""
    groups = content_groups(table)
    return {
        "rows": int(len(table)),
        "contents": int(groups.nunique()),
        "clips_per_content": round(float(len(table) / max(groups.nunique(), 1)), 2),
        "label_min": round(float(table[label_column].min()), 3),
        "label_max": round(float(table[label_column].max()), 3),
        "label_mean": round(float(table[label_column].mean()), 3),
    }
