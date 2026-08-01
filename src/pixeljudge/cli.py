"""Command-line interface.

Subcommands mirror the pipeline stages, and each one is a thin shell around
library code so everything stays testable without a terminal:

    doctor    check the environment (ffmpeg, libvmaf, encoders)
    encode    run a master clip through a ladder
    measure   compute PSNR/SSIM/MS-SSIM/VMAF for encoded rungs
    scan      artifact detectors over encoded rungs
    ladder    build a per-title convex-hull ladder from probe encodes
    train     fit and evaluate the MOS regressor
    report    regenerate every plot and table from cached metrics
    mos       the LIVE-Netflix subjective regression (load / train / table)

Intermediate results are always written to disk (manifests, metrics CSVs), so any
stage can be re-run without repeating the expensive one before it. Encoding a
four-codec sweep takes hours; measuring it again should not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.table import Table

from .errors import PixelJudgeError
from .logging_conf import console, get_logger, setup_logging

if TYPE_CHECKING:  # heavy imports stay out of CLI start-up
    import pandas as pd

    from .config import LadderConfig, PipelineConfig

log = get_logger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Full-reference video quality analysis: PSNR/SSIM/VMAF, ladders, BD-Rate, MOS regression.",
)

DEFAULT_CONFIG = Path("configs/pipeline.yaml")
DEFAULT_MOS_CONFIG = Path("configs/livenflx.yaml")
ALL_METRICS_CSV = "all_metrics.csv"
ARTIFACTS_CSV = "artifacts.csv"
TRAINING_REPORT_JSON = "training_report.json"
# Cached pooled features, so `mos train` and `mos table` can re-run without
# re-reading 112 MATLAB files.
LIVENFLX_FEATURES_CSV = "livenflx_features.csv"
LIVENFLX_REPORT_JSON = "livenflx_report.json"
LIVENFLX_TABLE_CSV = "mos_correlation.csv"

ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="Pipeline config YAML.")]
MosConfigOption = Annotated[
    Path, typer.Option("--mos-config", "-m", help="LIVE-Netflix regression config YAML.")
]


@app.callback()
def _root(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    setup_logging("DEBUG" if verbose else "INFO")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@app.command()
def doctor() -> None:
    """Check that ffmpeg, libvmaf and the encoders we need are present."""
    from . import __version__
    from .io.ffmpeg import (
        INSTALL_HINT,
        environment_report,
        ffmpeg_available,
        has_libvmaf,
        libvmaf_models,
    )

    if not ffmpeg_available():
        console.print("[bold red]ffmpeg/ffprobe not found[/bold red]")
        console.print(INSTALL_HINT)
        raise typer.Exit(code=1)

    table = Table(title=f"PixelJudge {__version__} environment", header_style="bold")
    table.add_column("check")
    table.add_column("result")
    table.add_row("python", sys.version.split()[0])
    for key, value in environment_report().items():
        style = "red" if value in {"no", "missing"} or value.startswith("MISSING") else ""
        table.add_row(key, f"[{style}]{value}[/{style}]" if style else value)
    table.add_row("vmaf models", ", ".join(libvmaf_models()) or "-")
    console.print(table)

    if not has_libvmaf():
        console.print(
            "[bold red]libvmaf is missing:[/bold red] PSNR/SSIM still work, VMAF does not."
        )
        console.print(INSTALL_HINT)
        raise typer.Exit(code=1)
    console.print("[bold green]environment looks good.[/bold green]")


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


@app.command()
def encode(
    config: ConfigOption = DEFAULT_CONFIG,
    source: Annotated[
        list[str] | None, typer.Option("--source", "-s", help="Master clip (repeatable).")
    ] = None,
    ladder: Annotated[
        list[str] | None, typer.Option("--ladder", "-l", help="Ladder name (repeatable).")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Re-encode existing files.")
    ] = False,
) -> None:
    """Encode each master through each ladder, writing a manifest per pair."""
    from .config import load_pipeline_config
    from .encode.encoder import encode_ladder, manifest_path, write_manifest

    cfg = load_pipeline_config(config)
    cfg.paths.ensure()
    sources = _resolve_sources(cfg, source)
    ladders = _resolve_ladders(cfg, ladder)

    for source_path in sources:
        for ladder_cfg in ladders:
            rungs = encode_ladder(
                source_path, ladder_cfg, cfg.encode, cfg.paths.encoded, overwrite=overwrite
            )
            write_manifest(rungs, manifest_path(source_path, ladder_cfg, cfg.paths.encoded))
    console.print(
        f"[green]encoded[/green] {len(sources)} source(s) x {len(ladders)} ladder(s) "
        f"into {cfg.paths.encoded}"
    )


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------


@app.command()
def measure(
    config: ConfigOption = DEFAULT_CONFIG,
    manifest: Annotated[
        list[Path] | None,
        typer.Option("--manifest", "-m", help="Encode manifest (default: all in data/encoded)."),
    ] = None,
    per_frame: Annotated[
        bool, typer.Option("--per-frame/--no-per-frame", help="Save per-frame CSVs.")
    ] = True,
) -> None:
    """Measure every encoded rung against its master and cache the results."""

    from .config import load_pipeline_config
    from .encode.encoder import read_manifest
    from .metrics.vqm import measure_many

    cfg = load_pipeline_config(config)
    cfg.paths.ensure()
    manifests = _resolve_manifests(cfg, manifest)

    frames: list[pd.DataFrame] = []
    for manifest_file in manifests:
        rungs = read_manifest(manifest_file)
        jobs = [
            (
                Path(rung.path),
                Path(rung.source),
                {
                    "source": Path(rung.source).stem,
                    "ladder": rung.ladder,
                    "codec": rung.codec,
                    "rung": rung.rung,
                    "width": rung.width,
                    "height": rung.height,
                    "crf": rung.crf,
                    "target_bitrate_kbps": rung.target_bitrate_kbps,
                    "actual_bitrate_kbps": rung.actual_bitrate_kbps,
                    "encode_seconds": rung.encode_seconds,
                },
            )
            for rung in rungs
        ]
        table = measure_many(
            jobs,
            cfg.vmaf,
            scale_flags=cfg.encode.scale_flags,
            per_frame_dir=cfg.paths.metrics / "frames" if per_frame else None,
        )
        out_path = cfg.paths.metrics / f"{manifest_file.stem}.metrics.csv"
        table.to_csv(out_path, index=False)
        log.info("wrote %s (%d rows)", out_path, len(table))
        frames.append(table)

    # Rebuild the combined table from every per-manifest CSV on disk, not just the
    # ones measured in this run. Re-measuring one ladder (after widening a CRF
    # sweep, say) should not silently drop the other twenty-seven from the report.
    combined = _combine_metrics(cfg.paths.metrics)
    combined_path = cfg.paths.metrics / ALL_METRICS_CSV
    combined.to_csv(combined_path, index=False)
    console.print(
        f"[green]measured[/green] {sum(len(f) for f in frames)} rungs this run; "
        f"{len(combined)} total -> {combined_path}"
    )


def _combine_metrics(metrics_dir: Path) -> pd.DataFrame:
    import pandas as pd

    parts = [pd.read_csv(path) for path in sorted(metrics_dir.glob("*.metrics.csv"))]
    if not parts:
        raise PixelJudgeError(f"no per-manifest metrics found in {metrics_dir}")
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


@app.command()
def scan(
    config: ConfigOption = DEFAULT_CONFIG,
    manifest: Annotated[list[Path] | None, typer.Option("--manifest", "-m")] = None,
    n_sample: Annotated[int, typer.Option("--frames", help="Frames sampled per clip.")] = 0,
) -> None:
    """Run the artifact detectors over encoded rungs and save evidence frames."""
    from .artifacts.scan import scan_many
    from .config import load_pipeline_config
    from .encode.encoder import read_manifest

    cfg = load_pipeline_config(config)
    cfg.paths.ensure()
    manifests = _resolve_manifests(cfg, manifest)
    rungs = [rung for file in manifests for rung in read_manifest(file)]
    paths = [Path(rung.path) for rung in rungs]
    # Each rung's master, so the scan can report banding the encode *introduced*
    # rather than banding the content already had.
    references = {rung.path: rung.source for rung in rungs}

    table = scan_many(
        paths,
        references=references,
        n_sample=n_sample or cfg.artifact_sample_frames,
        evidence_dir=cfg.paths.reports / "evidence",
    )
    out_path = cfg.paths.metrics / ARTIFACTS_CSV
    table.to_csv(out_path, index=False)
    console.print(f"[green]scanned[/green] {len(table)} clips -> {out_path}")


# ---------------------------------------------------------------------------
# ladder (per-title)
# ---------------------------------------------------------------------------


@app.command()
def ladder(
    config: ConfigOption = DEFAULT_CONFIG,
    source: Annotated[str, typer.Option("--source", "-s", help="Master clip.")] = "",
    codec: Annotated[str, typer.Option("--codec", help="Codec for the probe grid.")] = "h264",
    n_rungs: Annotated[int, typer.Option("--rungs", help="Rungs in the final ladder.")] = 5,
    target_vmaf: Annotated[float, typer.Option("--target-vmaf")] = 94.0,
    out: Annotated[
        Path | None, typer.Option("--out", help="Where to write the ladder YAML.")
    ] = None,
) -> None:
    """Build a per-title convex-hull ladder: probe encodes, measure, pick the hull."""
    import pandas as pd
    import yaml

    from .config import load_pipeline_config
    from .encode.encoder import encode_ladder
    from .io.ffmpeg import probe
    from .ladder.builder import (
        RdPoint,
        ladder_to_yaml_dict,
        per_title_ladder,
        probe_grid,
        upper_convex_hull,
    )
    from .metrics.vqm import measure_many
    from .viz.plots import plot_convex_hull

    cfg = load_pipeline_config(config)
    cfg.paths.ensure()
    if not source:
        raise PixelJudgeError("--source is required: a per-title ladder is built for one title")
    source_path = cfg.resolve_source(source)
    info = probe(source_path)

    grid = probe_grid(codec, source_height=info.height)  # type: ignore[arg-type]
    log.info("probe grid: %d encodes for %s", len(grid.rungs), source_path.name)
    rungs = encode_ladder(source_path, grid, cfg.encode, cfg.paths.encoded / "probe")

    jobs = [
        (
            Path(rung.path),
            source_path,
            {
                "height": rung.height,
                "crf": rung.crf,
                "actual_bitrate_kbps": rung.actual_bitrate_kbps,
            },
        )
        for rung in rungs
    ]
    measured = measure_many(jobs, cfg.vmaf, scale_flags=cfg.encode.scale_flags)
    measured.to_csv(cfg.paths.metrics / f"{source_path.stem}.probe_grid.csv", index=False)

    points = [
        RdPoint(
            height=int(row["height"]),
            bitrate_kbps=float(row["actual_bitrate_kbps"]),
            vmaf=float(row["vmaf"]),
            crf=int(row["crf"]) if row.get("crf") is not None else None,
        )
        for row in measured.to_dict(orient="records")
    ]
    built = per_title_ladder(
        points, codec=codec, name=f"per_title_{source_path.stem}", n_rungs=n_rungs, target_vmaf=target_vmaf  # type: ignore[arg-type]
    )

    hull = upper_convex_hull(points)
    plot_convex_hull(
        measured,
        cfg.paths.reports / f"convex_hull_{source_path.stem}.png",
        hull=pd.DataFrame(
            [
                {"actual_bitrate_kbps": p.bitrate_kbps, "vmaf": p.vmaf, "height": p.height}
                for p in hull
            ]
        ),
        title=f"Per-title ladder selection: {source_path.name}",
    )

    destination = out or cfg.paths.ladder_dir / f"{built.name}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(ladder_to_yaml_dict(built), sort_keys=False), encoding="utf-8"
    )
    console.print(f"[green]per-title ladder[/green] ({len(built.rungs)} rungs) -> {destination}")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@app.command()
def train(
    features: Annotated[Path, typer.Option("--features", "-f", help="Feature table CSV.")],
    config: ConfigOption = DEFAULT_CONFIG,
    n_splits: Annotated[int, typer.Option("--folds", help="Outer CV folds.")] = 5,
    feature_columns: Annotated[
        list[str] | None, typer.Option("--feature", help="Feature column (repeatable).")
    ] = None,
) -> None:
    """Fit the MOS regressor with content-grouped CV and compare against baselines."""
    from .config import load_pipeline_config
    from .model.dataset import DEFAULT_FEATURES, load_feature_table
    from .model.train import fit_final_model, save_model, save_report, train_and_evaluate

    cfg = load_pipeline_config(config)
    cfg.paths.ensure()
    columns = feature_columns or list(DEFAULT_FEATURES)
    table = load_feature_table(features, features=columns)

    report = train_and_evaluate(
        table, features=columns, n_splits=n_splits, random_seed=cfg.random_seed
    )
    save_report(report, cfg.paths.models / TRAINING_REPORT_JSON)
    report.oof.to_csv(cfg.paths.models / "oof_predictions.csv", index=False)
    model = fit_final_model(
        table, columns, report.label, report.best_model, report.best_params, cfg.random_seed
    )
    save_model(model, cfg.paths.models / f"{report.best_model}.joblib")

    console.print(_results_table(report.table.to_dict(orient="records")))
    verdict = report.beats_vmaf_baseline()
    if verdict is True:
        console.print("[green]the fused model matches or beats VMAF alone on SROCC.[/green]")
    elif verdict is False:
        console.print(
            "[yellow]VMAF alone still ranks better than the fused model. "
            "That is a legitimate result and belongs in the README as-is.[/yellow]"
        )


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@app.command()
def report(
    config: ConfigOption = DEFAULT_CONFIG,
    anchor: Annotated[str, typer.Option("--anchor", help="Anchor codec for BD-Rate.")] = "h264",
) -> None:
    """Regenerate every plot and table from cached metrics. No encoding, no measuring."""
    import pandas as pd

    from .artifacts.scan import find_metric_blind_spots
    from .config import load_pipeline_config
    from .viz.plots import (
        correlation_table,
        plot_banding_comparison,
        plot_banding_gallery,
        plot_metric_vs_banding,
        plot_predicted_vs_mos,
        plot_rd_curves,
        save_table,
    )

    cfg = load_pipeline_config(config)
    cfg.paths.ensure()
    reports = cfg.paths.reports
    written: list[Path] = []

    metrics_path = cfg.paths.metrics / ALL_METRICS_CSV
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    if metrics.empty:
        log.warning("no metrics at %s; run 'pixeljudge measure' first", metrics_path)
    else:
        # One figure per source. Curves from different content must never share an
        # axis: an easy clip at 1 Mbps and a hard clip at 1 Mbps are not two points
        # on one curve, they are two different questions.
        for source, group in metrics.groupby("source"):
            # Two separate questions, so two separate figures: "which delivery
            # recipe is better" (bitrate-targeted ladders, resolution changing with
            # each rung) and "which codec is more efficient" (constant-quality
            # sweeps at a fixed resolution).
            ladders = group[group["crf"].isna()] if "crf" in group else group
            sweeps = group[group["crf"].notna()] if "crf" in group else group.iloc[0:0]
            if not ladders.empty:
                written.append(
                    plot_rd_curves(
                        ladders,
                        reports / f"rd_curves_ladders_{source}.png",
                        quality="vmaf",
                        title=f"Delivery ladders: quality vs bitrate ({source})",
                    )
                )
            if not sweeps.empty:
                written.append(
                    plot_rd_curves(
                        sweeps,
                        reports / f"rd_curves_codecs_{source}.png",
                        quality="vmaf",
                        group_column="codec",
                        title=f"Codec efficiency at 720p, constant quality ({source})",
                    )
                )
                written.append(
                    plot_rd_curves(
                        sweeps,
                        reports / f"rd_curves_codecs_psnr_{source}.png",
                        quality="psnr_y",
                        group_column="codec",
                        title=f"Codec efficiency measured with PSNR-Y ({source})",
                    )
                )
        written.append(save_table(_ladder_summary(metrics), reports / "ladder_summary.csv"))
        bd = _bd_rate_table(metrics, anchor)
        if bd is not None:
            written.append(save_table(bd, reports / "bd_rate.csv"))

    artifacts_path = cfg.paths.metrics / ARTIFACTS_CSV
    artifacts = pd.read_csv(artifacts_path) if artifacts_path.exists() else pd.DataFrame()
    if not metrics.empty and not artifacts.empty:
        merged = metrics.merge(
            artifacts.assign(distorted=[Path(p).name for p in artifacts["path"]]),
            on="distorted",
            how="inner",
        )
        if not merged.empty:
            blind = find_metric_blind_spots(metrics, artifacts)
            # Draw the figure with the table's own threshold, so the shaded quadrant
            # and the listed clips are guaranteed to be the same set.
            threshold = (
                float(blind["banding_threshold"].iloc[0])
                if not blind.empty and "banding_threshold" in blind
                else None
            )
            written.append(
                plot_metric_vs_banding(
                    merged,
                    reports / "metric_vs_banding.png",
                    banding_column=_banding_column(merged),
                    banding_threshold=threshold,
                    selected=len(blind),
                )
            )
            written.append(save_table(_blind_spot_columns(blind), reports / "blind_spots.csv"))
            gallery = _gallery_items(blind, reports / "evidence")
            if gallery:
                written.append(plot_banding_gallery(gallery, reports / "banding_gallery.png"))
            # The headline figure: the worst offender's frame beside its master.
            pair = _comparison_pair(blind, reports / "evidence")
            if pair is not None:
                reference_png, distorted_png, caption = pair
                written.append(
                    plot_banding_comparison(
                        reference_png,
                        distorted_png,
                        reports / "banding_master_vs_encode.png",
                        caption=caption,
                    )
                )

    training_path = cfg.paths.models / TRAINING_REPORT_JSON
    if training_path.exists():
        payload = json.loads(training_path.read_text(encoding="utf-8"))
        written.append(
            save_table(correlation_table(payload["results"]), reports / "correlations.csv")
        )
        oof_path = cfg.paths.models / "oof_predictions.csv"
        if oof_path.exists():
            oof = pd.read_csv(oof_path)
            column = f"pred_{payload['best_model']}"
            if column in oof.columns:
                written.append(
                    plot_predicted_vs_mos(
                        oof,
                        reports / "predicted_vs_mos.png",
                        prediction_column=column,
                        label_column=payload["label"],
                    )
                )
    else:
        log.warning("no training report at %s; run 'pixeljudge train' first", training_path)

    if not written:
        raise PixelJudgeError(
            "nothing to report yet. Run encode -> measure -> scan (and train) first."
        )
    console.print(f"[green]regenerated[/green] {len(written)} artifact(s) in {reports}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_sources(cfg: PipelineConfig, source: list[str] | None) -> list[Path]:
    names = source or cfg.sources
    if not names:
        raise PixelJudgeError(
            "no sources given: pass --source, or list them under 'sources' in the config"
        )
    paths = [cfg.resolve_source(name) for name in names]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise PixelJudgeError(f"source clip(s) not found: {', '.join(missing)}")
    return paths


def _resolve_ladders(cfg: PipelineConfig, ladder: list[str] | None) -> list[LadderConfig]:
    from .config import load_ladder

    if ladder:
        return [load_ladder(cfg.ladder_path(name)) for name in ladder]
    if not cfg.ladders:
        raise PixelJudgeError("no ladders given: pass --ladder, or list them in the config")
    return cfg.load_ladders()


def _resolve_manifests(cfg: PipelineConfig, manifest: list[Path] | None) -> list[Path]:
    if manifest:
        missing = [str(path) for path in manifest if not path.exists()]
        if missing:
            raise PixelJudgeError(f"manifest(s) not found: {', '.join(missing)}")
        return list(manifest)
    found = sorted(cfg.paths.encoded.glob("*.manifest.json"))
    if not found:
        raise PixelJudgeError(f"no manifests in {cfg.paths.encoded}; run 'pixeljudge encode' first")
    return found


def _ladder_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-source, per-ladder overview: what each recipe spends and what it buys."""
    columns = {
        "actual_bitrate_kbps": ["min", "max", "mean"],
        "vmaf": ["min", "max", "mean"],
    }
    available = {key: aggs for key, aggs in columns.items() if key in metrics.columns}
    summary = metrics.groupby(["source", "ladder"]).agg(available).round(2)
    summary.columns = ["_".join(parts) for parts in summary.columns]
    flattened: pd.DataFrame = summary.reset_index()
    return flattened


def _bd_rate_table(metrics: pd.DataFrame, anchor: str) -> pd.DataFrame | None:
    """BD-Rate across codecs, computed per source clip.

    Two restrictions, both of which change the answer:

    * **Constant-quality rows only.** A BD-Rate needs each encoder asked for the
      same *quality* and free to spend what it needs. The platform ladders are
      bitrate-targeted and change resolution between rungs, so their curves answer
      a different question and are excluded here.
    * **One curve per source.** Rate-distortion is a property of a codec *on a
      given clip*. Pooling clips into one curve would compare "AV1 on the easy
      scene" against "H.264 on the hard one" and call the difference codec
      efficiency.
    """
    import pandas as pd  # a runtime import: the module-level one is TYPE_CHECKING only

    from .metrics.bdrate import MIN_POINTS, RdCurve, bd_table

    if "crf" not in metrics.columns:
        log.warning("no crf column: BD-Rate needs constant-quality sweeps")
        return None
    sweeps = metrics[metrics["crf"].notna()]
    if sweeps.empty:
        log.warning("no constant-quality rows found. Encode a crf_sweep_* ladder to get BD-Rate.")
        return None

    tables: list[pd.DataFrame] = []
    for source, per_source in sweeps.groupby("source"):
        curves: dict[str, RdCurve] = {}
        for codec, group in per_source.groupby("codec"):
            if len(group) < MIN_POINTS:
                log.warning("skipping BD-Rate for %s/%s: only %d points", source, codec, len(group))
                continue
            curves[str(codec)] = RdCurve.from_frame(group, f"{source}/{codec}")
        if anchor not in curves or len(curves) < 2:
            log.warning(
                "%s: BD-Rate needs anchor %r plus another codec; have %s",
                source,
                anchor,
                sorted(curves),
            )
            continue
        table = bd_table(curves, anchor=anchor)
        table.insert(0, "source", source)
        tables.append(table)

    if not tables:
        return None
    combined = pd.concat(tables, ignore_index=True)
    # A mean across sources is the headline number, but only alongside the
    # per-source rows: BD-Rate varies a lot with content, and hiding that spread
    # behind one average is how codec comparisons get oversold.
    averages = (
        combined.groupby("codec")[["bd_rate_pct", "bd_quality"]]
        .mean()
        .round(2)
        .reset_index()
        .assign(source="MEAN across sources", anchor=anchor, metric="vmaf")
    )
    return pd.concat([combined, averages], ignore_index=True)


def _banding_column(merged: pd.DataFrame) -> str:
    """Prefer the reference-relative delta; fall back only if the scan had no reference."""
    if "banding_delta_max" in merged.columns and merged["banding_delta_max"].notna().any():
        return "banding_delta_max"
    log.warning("no reference-relative banding available; plotting absolute scores")
    return "banding_max"


def _blind_spot_columns(blind: pd.DataFrame) -> pd.DataFrame:
    keep = [
        c
        for c in (
            "distorted",
            "ladder",
            "codec",
            "rung",
            "actual_bitrate_kbps",
            "vmaf",
            "psnr_y",
            "float_ssim",
            "banding_max",
            "reference_banding_mean",
            "banding_delta_mean",
            "banding_delta_max",
            "banding_delta_worst_frame",
            "banding_threshold",
        )
        if c in blind.columns
    ]
    return blind[keep] if keep else blind


def _gallery_items(blind: pd.DataFrame, evidence_dir: Path) -> list[tuple[Path, str]]:
    """Match blind-spot rows to the evidence frames saved during the scan."""
    items: list[tuple[Path, str]] = []
    for row in blind.to_dict(orient="records"):
        stem = Path(str(row.get("distorted", ""))).stem
        frame = row.get("banding_delta_worst_frame") or row.get("banding_worst_frame")
        candidate = evidence_dir / f"{stem}__banding_frame{frame}.png"
        if not candidate.exists():
            continue
        caption = (
            f"{stem}\nVMAF {row.get('vmaf', float('nan')):.1f} | "
            f"PSNR {row.get('psnr_y', float('nan')):.1f} dB | "
            f"banding added {row.get('banding_delta_max', float('nan')):+.1f}"
        )
        items.append((candidate, caption))
    return items


def _comparison_pair(blind: pd.DataFrame, evidence_dir: Path) -> tuple[Path, Path, str] | None:
    """Pick the clip that best demonstrates the claim, and its master's matching frame.

    Every candidate has already passed the "metrics say this is fine" filter (VMAF >= 80,
    PSNR >= 38 dB), so ranking within that set by *added* banding gives the strongest
    honest example: good scores and visible damage at once. Ranking by metric score
    instead picks the least damaged clip and produces a figure where the two panels look
    identical - which is what the first version of this did.
    """
    if blind.empty:
        return None
    # blind is already filtered to "the metrics call this good", so the most convincing
    # member of that set is the one where the encoder added the most contouring - which
    # is the order find_metric_blind_spots already returns.
    for row in blind.to_dict(orient="records"):
        stem = Path(str(row.get("distorted", ""))).stem
        frame = row.get("banding_delta_worst_frame")
        distorted_png = evidence_dir / f"{stem}__banding_frame{frame}.png"
        reference_png = evidence_dir / f"{stem}__reference_frame{frame}.png"
        if distorted_png.exists() and reference_png.exists():
            caption = (
                f"{stem}\n"
                f"VMAF {row.get('vmaf', float('nan')):.1f}  |  "
                f"PSNR {row.get('psnr_y', float('nan')):.1f} dB  |  "
                f"SSIM {row.get('float_ssim', float('nan')):.4f}  |  "
                f"banding added {row.get('banding_delta_max', float('nan')):+.1f}"
            )
            return reference_png, distorted_png, caption
    return None


def _results_table(rows: list[dict[Any, Any]]) -> Table:
    table = Table(title="Correlation with subjective MOS", header_style="bold")
    for column in ("name", "n", "plcc", "srocc", "krocc", "rmse"):
        table.add_column(column)
    for row in rows:
        table.add_row(
            str(row.get("name", "")),
            str(row.get("n", "")),
            f"{row.get('plcc', float('nan')):.4f}",
            f"{row.get('srocc', float('nan')):.4f}",
            f"{row.get('krocc', float('nan')):.4f}",
            f"{row.get('rmse', float('nan')):.4f}",
        )
    return table


# ---------------------------------------------------------------------------
# mos: the LIVE-Netflix subjective regression
# ---------------------------------------------------------------------------

mos_app = typer.Typer(
    no_args_is_help=True,
    help="Map objective metrics onto human opinion scores (LIVE-Netflix Video QoE DB).",
)
app.add_typer(mos_app, name="mos")


def _mos_paths(config: Path, mos_config: Path) -> tuple[PipelineConfig, Any]:
    from .config import load_livenflx_config, load_pipeline_config

    cfg = load_pipeline_config(config)
    cfg.paths.ensure()
    return cfg, load_livenflx_config(mos_config)


def _scope_note(scope: str) -> str:
    """The caption that has to travel with every number from this dataset."""
    if scope == "compression_only":
        return (
            "scope=compression_only: the 4 continuous-playback conditions. "
            "Bitrate adaptation retained, rebuffering excluded."
        )
    return (
        "scope=all: includes rebuffering conditions. These are QoE correlations, "
        "NOT compression-quality correlations - compression metrics cannot observe a stall."
    )


@mos_app.command("load")
def mos_load(
    config: ConfigOption = DEFAULT_CONFIG,
    mos_config: MosConfigOption = DEFAULT_MOS_CONFIG,
) -> None:
    """Read the release, pool per-frame vectors to clip features, cache as CSV."""
    from .model.dataset import build_livenflx_features

    cfg, mos = _mos_paths(config, mos_config)
    table = build_livenflx_features(
        mos.release_dir, scope=mos.scope, metrics=mos.metrics, poolings=mos.poolings
    )
    path = cfg.paths.datasets / LIVENFLX_FEATURES_CSV
    table.to_csv(path, index=False)
    console.print(f"wrote {path}: {len(table)} clips, {table['content'].nunique()} contents")
    console.print(f"[dim]{_scope_note(mos.scope)}[/dim]")


@mos_app.command("train")
def mos_train(
    config: ConfigOption = DEFAULT_CONFIG,
    mos_config: MosConfigOption = DEFAULT_MOS_CONFIG,
    features_csv: Annotated[
        Path | None, typer.Option("--features", "-f", help="Cached feature CSV (default: rebuild).")
    ] = None,
) -> None:
    """Fit SVR/RF/Ridge against the subjective scores and compare with baselines."""
    import pandas as pd

    from .model.dataset import build_livenflx_features
    from .model.features import feature_columns
    from .model.livenflx import LABEL_COLUMN, load_release_splits, released_split_masks
    from .model.train import fit_final_model, masks_to_splits, save_model, save_report
    from .model.train import train_and_evaluate as fit

    cfg, mos = _mos_paths(config, mos_config)
    cached = features_csv or cfg.paths.datasets / LIVENFLX_FEATURES_CSV
    table = (
        pd.read_csv(cached)
        if cached.exists()
        else build_livenflx_features(
            mos.release_dir, scope=mos.scope, metrics=mos.metrics, poolings=mos.poolings
        )
    )

    columns = feature_columns(table, metrics=mos.metrics, poolings=mos.poolings)
    splits = None
    if mos.split_mode == "released_splits":
        masks = released_split_masks(
            table, load_release_splits(mos.split_matrix), n_trials=mos.n_trials
        )
        splits = masks_to_splits(masks)

    report = fit(
        table,
        features=columns,
        label=LABEL_COLUMN,
        n_splits=mos.n_splits,
        random_seed=mos.random_seed,
        baseline_metrics=[c for c in mos.baselines if c in table.columns],
        splits=splits,
    )
    save_report(report, cfg.paths.models / LIVENFLX_REPORT_JSON)
    if report.pooled_is_valid:
        report.oof.to_csv(cfg.paths.models / "livenflx_oof.csv", index=False)
    model = fit_final_model(
        table, columns, report.label, report.best_model, report.best_params, mos.random_seed
    )
    save_model(model, cfg.paths.models / f"livenflx_{report.best_model}.joblib")

    console.print(_results_table(report.table.to_dict(orient="records")))
    console.print(f"[dim]{_scope_note(mos.scope)}  split_mode={mos.split_mode}[/dim]")
    _verdict(report)


def _verdict(report: Any) -> None:
    """State plainly whether fusion earned its complexity, either way."""
    best = report.result_for(report.best_model)
    baseline = max(
        (r for r in report.results if r.name.endswith(" alone")),
        key=lambda r: r.srocc,
        default=None,
    )
    if best is None or baseline is None:
        return
    spread = next((s.srocc_std for s in report.fold_summaries if s.name == report.best_model), 0.0)
    if best.srocc >= baseline.srocc:
        margin = best.srocc - baseline.srocc
        note = " (inside fold noise)" if margin < spread else ""
        console.print(
            f"[green]{report.best_model} SROCC {best.srocc:.4f} >= best single metric "
            f"{baseline.name} {baseline.srocc:.4f}{note}.[/green]"
        )
    else:
        console.print(
            f"[yellow]{baseline.name} (SROCC {baseline.srocc:.4f}) still out-ranks the fused "
            f"{report.best_model} ({best.srocc:.4f}). A strong single metric is hard to beat; "
            "that is a legitimate result and belongs in the README as-is.[/yellow]"
        )


@mos_app.command("validate")
def mos_validate(
    ours: Annotated[
        Path,
        typer.Option("--ours", help="Feature CSV measured by PixelJudge's own measure pipeline."),
    ],
    config: ConfigOption = DEFAULT_CONFIG,
    mos_config: MosConfigOption = DEFAULT_MOS_CONFIG,
) -> None:
    """Compare our own measurements against the release's, before re-training.

    Stage 3 of the subjective work: the release ships its authors' quality vectors,
    so measuring the same videos with our pipeline and correlating the two is an
    independent check on our measurement code. Run this *before* re-training on our
    features - if the two disagree, a second regression tells you nothing.

    Needs the LIVE-Netflix source and distorted videos, which are distributed by
    request (a form, then an emailed password) rather than by direct download. See
    DATA_CARD.md.
    """
    import pandas as pd

    from .model.dataset import build_livenflx_features
    from .model.features import compare_feature_tables
    from .viz.plots import save_table

    cfg, mos = _mos_paths(config, mos_config)
    if not ours.exists():
        console.print(f"[red]no measured feature table at {ours}[/red]")
        raise typer.Exit(1)

    theirs = build_livenflx_features(
        mos.release_dir, scope=mos.scope, metrics=mos.metrics, poolings=mos.poolings
    )
    agreement = compare_feature_tables(theirs, pd.read_csv(ours))
    save_table(agreement, cfg.paths.reports / "mos_feature_agreement.csv")

    table = Table(title="Our measurements vs the release's", header_style="bold")
    for column in agreement.columns:
        table.add_column(str(column))
    for row in agreement.to_dict(orient="records"):
        table.add_row(*(str(value) for value in row.values()))
    console.print(table)

    vmaf = agreement[agreement["feature"].str.startswith("vmaf")]
    if not vmaf.empty and float(vmaf["spearman"].min()) < 0.95:
        console.print(
            "[yellow]VMAF rank agreement is below 0.95. That is a real discrepancy, not a "
            "rounding difference - check the VMAF model version, the pooling, and whether "
            "the two videos are time-aligned before trusting anything downstream.[/yellow]"
        )


@mos_app.command("table")
def mos_table(
    config: ConfigOption = DEFAULT_CONFIG,
    mos_config: MosConfigOption = DEFAULT_MOS_CONFIG,
) -> None:
    """Regenerate the correlation table and scatter from the cached training report."""
    import pandas as pd

    from .viz.plots import (
        correlation_table,
        plot_correlation_bars,
        plot_predicted_vs_mos,
        save_table,
    )

    cfg, mos = _mos_paths(config, mos_config)
    report_path = cfg.paths.models / LIVENFLX_REPORT_JSON
    if not report_path.exists():
        console.print(f"[red]no training report at {report_path}; run `pixeljudge mos train`[/red]")
        raise typer.Exit(1)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    table = correlation_table(payload["results"])
    save_table(table, cfg.paths.reports / LIVENFLX_TABLE_CSV)
    folds = pd.DataFrame(payload.get("fold_summaries", []))
    if not folds.empty:
        save_table(folds, cfg.paths.reports / "mos_correlation_folds.csv")
        plot_correlation_bars(
            folds,
            cfg.paths.reports / "mos_correlation.png",
            title=f"LIVE-Netflix: agreement with subjective score ({mos.scope})",
            caption=_scope_note(mos.scope),
        )

    oof_path = cfg.paths.models / "livenflx_oof.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path)
        best = f"pred_{payload['best_model']}"
        if best in oof.columns:
            plot_predicted_vs_mos(
                oof,
                cfg.paths.reports / "mos_predicted_vs_subjective.png",
                prediction_column=best,
                label_column=payload["label"],
                title=f"{payload['best_model']} vs subjective score ({mos.scope})",
            )

    console.print(_results_table(table.to_dict(orient="records")))
    console.print(f"[dim]{_scope_note(mos.scope)}[/dim]")


def main() -> None:
    """Entry point that turns our own exceptions into one clear line.

    A missing binary or a typo in a YAML file is a user-facing problem, not a bug,
    so it should not print a traceback.
    """
    try:
        app()
    except PixelJudgeError as exc:
        console.print(f"[bold red]error:[/bold red] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
