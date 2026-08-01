"""Run a master clip through a ladder and record what came out.

Two design decisions worth calling out:

* Encoding is *idempotent*. If the output file already exists we probe it and
  move on. A full four-codec sweep is hours of CPU, and losing all of it because
  the tenth rung failed would be miserable.
* We record the *measured* bitrate of every output, not the requested one. The
  encoder often misses its target (especially at low bitrates or in CRF mode
  where there is no target at all), and a rate-distortion curve plotted against
  requested bitrate is simply wrong.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import EncodeConfig, LadderConfig, Rung
from ..errors import FfmpegError
from ..io.ffmpeg import VideoInfo, available_encoders, probe, run_ffmpeg
from ..logging_conf import get_logger
from .codecs import CodecSpec, build_encode_args, get_codec

log = get_logger(__name__)


@dataclass(frozen=True)
class EncodedRung:
    """One distorted output plus everything needed to interpret it later."""

    source: str
    ladder: str
    codec: str
    rung: str
    path: str
    width: int
    height: int
    target_bitrate_kbps: int | None
    crf: int | None
    actual_bitrate_kbps: float
    size_bytes: int
    n_frames: int | None
    encode_seconds: float

    @property
    def file(self) -> Path:
        return Path(self.path)


def output_path(
    source: Path, ladder: LadderConfig, rung: Rung, spec: CodecSpec, out_dir: Path
) -> Path:
    """``<out_dir>/<source stem>__<ladder>__<rung>.<ext>``.

    The name carries enough information to identify a file on sight, which
    matters when a directory holds a hundred encodes.
    """
    return out_dir / f"{source.stem}__{ladder.name}__{rung.name}.{spec.container}"


def encode_rung(
    source: Path,
    rung: Rung,
    ladder: LadderConfig,
    cfg: EncodeConfig,
    out_dir: Path,
    *,
    source_info: VideoInfo | None = None,
    overwrite: bool = False,
) -> EncodedRung:
    """Encode a single rung and return its record."""
    spec = get_codec(ladder.codec)
    if spec.encoder not in available_encoders():
        raise FfmpegError(
            f"this ffmpeg has no {spec.encoder} encoder, needed for codec {ladder.codec!r}. "
            "Run 'pixeljudge doctor' to see what is available."
        )

    info = source_info or probe(source)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = output_path(source, ladder, rung, spec, out_dir)

    if destination.exists() and not overwrite:
        log.info("skip (exists): %s", destination.name)
        elapsed = 0.0
    else:
        args = [
            "-i",
            str(source),
            *build_encode_args(spec, rung, cfg, info.fps),
            str(destination),
        ]
        log.info(
            "encode %s -> %s (%s, %s)",
            source.name,
            destination.name,
            ladder.codec,
            f"crf {rung.crf}" if rung.crf is not None else f"{rung.bitrate_kbps} kbps",
        )
        started = time.perf_counter()
        run_ffmpeg(args)
        elapsed = time.perf_counter() - started

    out_info = probe(destination)
    return EncodedRung(
        source=str(source),
        ladder=ladder.name,
        codec=ladder.codec,
        rung=rung.name,
        path=str(destination),
        width=out_info.width,
        height=out_info.height,
        target_bitrate_kbps=rung.bitrate_kbps,
        crf=rung.crf,
        actual_bitrate_kbps=round(out_info.bitrate_kbps, 2),
        size_bytes=out_info.size_bytes,
        n_frames=out_info.n_frames,
        encode_seconds=round(elapsed, 2),
    )


def encode_ladder(
    source: Path,
    ladder: LadderConfig,
    cfg: EncodeConfig,
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> list[EncodedRung]:
    """Encode every rung of one ladder. Probes the source once.

    Rungs taller than the master are skipped: upscaling a 720p master to 1080p
    invents no detail and would only produce a misleading RD point.
    """
    info = probe(source)
    results: list[EncodedRung] = []
    for rung in ladder.sorted_by_height():
        if rung.height > info.height:
            log.warning(
                "skip rung %s: taller than the %dp master (upscaling adds no information)",
                rung.name,
                info.height,
            )
            continue
        results.append(
            encode_rung(
                source,
                rung,
                ladder,
                cfg,
                out_dir,
                source_info=info,
                overwrite=overwrite,
            )
        )
    return results


def manifest_path(source: Path, ladder: LadderConfig, out_dir: Path) -> Path:
    return out_dir / f"{source.stem}__{ladder.name}.manifest.json"


def write_manifest(rungs: list[EncodedRung], path: Path) -> Path:
    """Save the ladder's records so ``measure`` does not have to guess pairings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(rung) for rung in rungs]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("wrote manifest %s (%d rungs)", path.name, len(rungs))
    return path


def read_manifest(path: Path) -> list[EncodedRung]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EncodedRung(**row) for row in data]
