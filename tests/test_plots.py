"""Plot and table tests.

Figures are checked for "a non-trivial file appeared", not pixel-compared: a
screenshot test would fail on every matplotlib release and tell us nothing about
the analysis. What is worth asserting is that a missing column produces a clear
error instead of a blank panel, and that the markdown table renderer (hand-rolled
to avoid a dependency) is correct.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pixeljudge.errors import DatasetError
from pixeljudge.viz.plots import (
    correlation_table,
    plot_banding_comparison,
    plot_banding_gallery,
    plot_convex_hull,
    plot_metric_vs_banding,
    plot_per_frame_quality,
    plot_predicted_vs_mos,
    plot_rd_curves,
    save_table,
    stretch_for_display,
    to_markdown,
    worst_region,
)


@pytest.fixture()
def metrics() -> pd.DataFrame:
    rows = []
    for ladder, factor in (("apple_h264", 1.0), ("netflix_style_hevc", 0.55)):
        for rate, vmaf in zip(
            (500.0, 1000.0, 2000.0, 4000.0), (62.0, 78.0, 89.0, 95.0), strict=True
        ):
            rows.append(
                {
                    "source": "clip",
                    "ladder": ladder,
                    "codec": "h264" if "h264" in ladder else "hevc",
                    "height": 720,
                    "actual_bitrate_kbps": rate * factor,
                    "vmaf": vmaf,
                    "psnr_y": 30.0 + vmaf / 6,
                    "banding_max": 100.0 - vmaf,
                }
            )
    return pd.DataFrame(rows)


def _written(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 1000  # a real PNG, not a stub


def test_rd_curves_are_written(metrics: pd.DataFrame, tmp_path: Path) -> None:
    assert _written(plot_rd_curves(metrics, tmp_path / "rd.png"))


def test_rd_curves_can_group_by_codec(metrics: pd.DataFrame, tmp_path: Path) -> None:
    assert _written(plot_rd_curves(metrics, tmp_path / "rd2.png", group_column="codec"))


def test_rd_curves_missing_column_is_a_clear_error(tmp_path: Path) -> None:
    frame = pd.DataFrame({"vmaf": [90.0], "ladder": ["x"]})
    with pytest.raises(DatasetError, match="missing column"):
        plot_rd_curves(frame, tmp_path / "rd.png")


def test_metric_vs_banding_is_written(metrics: pd.DataFrame, tmp_path: Path) -> None:
    assert _written(plot_metric_vs_banding(metrics, tmp_path / "blind.png"))


def test_convex_hull_plot_accepts_a_hull_overlay(metrics: pd.DataFrame, tmp_path: Path) -> None:
    hull = metrics.nlargest(3, "vmaf")
    assert _written(plot_convex_hull(metrics, tmp_path / "hull.png", hull=hull))


def test_convex_hull_plot_without_a_hull(metrics: pd.DataFrame, tmp_path: Path) -> None:
    assert _written(plot_convex_hull(metrics, tmp_path / "hull2.png"))


def test_predicted_vs_mos_is_written(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    oof = pd.DataFrame(
        {
            "content": ["a"] * 10 + ["b"] * 10,
            "mos": rng.uniform(1, 5, 20),
            "pred_svr_rbf": rng.uniform(1, 5, 20),
        }
    )
    assert _written(
        plot_predicted_vs_mos(oof, tmp_path / "scatter.png", prediction_column="pred_svr_rbf")
    )


def test_predicted_vs_mos_works_without_a_group_column(tmp_path: Path) -> None:
    oof = pd.DataFrame({"mos": [1.0, 3.0, 5.0], "pred_x": [1.2, 2.8, 4.6]})
    assert _written(
        plot_predicted_vs_mos(oof, tmp_path / "scatter2.png", prediction_column="pred_x")
    )


def test_per_frame_plot_marks_the_pooled_values(tmp_path: Path) -> None:
    per_frame = pd.DataFrame({"frame": range(50), "vmaf": np.linspace(95, 40, 50)})
    assert _written(plot_per_frame_quality(per_frame, tmp_path / "frames.png"))


def test_banding_gallery_needs_at_least_one_frame(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="no evidence frames"):
        plot_banding_gallery([], tmp_path / "gallery.png")


def test_banding_gallery_renders_saved_frames(tmp_path: Path) -> None:
    import cv2

    from pixeljudge.artifacts.detectors import banded_gradient

    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), banded_gradient(levels=6).astype(np.uint8))
    out = plot_banding_gallery(
        [(image_path, "clip\nVMAF 88.2 | banding 128.0")], tmp_path / "gallery.png"
    )
    assert _written(out)


def test_banding_gallery_survives_an_unreadable_frame(tmp_path: Path) -> None:
    # An evidence frame can go missing between the scan and the report; the gallery
    # should say so in the panel rather than crash the whole report.
    out = plot_banding_gallery([(tmp_path / "ghost.png", "missing")], tmp_path / "g2.png")
    assert _written(out)


def test_save_table_writes_csv_and_markdown(tmp_path: Path) -> None:
    frame = pd.DataFrame({"name": ["vmaf alone"], "srocc": [0.9563]})
    path = save_table(frame, tmp_path / "correlations.csv")
    assert path.exists()
    markdown = path.with_suffix(".md").read_text(encoding="utf-8")
    assert "| name | srocc |" in markdown
    assert "0.9563" in markdown


def test_markdown_renders_a_pipe_table() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    lines = to_markdown(frame).strip().splitlines()
    assert lines[0] == "| a | b |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 1 | x |"


def test_markdown_trims_trailing_zeros_from_floats() -> None:
    frame = pd.DataFrame({"v": [0.5000, 1.0, 0.9563]})
    rendered = to_markdown(frame)
    assert "| 0.5 |" in rendered
    assert "| 1 |" in rendered
    assert "| 0.9563 |" in rendered


def test_correlation_table_is_sorted_and_column_ordered() -> None:
    results = [
        {"name": "psnr_y alone", "n": 70, "plcc": 0.90, "srocc": 0.88, "krocc": 0.7, "rmse": 0.5},
        {"name": "vmaf alone", "n": 70, "plcc": 0.96, "srocc": 0.95, "krocc": 0.8, "rmse": 0.3},
    ]
    table = correlation_table(results)
    assert list(table.columns) == ["name", "n", "plcc", "srocc", "krocc", "rmse"]
    assert table["name"].iloc[0] == "vmaf alone"


def test_correlation_table_on_empty_input() -> None:
    assert correlation_table([]).empty


# ---------------------------------------------------------------------------
# The master-vs-encode comparison figure
# ---------------------------------------------------------------------------


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    """A smooth master and an encode whose bottom-right quadrant has been quantised."""
    import cv2

    from pixeljudge.artifacts.detectors import banded_gradient, smooth_gradient

    reference = smooth_gradient(width=320, height=240).astype(np.uint8)
    distorted = reference.copy()
    distorted[120:240, 160:320] = banded_gradient(width=160, height=120, levels=5).astype(np.uint8)
    reference_path, distorted_path = tmp_path / "ref.png", tmp_path / "dis.png"
    cv2.imwrite(str(reference_path), reference)
    cv2.imwrite(str(distorted_path), distorted)
    return reference_path, distorted_path


def test_comparison_figure_is_written(tmp_path: Path) -> None:
    reference_path, distorted_path = _write_pair(tmp_path)
    out = plot_banding_comparison(
        reference_path, distorted_path, tmp_path / "cmp.png", caption="VMAF 95.0"
    )
    assert _written(out)


def test_comparison_figure_reports_a_missing_frame(tmp_path: Path) -> None:
    reference_path, _ = _write_pair(tmp_path)
    with pytest.raises(DatasetError, match="cannot build the comparison figure"):
        plot_banding_comparison(reference_path, tmp_path / "ghost.png", tmp_path / "cmp.png")


def _read(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imread(str(path))
    assert image is not None, path
    return image


def test_worst_region_finds_the_quadrant_that_changed(tmp_path: Path) -> None:
    """The crop must be chosen by the detector, not by eye.

    The banded patch is in the bottom-right quadrant, so the crop's centre has to land
    there. Picking a crop by hand is what invites "you chose the worst bit".
    """
    reference_path, distorted_path = _write_pair(tmp_path)
    reference = _read(reference_path)
    distorted = _read(distorted_path)
    rows, columns = worst_region(reference, distorted)
    centre_y = (rows.start + rows.stop) / 2
    centre_x = (columns.start + columns.stop) / 2
    assert centre_y > 120  # bottom half
    assert centre_x > 160  # right half


def test_worst_region_stays_inside_the_frame(tmp_path: Path) -> None:
    reference_path, _ = _write_pair(tmp_path)
    reference = _read(reference_path)
    rows, columns = worst_region(reference, reference)  # no difference at all
    assert rows.start >= 0 and columns.start >= 0
    assert rows.stop <= reference.shape[0]
    assert columns.stop <= reference.shape[1]


def test_display_stretch_is_linear_and_shared() -> None:
    """The stretch may make an edge visible; it must not invent one."""
    reference = np.linspace(60, 90, 256, dtype=np.float32).reshape(16, 16)
    stretched = stretch_for_display(reference, reference)
    # Monotonic in, monotonic out, and spanning the display range.
    assert stretched.min() == 0
    assert stretched.max() == 255
    flat = stretched.ravel().astype(int)
    assert all(b >= a for a, b in zip(flat, flat[1:], strict=False))
    # A second image gets the *same* mapping, so the two stay comparable: a constant
    # offset in must be a constant offset out, not a re-normalisation to full range.
    offset = stretch_for_display(reference + 5.0, reference)
    assert offset.max() == 255  # clipped at the top
    assert int(offset.min()) > int(stretched.min())
