"""Dataset loading tests.

Two things are worth guarding here. First, the label file is parsed with ``ast``
rather than imported, so a file that is not a pair of literal lists must fail
loudly instead of executing. Second, the failure mode when the *videos* are absent
(the common case, since the label file ships without them) has to be a clear
message about where to get them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from pixeljudge.errors import DatasetError
from pixeljudge.model.dataset import (
    build_feature_table,
    content_groups,
    describe,
    load_feature_table,
    load_nflx_labels,
    parse_vmaf_dataset_file,
    require_columns,
)

FIXTURES = Path(__file__).parent / "fixtures"

DATASET_FILE = """
dataset_name = 'NFLX_dataset_public_demo'
yuv_fmt = 'yuv420p'
width = 1920
height = 1080

ref_videos = [
    {'content_id': 0, 'content_name': 'BigBuckBunny', 'path': '/media/ref/bbb.yuv'},
    {'content_id': 1, 'content_name': 'CrowdRun', 'path': '/media/ref/crowd.yuv'},
]

dis_videos = [
    {'content_id': 0, 'asset_id': 0, 'dmos': 92.5, 'path': '/media/dis/bbb_high.yuv'},
    {'content_id': 0, 'asset_id': 1, 'dmos': 61.0, 'path': '/media/dis/bbb_low.yuv'},
    {'content_id': 1, 'asset_id': 2, 'dmos': 88.0, 'path': '/media/dis/crowd_high.yuv'},
]
"""


@pytest.fixture()
def dataset_file(tmp_path: Path) -> Path:
    path = tmp_path / "NFLX_dataset_public.py"
    path.write_text(DATASET_FILE, encoding="utf-8")
    return path


def test_parses_reference_and_distorted_lists(dataset_file: Path) -> None:
    payload = parse_vmaf_dataset_file(dataset_file)
    assert len(payload["ref_videos"]) == 2
    assert len(payload["dis_videos"]) == 3
    assert payload["ref_videos"][0]["content_name"] == "BigBuckBunny"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        parse_vmaf_dataset_file(tmp_path / "absent.py")


def test_unparseable_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.py"
    path.write_text("ref_videos = [ {'content_id': \n", encoding="utf-8")
    with pytest.raises(DatasetError, match="not a parseable"):
        parse_vmaf_dataset_file(path)


def test_file_without_the_expected_names_raises(tmp_path: Path) -> None:
    path = tmp_path / "other.py"
    path.write_text("something_else = [1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="does not define"):
        parse_vmaf_dataset_file(path)


def test_function_calls_are_refused(tmp_path: Path) -> None:
    # The file is data. If it tries to compute its contents we stop, rather than
    # executing a downloaded script to read a list of numbers.
    path = tmp_path / "sneaky.py"
    path.write_text(
        "ref_videos = list(open('/etc/passwd'))\ndis_videos = []\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="unsupported expression Call"):
        parse_vmaf_dataset_file(path)


def test_paths_built_by_concatenation_are_resolved(tmp_path: Path) -> None:
    # This is how the real NFLX-P file writes its paths, and plain literal_eval
    # cannot read it: `ref_dir` is a name and `+` is an operator, not a literal.
    path = tmp_path / "concat.py"
    path.write_text(
        "ref_dir = '/media/ref'\n"
        "dis_dir = '/media/dis'\n"
        "ref_videos = [{'content_id': 0, 'content_name': 'A', 'path': ref_dir + '/a.yuv'}]\n"
        "dis_videos = [{'content_id': 0, 'asset_id': 1, 'dmos': 50.0,"
        " 'path': dis_dir + '/a_50_480_1050.yuv'}]\n",
        encoding="utf-8",
    )
    payload = parse_vmaf_dataset_file(path)
    assert payload["ref_videos"][0]["path"] == "/media/ref/a.yuv"
    assert payload["dis_videos"][0]["path"] == "/media/dis/a_50_480_1050.yuv"


def test_unrelated_unsupported_assignments_are_skipped(tmp_path: Path) -> None:
    # Real descriptors carry extra assignments we do not need. Only the two lists
    # we actually read have to be understandable.
    path = tmp_path / "extra.py"
    path.write_text(
        "import os\n"
        "quality_runner = os.environ.get('RUNNER')\n"
        "ref_videos = [{'content_id': 0, 'content_name': 'A', 'path': 'a.yuv'}]\n"
        "dis_videos = [{'content_id': 0, 'asset_id': 1, 'dmos': 50.0, 'path': 'b.yuv'}]\n",
        encoding="utf-8",
    )
    assert len(parse_vmaf_dataset_file(path)["dis_videos"]) == 1


def test_hidden_references_are_dropped_by_default(tmp_path: Path) -> None:
    # A clip scored against itself has infinite PSNR and DMOS 100: a degenerate
    # point that flatters any correlation.
    path = tmp_path / "hidden.py"
    path.write_text(
        "ref_videos = [{'content_id': 0, 'content_name': 'A', 'path': '/r/a.yuv'}]\n"
        "dis_videos = [\n"
        "  {'content_id': 0, 'asset_id': 0, 'dmos': 100.0, 'path': '/r/a.yuv'},\n"
        "  {'content_id': 0, 'asset_id': 1, 'dmos': 60.0, 'path': '/d/a_60_480_1050.yuv'},\n"
        "]\n",
        encoding="utf-8",
    )
    assert len(load_nflx_labels(path)) == 1
    assert len(load_nflx_labels(path, include_hidden_reference=True)) == 2


def test_encoding_parameters_are_read_from_the_filename(tmp_path: Path) -> None:
    # NFLX-P encodes {content}_{expert score}_{height}_{bitrate} into the name, so
    # resolution and bitrate are recoverable without the videos.
    path = tmp_path / "named.py"
    path.write_text(
        "ref_videos = [{'content_id': 0, 'content_name': 'A', 'path': '/r/a_25fps.yuv'}]\n"
        "dis_videos = [{'content_id': 0, 'asset_id': 1, 'dmos': 61.7,"
        " 'path': '/d/BigBuckBunny_50_480_1050.yuv'}]\n",
        encoding="utf-8",
    )
    row = load_nflx_labels(path).iloc[0]
    assert row["expert_score"] == 50
    assert row["height"] == 480
    assert row["target_bitrate_kbps"] == 1050


def test_unparseable_filenames_give_empty_parameters(tmp_path: Path) -> None:
    path = tmp_path / "odd.py"
    path.write_text(
        "ref_videos = [{'content_id': 0, 'content_name': 'A', 'path': '/r/a.yuv'}]\n"
        "dis_videos = [{'content_id': 0, 'asset_id': 1, 'dmos': 61.7,"
        " 'path': '/d/some_odd_name.yuv'}]\n",
        encoding="utf-8",
    )
    row = load_nflx_labels(path).iloc[0]
    assert pd.isna(row["expert_score"])
    assert pd.isna(row["height"])


def test_load_nflx_labels_builds_one_row_per_distorted_clip(dataset_file: Path) -> None:
    labels = load_nflx_labels(dataset_file)
    assert len(labels) == 3
    assert set(labels["content"]) == {"BigBuckBunny", "CrowdRun"}
    assert labels["mos"].tolist() == [92.5, 61.0, 88.0]
    assert str(labels.loc[0, "reference_path"]).endswith("bbb.yuv")


def test_load_nflx_labels_rejects_an_unknown_content_id(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    path.write_text(
        "ref_videos = [{'content_id': 0, 'content_name': 'A', 'path': 'a.yuv'}]\n"
        "dis_videos = [{'content_id': 9, 'asset_id': 1, 'dmos': 50.0, 'path': 'x.yuv'}]\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="unknown content_id"):
        load_nflx_labels(path)


def test_load_nflx_labels_rejects_a_clip_without_a_score(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    path.write_text(
        "ref_videos = [{'content_id': 0, 'content_name': 'A', 'path': 'a.yuv'}]\n"
        "dis_videos = [{'content_id': 0, 'asset_id': 1, 'path': 'x.yuv'}]\n",
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="no subjective score"):
        load_nflx_labels(path)


def fake_measure(distorted: Path, reference: Path) -> Mapping[str, float]:
    """Stand-in for a real libvmaf pass, so this file needs no ffmpeg."""
    return {"vmaf": 90.0, "psnr_y": 42.0, "float_ssim": 0.99, "float_ms_ssim": 0.995}


def test_build_feature_table_joins_measurements(tmp_path: Path, dataset_file: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    for name in ("bbb.yuv", "crowd.yuv", "bbb_high.yuv", "bbb_low.yuv", "crowd_high.yuv"):
        (media / name).write_bytes(b"\x00")

    labels = load_nflx_labels(dataset_file)
    table = build_feature_table(labels, fake_measure, root=media)
    assert len(table) == 3
    assert {"vmaf", "psnr_y", "mos", "content"}.issubset(table.columns)


def test_build_feature_table_reports_partially_missing_media(
    tmp_path: Path, dataset_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    media = tmp_path / "media"
    media.mkdir()
    for name in ("bbb.yuv", "bbb_high.yuv"):
        (media / name).write_bytes(b"\x00")

    labels = load_nflx_labels(dataset_file)
    with caplog.at_level("WARNING"):
        table = build_feature_table(labels, fake_measure, root=media)
    assert len(table) == 1
    assert "had no media on disk" in caplog.text


def test_build_feature_table_explains_how_to_get_the_videos(
    tmp_path: Path, dataset_file: Path
) -> None:
    labels = load_nflx_labels(dataset_file)
    with pytest.raises(DatasetError, match="DATA_CARD.md"):
        build_feature_table(labels, fake_measure, root=tmp_path / "empty")


def test_load_feature_table_validates_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "features.csv"
    pd.DataFrame({"vmaf": [90.0], "mos": [4.0]}).to_csv(path, index=False)
    with pytest.raises(DatasetError, match="missing required column"):
        load_feature_table(path)


def test_load_feature_table_reads_the_shipped_fixture() -> None:
    table = load_feature_table(FIXTURES / "features_sample.csv")
    assert len(table) == 48
    assert table["content"].nunique() == 8
    # The fixture must announce that it is not a result.
    assert (table["source"] == "synthetic-fixture-not-a-result").all()


def test_require_columns_names_what_is_missing() -> None:
    with pytest.raises(DatasetError, match=r"\['vmaf'\]"):
        require_columns(pd.DataFrame({"mos": [1.0]}), ["vmaf"], source="my table")


def test_content_groups_are_strings() -> None:
    table = pd.DataFrame({"content": [1, 1, 2], "mos": [1.0, 2.0, 3.0]})
    groups = content_groups(table)
    assert list(groups) == ["1", "1", "2"]


def test_describe_summarises_shape_and_labels() -> None:
    table = load_feature_table(FIXTURES / "features_sample.csv")
    summary = describe(table)
    assert summary["rows"] == 48
    assert summary["contents"] == 8
    assert summary["clips_per_content"] == 6.0
    assert 1.0 <= summary["label_min"] <= summary["label_max"] <= 5.0
