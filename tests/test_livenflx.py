"""LIVE-Netflix loader and pooling tests, on synthetic .mat files.

The fixtures are written here rather than shipped, and they mimic the real
on-disk shape: MATLAB files named ``content_C_seq_S.mat``, upper-case metric keys,
1x1 arrays for every scalar, and ``MSSIM`` rather than ``MS_SSIM``. That keeps CI
independent of the 10 MB release while still exercising the parts that would
actually break - the naming convention, the scalar unwrapping, and the condition
scope.

The two behaviours worth the most here are the scope filter (getting it wrong
means silently correlating compression metrics against rebuffering) and the
group-preserving split mapping (getting it wrong means content leakage, which
inflates every number in the report without failing anything).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from pixeljudge.errors import DatasetError
from pixeljudge.model.features import feature_columns, pool_to_features
from pixeljudge.model.livenflx import (
    load_livenflx,
    load_release_splits,
    read_video_mat,
    released_split_masks,
)

FPS = 25.0


def write_video(
    directory: Path,
    content: int,
    sequence: int,
    *,
    n_frames: int = 5,
    n_stalls: int = 0,
    stall_seconds: float = 0.0,
    score: float = 0.5,
    vmaf: list[float] | None = None,
) -> Path:
    """One synthetic .mat in the release's exact layout.

    ``Nframes`` is deliberately the *playout* length (vector length plus the
    frozen frames of any stall), because that is the relationship the real files
    have and the loader checks it.
    """
    values = np.asarray(vmaf if vmaf is not None else [float(i) for i in range(n_frames)])
    payload: dict[str, object] = {
        "Nframes": np.array([[n_frames + stall_seconds * FPS]]),
        "vid_fps": np.array([[FPS]]),
        "final_subj_score": np.array([[score]]),
        "ns": np.array([[n_stalls]]),
        "ds": np.array([[stall_seconds]]),
        "lt": np.array([[0.0]]),
        "tsl": np.array([[1.0]]),
    }
    for key, offset in (
        ("VMAF", 0.0),
        ("PSNR", 30.0),
        ("SSIM", 0.9),
        ("MSSIM", 0.8),
        ("STRRED", 10.0),
        ("NIQE", 4.0),
    ):
        payload[f"{key}_vec"] = values + offset
    path = directory / f"content_{content}_seq_{sequence}.mat"
    savemat(str(path), payload)
    return path


@pytest.fixture
def release(tmp_path: Path) -> Path:
    """Three contents x four patterns: three continuous, one with a stall.

    Shaped after the real release, where each content appears under four
    continuous-playback patterns and four rebuffering ones. Three contents rather
    than two because a content-grouped split has to leave at least two contents on
    the training side - hyperparameters are tuned with an inner grouped split,
    which needs two groups to divide.
    """
    directory = tmp_path / "release"
    directory.mkdir()
    for content in (1, 2, 3):
        write_video(directory, content, 0, score=1.0)
        write_video(directory, content, 2, score=0.4)
        write_video(directory, content, 4, score=0.7)
        write_video(directory, content, 1, n_stalls=1, stall_seconds=2.0, score=-0.5)
    return directory


def test_read_video_mat_unwraps_scalars_and_vectors(tmp_path: Path) -> None:
    path = write_video(tmp_path, 3, 4, n_frames=5, score=0.25)
    record = read_video_mat(path)

    assert record["video_id"] == "content_3_seq_4"
    assert record["content"] == "content_03"
    assert record["condition"] == "seq_4"
    assert record["condition_index"] == 4
    # loadmat hands back array([[0.25]]); the loader must yield a plain float.
    assert record["subj_score"] == pytest.approx(0.25)
    assert isinstance(record["subj_score"], float)
    assert record["n_frames"] == 5
    assert record["ms_ssim"].tolist() == pytest.approx([0.8, 1.8, 2.8, 3.8, 4.8])


def test_filename_convention_is_enforced(tmp_path: Path) -> None:
    stray = tmp_path / "not_a_video.mat"
    savemat(str(stray), {"Nframes": np.array([[1]])})
    with pytest.raises(DatasetError, match="naming convention"):
        read_video_mat(stray)


def test_ragged_vectors_are_rejected(tmp_path: Path) -> None:
    path = write_video(tmp_path, 1, 0, n_frames=5)
    from scipy.io import loadmat

    payload = {k: v for k, v in loadmat(str(path)).items() if not k.startswith("__")}
    payload["PSNR_vec"] = np.arange(4.0).reshape(1, 4)  # one frame short
    savemat(str(path), payload)
    with pytest.raises(DatasetError, match="disagree in length"):
        read_video_mat(path)


def test_compression_only_scope_drops_rebuffered_videos(release: Path) -> None:
    table = load_livenflx(release, scope="compression_only")

    assert set(table["condition"].unique()) == {"seq_0", "seq_2", "seq_4"}
    assert table["video_id"].nunique() == 9
    assert (table["n_stalls"] == 0).all()
    # The grouping key must be on every row - that is what stops a later split
    # from being made without it.
    assert table["content"].notna().all()


def test_all_scope_keeps_everything(release: Path) -> None:
    table = load_livenflx(release, scope="all")
    assert table["video_id"].nunique() == 12
    assert (table["n_stalls"] > 0).any()


def test_unknown_scope_and_metric_are_rejected(release: Path) -> None:
    with pytest.raises(DatasetError, match="unknown condition scope"):
        load_livenflx(release, scope="everything")  # type: ignore[arg-type]
    with pytest.raises(DatasetError, match="unknown metric"):
        load_livenflx(release, metrics=["vmaf", "butteraugli"])


def test_missing_release_directory_says_where_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="DATA_CARD"):
        load_livenflx(tmp_path / "absent")


def test_pooling_matches_hand_computed_values(tmp_path: Path) -> None:
    directory = tmp_path / "release"
    directory.mkdir()
    write_video(directory, 1, 0, vmaf=[0.0, 10.0, 20.0, 30.0, 40.0])
    write_video(directory, 2, 0, vmaf=[0.0, 10.0, 20.0, 30.0, 40.0])

    features = pool_to_features(load_livenflx(directory), poolings=["mean", "hmean"])
    assert len(features) == 2

    row = features.iloc[0]
    assert row["vmaf_mean"] == pytest.approx(20.0)
    # libvmaf's offset harmonic mean: n / sum(1/(x+1)) - 1, which sits below the
    # arithmetic mean because the low frames dominate it.
    expected = 5.0 / sum(1.0 / (v + 1.0) for v in (0.0, 10.0, 20.0, 30.0, 40.0)) - 1.0
    assert row["vmaf_hmean"] == pytest.approx(expected)
    assert row["vmaf_hmean"] < row["vmaf_mean"]


def test_pooling_carries_identity_and_label(release: Path) -> None:
    features = pool_to_features(load_livenflx(release))
    assert len(features) == 9
    assert set(features["content"]) == {"content_01", "content_02", "content_03"}
    assert features["subj_score"].notna().all()
    # Row order is deterministic, which the positional split matrix depends on.
    assert list(features["video_id"]) == [
        "content_1_seq_0",
        "content_1_seq_2",
        "content_1_seq_4",
        "content_2_seq_0",
        "content_2_seq_2",
        "content_2_seq_4",
        "content_3_seq_0",
        "content_3_seq_2",
        "content_3_seq_4",
    ]


def test_feature_columns_reports_missing_combinations(release: Path) -> None:
    features = pool_to_features(load_livenflx(release))
    assert feature_columns(features, metrics=["vmaf"], poolings=["mean", "hmean"]) == [
        "vmaf_mean",
        "vmaf_hmean",
    ]
    with pytest.raises(DatasetError, match="missing"):
        feature_columns(features, metrics=["vmaf"], poolings=["mean", "median"])


def test_release_splits_map_by_identity_not_position(tmp_path: Path, release: Path) -> None:
    features = pool_to_features(load_livenflx(release))
    # Full 112-row matrix, as shipped. Mark contents 1-2 (rows 0-15) as training
    # and content 3 (rows 16-23) as test.
    matrix = np.zeros((112, 3), dtype=np.uint8)
    matrix[0:16, :] = 1
    path = tmp_path / "splits.mat"
    savemat(str(path), {"TrainingMatrix": matrix})

    masks = released_split_masks(features, load_release_splits(path))
    assert masks, "expected at least one usable trial"
    for mask in masks:
        train, test = set(features["content"][mask]), set(features["content"][~mask])
        # Scoping removed the stalled pattern from each content, so the table has
        # 9 rows where the matrix describes 24. A positional read would take the
        # matrix's first 9 rows and put every content on the training side.
        # Identity mapping keeps the intended content split intact.
        assert train == {"content_01", "content_02"}
        assert test == {"content_03"}


def test_degenerate_trials_are_dropped(tmp_path: Path, release: Path) -> None:
    features = pool_to_features(load_livenflx(release))
    matrix = np.ones((112, 2), dtype=np.uint8)  # everything is training: no test side
    path = tmp_path / "splits.mat"
    savemat(str(path), {"TrainingMatrix": matrix})
    with pytest.raises(DatasetError, match="no usable trial"):
        released_split_masks(features, load_release_splits(path))


# --- checks that need the real release on disk ------------------------------

RELEASE = Path("data/datasets/LIVE_NFLX_PublicData_VideoATLAS_Release")
needs_release = pytest.mark.skipif(
    not RELEASE.is_dir(), reason="LIVE-Netflix VideoATLAS release not present"
)


@pytest.mark.integration
@needs_release
def test_release_has_the_expected_shape() -> None:
    """14 contents x 8 patterns, of which 4 patterns are stall-free."""
    table = load_livenflx(RELEASE, scope="all")
    assert table["video_id"].nunique() == 112
    assert table["content"].nunique() == 14

    scoped = load_livenflx(RELEASE, scope="compression_only")
    assert scoped["video_id"].nunique() == 56
    assert scoped["content"].nunique() == 14
    assert set(scoped["condition"].unique()) == {"seq_0", "seq_2", "seq_4", "seq_7"}


@pytest.mark.integration
@needs_release
def test_our_mean_pooling_reproduces_the_releases_own_pooled_scores() -> None:
    """Our pooling must agree with the release's, on the release's own numbers.

    This is the half of the Stage 3 validation that can be run without the videos.
    Each .mat carries both the per-frame vector and the pooled scalar the authors
    computed from it, so pooling ours and comparing against theirs checks the
    pooling code against an independent implementation. It says nothing about our
    *measurement* code - that needs the source videos, which are behind a password
    request - but it does rule out the pooling half as a source of disagreement.
    """
    from scipy.io import loadmat

    features = pool_to_features(load_livenflx(RELEASE, scope="all"), poolings=["mean"])
    indexed = features.set_index("video_id")

    worst = 0.0
    for path in sorted(RELEASE.glob("content_*_seq_*.mat")):
        payload = loadmat(str(path))
        for release_key, ours in (("VMAF", "vmaf_mean"), ("PSNR", "psnr_mean")):
            theirs = float(np.ravel(payload[f"{release_key}_mean"])[0])
            # .loc on a DataFrame widens to pandas-stubs' full scalar union; the value
            # here is a float by construction (pool_metric returns floats).
            mine = float(indexed.loc[path.stem, ours])  # type: ignore[arg-type]
            worst = max(worst, abs(mine - theirs))

    # Floating-point noise only: same numbers, different summation order.
    assert worst < 1e-9, f"pooled means diverge from the release's by {worst}"
