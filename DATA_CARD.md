# Data card

What data PixelJudge uses, where it comes from, what it is licensed under, and
what each dataset can and cannot support.

## 1. Reference (master) video

| Clip | Source | Licence | Used for |
| --- | --- | --- | --- |
| `big_buck_bunny` | Blender Foundation, *Big Buck Bunny* | CC BY 3.0 | animated content, moderate motion |
| `sintel` | Blender Foundation, *Sintel* | CC BY 3.0 | animated content, dark scenes and gradients |
| `jellyfish` | Public test clip (high-motion, high-detail underwater footage) | free for testing | hard case: detail plus motion |
| `gradient` | Generated locally by `scripts/make_fixtures.py` | n/a (synthetic) | banding test case |

Obtained with `bash scripts/download_media.sh`, which downloads 720p 10-second
clips (about 10 MB each), keeps the first 6 seconds, and transcodes them to a
lossless local reference. Nothing is redistributed in this repository: `data/` is
gitignored and every clip is fetched from its original host.

### The mezzanine caveat (read this before quoting a VMAF number)

The downloaded clips are **already H.264 encoded**. They are mezzanine files, not
studio masters. We transcode them to lossless x264 so that our own reference stops
degrading, but the reference still carries whatever the original encode discarded.

Consequences, stated plainly:

* Absolute VMAF/PSNR values are **optimistic**. Detail the first encoder already
  removed cannot be lost again, so our encodes look slightly better than they
  would against a true master.
* Comparisons **between** our ladders and codecs on the same reference are
  unaffected: every one of them is measured against the same target, so the
  differences (and every BD-Rate in the report) remain valid.
* For a genuinely uncompressed reference, use Xiph's y4m sequences at
  <https://media.xiph.org/video/derf/>. They are correct and roughly 200x larger,
  which is the only reason they are not the default here.

The synthetic `gradient` clip has no such caveat: it is generated in-process and
encoded losslessly, so it is a true master.

## 2. Subjective quality datasets

Two, for different reasons. LIVE-Netflix is what the reported regression is trained
on; NFLX-P is kept wired up as a future third confirmation and as the basis for a
cross-dataset generalisation test.

### 2a. LIVE-Netflix Video QoE Database - VideoATLAS release (used)

From the [christosbampis/VideoATLAS_release](https://github.com/christosbampis/VideoATLAS_release)
repository, folder `LIVE_NFLX_PublicData_VideoATLAS_Release`.

| Property | Value |
| --- | --- |
| Type | full-reference features, pre-computed by the dataset authors |
| Contents | 14 source clips (1080p, 24/25/30 fps) |
| Videos | 112 = 14 contents x 8 streaming playout patterns |
| Subjects | 55+, rating **on a mobile device** |
| Label | `final_subj_score`: retrospective QoE, Z-scored per session and per subject (range -1.60 to 1.58, mean 0.004). **Not a 1-5 MOS.** |
| Why this one | it ships per-frame quality vectors *and* the subjective scores, so the regression needs no media access grant |
| Licence | UT Austin LIVE / CPS, free for research use with citation of Bampis & Bovik; see the release's own README, mirrored to `data/datasets/VideoATLAS_release_README.md` |

**On-disk shape**, which is not what the usual description of this release suggests -
verify before writing a parser. 112 MATLAB files named `content_{1..14}_seq_{0..7}`,
one per *video*, each holding that video's per-frame vectors (`VMAF_vec`, `PSNR_vec`,
`SSIM_vec`, `MSSIM_vec`, `STRRED_vec`, `NIQE_vec`, `GMSD_vec`, `PSNRhvs_oss_vec`),
pre-pooled scalars, the QoE covariates (`ns`, `ds`, `lt`, `tsl`) and the score. Keys
are upper-case and MS-SSIM is `MSSIM`.

**Condition scope matters more than anything else here.** Four of the eight patterns
contain rebuffering. Compression metrics cannot observe a stall, and the dataset makes
that concrete: for those 56 videos `Nframes - len(vec) == ds * fps` exactly, because
the vectors are computed on the stall-removed video. The default scope
(`compression_only` in `configs/livenflx.yaml`) keeps only the 56 continuous-playback
clips. Bitrate adaptation is retained.

The release also ships `TrainingMatrix_LIVENetflix_1000_trials.mat`: 1000 pre-generated
80/20 content splits (88 train / 24 test), used by `split_mode: released_splits` for
comparability with published results.

**The source videos are *not* in this release.** They are distributed from the LIVE
database page by a request form followed by an emailed password, and 11 of the 14
contents are proprietary Netflix material (the other 3 come from CDVL). That is why
the independent re-measurement in `pixeljudge mos validate` has not been run - see
`MODEL_CARD.md`.

### 2b. Netflix Public Dataset (NFLX-P) - pending

From the resources of the
[Netflix/vmaf](https://github.com/Netflix/vmaf) repository.

| Property | Value |
| --- | --- |
| Type | full-reference |
| References | 9 source clips |
| Distorted versions | ~70, produced by compression and scaling |
| Label | DMOS per distorted clip, from a controlled subjective study |
| Why this one | the distortions match our pipeline's, and it is what VMAF itself was validated on |
| Licence | as published by Netflix in the vmaf repository; check the repository's terms before redistributing |

The label file (`resource/dataset/NFLX_dataset_public.py`) is a Python module
listing reference/distorted paths with their DMOS. PixelJudge reads it with
`ast.literal_eval` rather than importing it: the labels are data, and executing a
downloaded file to read a list of numbers is a bad trade.

**The video files are distributed separately from the label file.** Obtain them
from the vmaf project's resources, then:

```bash
uv run pixeljudge train --features data/metrics/nflx_features.csv
```

after building the feature table with the label file plus `--media-root` pointing
at the directory holding the clips. Without the media, `build_feature_table`
raises a `DatasetError` that says exactly this, rather than silently producing a
smaller dataset.

### Datasets deliberately not used

| Dataset | Why not |
| --- | --- |
| KoNViD-1k, YouTube-UGC, LIVE-VQC, LSVQ | **No-reference.** These are "in the wild" clips with no pristine original, so PSNR/SSIM/VMAF cannot be computed against a source at all. This is a property of the data, not a limitation of the code. |
| MCL-V, BVI-HD | Full-reference and suitable, but access is by request. Good candidates for cross-dataset validation later, which is the honest way to test generalisation. |
| Real Netflix / Prime / YouTube streams | Two blockers. There is no studio master to compare against, and the delivered streams are DRM-protected (Widevine/PlayReady/FairPlay) with HDCP blocking capture. Circumventing that would be unlawful. See the README's framing section. |

## 3. Synthetic data in this repository

Two synthetic artifacts *are* committed, both small and both labelled:

| File | What it is | Why it exists |
| --- | --- | --- |
| `tests/fixtures/libvmaf_log.json` | a real libvmaf JSON log (ffmpeg 8.1.2 / libvmaf 3.2.0, 30 frames) | lets the log parser be tested in CI with no ffmpeg installed |
| `tests/fixtures/features_sample.csv` | 48 rows over 8 synthetic "contents", generated by `scripts/make_feature_fixture.py` | lets the training and evaluation code be tested offline |

`features_sample.csv` carries a `source` column whose every value is
`synthetic-fixture-not-a-result`, and a test asserts that. It exercises code
paths; it is never a finding, and no number in the README comes from it.

`tests/fixtures/*.mp4` are tiny generated clips (320x180, 2 s) from
`scripts/make_fixtures.py`.

## 4. Provenance and reproducibility

* Reference clips: `bash scripts/download_media.sh` (URLs and licences listed at
  the top of that script).
* Every measurement records the ffmpeg version, libvmaf version, VMAF model,
  pooling method and scaling filter used to produce it, in the `ctx_*` columns of
  the metrics CSV. A VMAF number without its model is not reproducible.
* Encoded outputs and metrics live under `data/`, which is gitignored and intended
  to be regenerated rather than shared.
