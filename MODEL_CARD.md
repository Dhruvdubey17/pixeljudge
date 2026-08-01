# Model card: PixelJudge MOS regressor

## What it is

A small supervised regressor that maps objective full-reference quality metrics
onto a subjective opinion score:

```
(VMAF, PSNR-Y, SSIM, MS-SSIM)  ->  predicted subjective score
```

Each metric enters twice, pooled by arithmetic mean and by libvmaf's harmonic mean,
so the model sees both "how good on average" and "how bad at its worst".

The selected model is **RandomForest**, chosen by cross-validated SROCC. What it
predicts is the LIVE-Netflix `final_subj_score`: a Z-scored retrospective quality
rating, not a 1-5 MOS. See "Data" below.

Three candidates are trained and compared:

| Candidate | Why it is in the list |
| --- | --- |
| `svr_rbf` | Support Vector Regression with an RBF kernel. The headline choice, because VMAF itself fuses its elementary features with an SVM: this is the same idea one level up. |
| `random_forest` | A nonlinear alternative that also yields feature importances. |
| `ridge` | A transparent linear floor. If a linear model does as well, the nonlinear ones have not earned their complexity. |

Each is a scikit-learn `Pipeline` of `StandardScaler` followed by the estimator, so
the scaler is fitted on training folds only.

## Intended use

* Comparing quality metrics against each other on a labelled dataset.
* Showing whether *fusing* metrics beats the best single metric, and by how much.
* Teaching the evaluation protocol: logistic fit, then PLCC/SROCC/KROCC/RMSE.

## Out of scope

* **Not a replacement for VMAF.** It is trained on far less data than VMAF was, on
  a narrower distortion range, and it consumes VMAF as an input.
* **Not a no-reference metric.** Every input needs the pristine original.
* **Not validated across datasets.** Trained and evaluated on one dataset, so
  nothing here supports a claim about content or distortions it has not seen. No
  generalisation is claimed beyond the reported cross-validation.
* **Not a QoE model, despite being trained on a QoE database.** It is trained only
  on LIVE-Netflix's **continuous-playback** conditions. Its inputs are compression
  metrics, which cannot observe a rebuffering event - a frozen frame is not a
  distorted frame. Applied to a session containing stalls it will predict as though
  the stalls did not happen, and it will be wrong in the optimistic direction.
  Bitrate adaptation *is* within scope: the metrics see it as a dip in per-frame
  quality.
* **Not calibrated to a viewing condition other than the one it was trained on.**
  LIVE-Netflix ratings were collected on a **mobile device**, which is not the
  1080p-on-a-TV model (`vmaf_v0.6.1`) used elsewhere in this project.
* **Not a substitute for looking at the video.** The banding result in the README
  is the standing demonstration of why.

## Training and evaluation protocol

**Nested, content-grouped cross-validation.**

* Outer loop: `GroupKFold` over source content, producing one out-of-fold
  prediction per clip.
* Inner loop: another `GroupKFold` inside each outer training split, driving the
  hyperparameter grid search.

Why grouped: a subjective dataset contains each source clip many times, once per
distortion level. A random row split puts the same content in train and test, and a
model can then score well by *recognising the content* rather than judging quality.
The reported correlation would be inflated and the model would fail on anything new.
Grouping by source clip makes that impossible. `tests/test_train.py` includes a
direct test: features of pure noise, labels determined entirely by content identity,
and the assertion is that grouped CV *cannot* find a strong correlation.

Why nested: hyperparameters chosen on the rows they are then scored on are chosen
partly by luck, and the score inherits that luck.

**Reported metrics**, following VQEG convention:

| Metric | What it measures |
| --- | --- |
| PLCC | accuracy, after a 5-parameter logistic maps predictions onto the MOS scale |
| SROCC | monotonicity; rank-based, so the mapping cannot change it |
| KROCC | a stricter ordinal agreement (concordant vs discordant pairs) |
| RMSE | typical error size, in MOS units, after the logistic fit |

Model selection uses **SROCC**, because ranking clips correctly is the property a
quality metric is actually for. Note the consequence: a model can be selected on
SROCC while another has a better RMSE, and both numbers are always shown.

**The baselines are given the advantage.** Each single-metric baseline (VMAF alone,
PSNR alone, ...) gets a logistic mapping fitted on each fold's training rows, which
is extra flexibility. The fused model's predictions are used as-is, since it is
trained to output MOS directly. If the model wins, it wins against a favoured
opponent.

## Data

Two datasets, one of which is trained on today.

**LIVE-Netflix Video QoE Database (VideoATLAS release)** - the dataset the reported
numbers come from. 112 videos: 14 source contents crossed with 8 streaming playout
patterns, rated by 55+ subjects **on a mobile device**. The release ships
pre-computed per-frame quality vectors (VMAF, PSNR, SSIM, MS-SSIM, ST-RRED, NIQE)
alongside the subjective scores, which is why it can be used without a media access
request.

*Scope actually trained on:* the **56 continuous-playback clips** (`ns == 0`), over
all 14 contents. This is the central caveat of this model card and is expanded under
"Out of scope" below.

*Label:* `final_subj_score`, kept under the honest name `subj_score`. It is **not a
MOS on a 1-5 scale** - it is Z-scored per viewing session and per subject, running
-1.60 to 1.58 with mean 0.004. Correlations are unaffected by that; **RMSE is in
Z-score units and is comparable only within the reported table.**

**Netflix Public Dataset (NFLX-P)** - pending, intended as a future third
confirmation and as the basis for a cross-dataset generalisation test. 70 distorted
clips over 9 source contents with panel DMOS. Its labels are parsed and in hand; its
videos are not. See `DATA_CARD.md` for provenance and licensing on both.

## Reported performance

LIVE-Netflix, compression-only scope, 56 clips / 14 contents, nested content-grouped
5-fold CV. Features are each metric pooled by arithmetic mean and by libvmaf's
harmonic mean (8 features).

| predictor | PLCC | SROCC | KROCC | RMSE (Z-units) |
|---|---|---|---|---|
| **RandomForest (selected)** | **0.8766** | **0.8353** | **0.6589** | **0.4146** |
| SVR-RBF | 0.7793 | 0.8253 | 0.6545 | 0.5517 |
| Ridge | 0.6979 | 0.7883 | 0.6169 | 0.6483 |
| VMAF alone | 0.7479 | 0.6876 | 0.5052 | 0.5659 |
| SSIM alone | 0.5117 | 0.5314 | 0.4182 | 0.8162 |
| PSNR alone | 0.4917 | 0.5094 | 0.3753 | 0.7754 |

**Read the fold spread before quoting the margin.** Per fold, RandomForest averages
SROCC 0.887 +/- 0.086 and VMAF-alone 0.788 +/- 0.139. The 0.099 gap between the means
is smaller than either standard deviation. The defensible claim is that the fused
model ranks ahead of VMAF-alone consistently across folds, **not** that the size of
that lead is well determined. With 14 contents a fold is 2-3 contents and a single
unusual content moves the number substantially.

Random-forest feature importances: `vmaf_hmean` 0.274, `ms_ssim_hmean` 0.173,
`ssim_hmean` 0.159, `ms_ssim_mean` 0.146, `ssim_mean` 0.139, `vmaf_mean` 0.085,
`psnr_hmean` 0.012, `psnr_mean` 0.012.

Reproduce with:

```bash
uv run pixeljudge mos load     # read the release, pool to clip features
uv run pixeljudge mos train    # nested content-grouped CV + baselines
uv run pixeljudge mos table    # correlation table, fold table, figures
```

## Validation status

**What is verified.** Our pooling reproduces the release authors' own pooled scalars
(`VMAF_mean`, `PSNR_mean`) to a worst-case absolute difference of **1.42e-13** across
all 112 videos - an independent implementation of the same operation agreeing to
floating-point noise. Pinned as an integration test that skips when the release is
absent.

**What is not verified.** That PixelJudge's own libvmaf measurement code reproduces
the authors' per-frame vectors. This requires the LIVE-Netflix source and distorted
videos, which are distributed by request form and emailed password (and 11 of the 14
contents are proprietary Netflix material). The check is implemented and tested -
`compare_feature_tables`, exposed as `pixeljudge mos validate --ours <csv>` - and
reports Spearman separately from the raw differences, so a constant offset between
two VMAF builds is distinguished from a genuine reordering. Expect Spearman > ~0.95
on VMAF; anything materially lower is a real discrepancy to investigate, not noise.

So: **the reported numbers validate the modelling, not the measurement.**

## Expectation, and how it turned out

Going in: published results put VMAF well ahead of PSNR and SSIM as a single
predictor, and fusing four correlated metrics on ~56 clips should be expected to
match it or beat it slightly, not transform it. If the fused model did **not** beat
VMAF alone, that was to go in the README as the finding - a well-designed single
metric is hard to beat with a handful of features and a small sample - and the CLI
prints exactly that message when it happens, so the honest outcome is the default.

What happened is the mild version of the optimistic case, and it lands about where
that expectation put it. VMAF was indeed the strongest single metric (SROCC 0.688 vs
0.531 and 0.509). Fusion did beat it (0.835), but by a margin no larger than the
fold-to-fold noise, which is "slightly", not "transformed".

The interesting part was *why* fusion helps, and it is more specific than "more
features are better". In the fold where SSIM-alone reached PLCC -0.355, one content's
worst condition scored a **higher** absolute SSIM than another content's best, while
being rated far lower. SSIM ranks conditions correctly within a content; its absolute
value is not comparable across contents, because it depends on how much detail the
source has. VMAF ranked the same pair correctly - content-independence is what it was
trained for. So the fused model's advantage comes from learning a content-independent
combination that a single scale-sensitive metric cannot express, which is also why
harmonic-mean VMAF carries the largest feature importance and both PSNR features sit
near 0.01.

## Ethical and practical notes

* MOS is an average of human opinions gathered under specific viewing conditions.
  A model trained on it inherits those conditions: screen, distance, and the panel
  itself. A phone viewer is a different question, which is why VMAF ships a
  separate phone model.
* Quality metrics get used to set bitrates, and bitrates cost money and bandwidth.
  A metric that is wrong in a systematic direction (for instance blind to banding)
  pushes those decisions in that direction at scale. That is the practical reason
  the artifact detectors exist.
