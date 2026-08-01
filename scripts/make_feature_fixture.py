#!/usr/bin/env python
"""Write the small synthetic feature table used by the offline model tests.

This exists so the training and evaluation code can be tested in CI, where there
is no ffmpeg, no dataset and no time to encode anything. It is emphatically **not**
a result: nothing in the README is computed from it. The real numbers come from the
Netflix Public Dataset, and the file this writes says so in a comment column.

The generator is deliberately simple and deliberately *not* linear:

* Eight synthetic "contents", six distortion levels each (48 rows), mirroring the
  shape of a subjective dataset where every source clip appears at several
  compression levels. That shape is what makes content-grouped cross-validation
  necessary in the first place.
* VMAF is the primary driver of opinion, through a saturating curve: below about
  50 everything looks bad, above about 90 nobody can tell the difference. That
  saturation is exactly what the VQEG logistic fit is there to absorb.
* Each content gets a small fixed opinion offset, which is the synthetic stand-in
  for "some material is simply more forgiving". A model that memorises those
  offsets instead of learning quality is leaking, and grouped CV catches it.
* PSNR, SSIM and MS-SSIM are derived from VMAF with their own noise, so they are
  correlated with the label but individually weaker: the situation a fusion model
  is supposed to exploit.

Usage:
    uv run python scripts/make_feature_fixture.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_CONTENTS = 8
LEVELS = (98.0, 92.0, 84.0, 72.0, 55.0, 32.0)  # nominal VMAF per distortion level
SEED = 20260730


def build(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    content_offset = rng.normal(0.0, 0.18, N_CONTENTS)  # per-content opinion bias
    rows: list[dict[str, object]] = []

    for content_index in range(N_CONTENTS):
        for level_index, nominal in enumerate(LEVELS):
            vmaf = float(np.clip(nominal + rng.normal(0, 2.0), 0, 100))
            # PSNR tracks VMAF loosely; the spread is what makes it a weak predictor.
            psnr = float(np.clip(24.0 + 0.24 * vmaf + rng.normal(0, 1.4), 15, 60))
            ssim = float(np.clip(1.0 - np.exp(-vmaf / 26.0) * 0.42 + rng.normal(0, 0.004), 0, 1))
            ms_ssim = float(np.clip(ssim + 0.004 + rng.normal(0, 0.002), 0, 1))
            # Opinion saturates at both ends of the VMAF range.
            latent = 1.0 + 4.0 / (1.0 + np.exp(-(vmaf - 68.0) / 9.0))
            mos = float(np.clip(latent + content_offset[content_index] + rng.normal(0, 0.12), 1, 5))
            rows.append(
                {
                    "content": f"synthetic_{content_index:02d}",
                    "distorted": f"synthetic_{content_index:02d}_level{level_index}.mp4",
                    "vmaf": round(vmaf, 3),
                    "psnr_y": round(psnr, 3),
                    "float_ssim": round(ssim, 5),
                    "float_ms_ssim": round(ms_ssim, 5),
                    "banding_max": round(float(abs(rng.normal(0, 8)) + (100 - vmaf) * 0.4), 3),
                    "mos": round(mos, 3),
                    "source": "synthetic-fixture-not-a-result",
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures/features_sample.csv"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    table = build(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(table)} rows, {table['content'].nunique()} contents)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
