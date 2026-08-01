"""PixelJudge: a full-reference video quality analysis engine.

The package is split by responsibility so each piece can be tested on its own:

* ``io``        - talking to the ffmpeg/ffprobe binaries
* ``encode``    - turning a master clip into distorted ladder rungs
* ``ladder``    - which rungs to produce (fixed table or per-title convex hull)
* ``metrics``   - PSNR/SSIM/MS-SSIM/VMAF measurement and BD-Rate comparison
* ``artifacts`` - OpenCV proxies for blur, blocking and banding
* ``model``     - regression from objective metrics to subjective MOS
* ``viz``       - plots and tables for the report
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
