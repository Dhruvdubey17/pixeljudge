# PixelJudge in a container.
#
# The reason this file exists is narrow and important: the only genuinely awkward
# dependency in the project is an ffmpeg built with libvmaf. Everything else is a
# pip install. Debian's ffmpeg package has included libvmaf since bookworm, so the
# image is a plain apt install rather than a source build, and `pixeljudge doctor`
# is run at build time so a broken image fails here instead of three commands into
# someone's evening.
#
# Build:  docker build -t pixeljudge .
# Check:  docker run --rm pixeljudge doctor
# Use:    docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" \
#             pixeljudge encode --source big_buck_bunny.mp4

FROM python:3.12-slim-bookworm

# ffmpeg brings libvmaf, libx264, libx265, libvpx and SVT-AV1 with it on bookworm.
# libgl1/libglib2.0-0 are OpenCV's runtime shared libraries: even the headless
# wheel needs them present.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv resolves and installs from the lockfile, so the image gets the same versions
# the tests ran against.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependencies first, in their own layer, so editing source code does not
# reinstall numpy.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

# Fail the build if the environment is not actually usable. This is the one check
# worth spending build time on: an image whose ffmpeg lacks libvmaf looks fine
# until the first measurement.
RUN pixeljudge doctor

# The offline unit suite must pass inside the image too, which also proves the
# tests do not secretly depend on anything from the host.
RUN pytest -m "not integration" -q

VOLUME ["/app/data", "/app/reports", "/app/models"]

ENTRYPOINT ["pixeljudge"]
CMD ["doctor"]
