# Container for the pip install path. Everything except pose extraction on
# real video runs in here: tests, the synthetic smoke test, ONNX export and
# verification, and training once the data is mounted in.
#
# Build:
#   docker build -t pif .
#
# Run the tests:
#   docker run --rm pif pytest tests -q
#
# Run the synthetic smoke test on the GPU (needs the NVIDIA container toolkit
# on the host; see the README troubleshooting section):
#   docker run --rm --gpus all pif python -m scripts.smoke_test
#
# Run with the annotations and the data mounted from the host:
#   docker run --rm --gpus all \
#     -v "$PWD/external:/app/external:ro" \
#     -v "$PWD/data:/app/data" \
#     -v "$PWD/artifacts:/app/artifacts" \
#     pif python -m scripts.parse_annotations --dataset jaad --limit 20

# A plain python base, not an nvidia/cuda one, on purpose: the torch cu121
# wheel ships its own CUDA runtime libraries, so the only thing the image
# needs from the host is the NVIDIA driver, which --gpus all provides. That
# keeps the image around 3 GB instead of 8 GB.
FROM python:3.11-slim

# libglib2 and libgomp are what opencv-python-headless and mediapipe link
# against at import time. Without them `import cv2` fails with a missing .so
# that is not obviously related to opencv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch first, from the PyTorch CUDA index, so the CPU-only wheel on PyPI does
# not get picked up instead. Its own layer so that editing the source does not
# re-download two gigabytes.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch>=2.2 \
        --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests

RUN pip install --no-cache-dir -e .

# Not baked into the image: external/ (the JAAD and PIE reference repos, ~620
# MB, fetched by scripts/fetch_external.sh) and data/ (the video clips and the
# pose cache). Mount both at run time.

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "scripts.smoke_test"]
