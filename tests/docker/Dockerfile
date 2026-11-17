# Runs the test suite on Linux, natively (local runner: TensorFlow with
# tensorflow-text, docTR, PaddleOCR).  Build from the repository root:
#
#   docker build -f tests/docker/Dockerfile -t inksight-tests .
#   docker run --rm inksight-tests                 # = python -m pytest -v
FROM python:3.12-slim

# libgl1 / libglib2.0-0: OpenCV (docTR, PaddleOCR)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /repo
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip install ".[tf,doctr,paddle]" pytest
COPY tests ./tests

CMD ["python", "-m", "pytest", "-v", "-p", "no:cacheprovider"]
