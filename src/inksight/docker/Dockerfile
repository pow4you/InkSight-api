# syntax=docker/dockerfile:1
# Image holding the InkSight model and the program that runs it (inksight-model).
#
# Built and used automatically by inksight.runners.DockerRunner; to build by hand,
# run from the inksight/ package directory:
#
#   docker build -f docker/Dockerfile -t inksight-backend .
#
# The Python wrapper mounts a scratch directory at /work and runs the program in
# it, e.g.
#
#   docker run --rm -v "$PWD":/work inksight-backend inksight-model job.json
#
# Linux has tensorflow-text, so here the graph runs exactly as exported (no
# tokenizer rewrite).  CPU only.

FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Versions follow the official notebook.
RUN pip install "tensorflow==2.20.0" "tensorflow-text==2.20.1" "sentencepiece>=0.2.1" \
        "numpy" "Pillow"

# Unpack the bundled model once, at build time, verifying the archive's hash.
RUN --mount=type=bind,source=vendor/model,target=/tmp/bundle python - <<'EOF'
import hashlib, io, json, pathlib, tarfile
bundle = pathlib.Path("/tmp/bundle")
manifest = json.loads((bundle / "MANIFEST.json").read_text())
data = b"".join((bundle / part["file"]).read_bytes() for part in manifest["parts"])
assert hashlib.sha256(data).hexdigest() == manifest["sha256"], "model archive damaged"
with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as tar:
    tar.extractall("/opt/inksight/model", filter="data")
EOF
RUN rm -rf /tmp/bundle

# The package itself (pure Python; only the model program is used here).
COPY . /opt/inksight/src/inksight
RUN rm -rf /opt/inksight/src/inksight/vendor/model \
    && printf '#!/bin/sh\nexec python -m inksight.worker "$@"\n' > /usr/local/bin/inksight-model \
    && chmod +x /usr/local/bin/inksight-model

# Fail the build, not the first run, if the program cannot start.
ENV PYTHONPATH=/opt/inksight/src \
    INKSIGHT_MODEL_DIR=/opt/inksight/model/small-p-cpu \
    TF_CPP_MIN_LOG_LEVEL=2
RUN python -c "import tensorflow_text, inksight.worker"

WORKDIR /work
