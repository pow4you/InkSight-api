# InkSight in a box: for Windows (where tensorflow-text has no wheels) or
# anywhere you don't want TensorFlow in your own environment. CPU only.
#
#   docker build -t inksight .
#   docker run --rm -v inksight-cache:/cache -v "${PWD}:/work" inksight word.jpg -o word.json
FROM python:3.12-slim

# tesseract-ocr: Tesseract segmenter   ffmpeg: mp4 animations
# libgl1 / libglib2.0-0: needed by OpenCV, which docTR imports
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/inksight-api
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
RUN pip install --no-cache-dir ".[all]"

# Model and docTR weights are downloaded on first run; keep them in a volume.
ENV INKSIGHT_CACHE=/cache/inksight \
    HF_HOME=/cache/huggingface \
    DOCTR_CACHE_DIR=/cache/doctr \
    MPLBACKEND=Agg \
    TF_CPP_MIN_LOG_LEVEL=2
VOLUME /cache

# Mount the folder with your images here; relative paths resolve against it.
WORKDIR /work
ENTRYPOINT ["inksight"]
CMD ["--help"]
