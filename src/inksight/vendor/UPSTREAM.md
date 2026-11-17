# Upstream material

## `model/` - the InkSight-Small-p checkpoint

| | |
|-|-|
| What | TensorFlow SavedModel `small-p-cpu` (CPU export of InkSight-Small-p), unmodified |
| Source | <https://storage.googleapis.com/derendering_model/small-p-cpu.zip> (referenced by the official notebook); the same model is on <https://huggingface.co/Derendering/InkSight-Small-p> |
| Paper | B. Mitrevski et al., *InkSight: Offline-to-Online Handwriting Conversion by Teaching Vision-Language Models to Read and Write*, TMLR 2025 |
| Code | <https://github.com/google-research/inksight> (`colab.ipynb`) |
| Licence | Apache-2.0, **research and non-commercial use only** (see `NOTICE`) |
| Size | 692 MB unpacked: `saved_model.pb` (25 MB graph) + 667 MB of bfloat16 weights |

Stored as `small-p-cpu.tar.xz.000` ... `.004`: one xz-compressed tar archive
(461 MB) split into parts below 100 MB; `MANIFEST.json` lists the parts with
their SHA-256 and the hash of the whole archive.  `inksight.bundle` unpacks it
on first use; `python -m inksight build --bundle <folder>` recreates it.

The graph's input is a JPEG-encoded 224 x 224 image plus a text prompt; its
output is text with `<ink_token_N>` markers.  Tokenization happens inside the
graph with three SentencePiece ops from `tensorflow-text`.

## `examples/`

`word.png` and `page.png` were generated for this package with Pillow's bundled
default font (Aileron, SIL Open Font License); they are not upstream files.
