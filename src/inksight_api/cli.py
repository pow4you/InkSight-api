"""Command line entry point: image in, stroke paths out.

    inksight word.jpg                       # JSON on stdout
    inksight page.jpg --page -o page.svg    # full page, written as SVG
    inksight word.jpg --mode d+t --text hello -o word.inkml

It exists mainly so the package is usable from a container (see the
Dockerfile), where importing it from your own Python is not an option.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Sequence

from .core.exceptions import InkSightError
from .core.prompts import InferenceMode
from .facade import InkSight
from .io import JSONWriter, ink_writers, write_ink


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inksight", description="Convert a photo of handwriting into stroke paths.")
    parser.add_argument("image", help="path to the input image")
    parser.add_argument("--page", action="store_true",
                        help="treat the image as a full page (segment into words first)")
    parser.add_argument("--mode", default="vanilla", choices=[m.alias for m in InferenceMode],
                        help="vanilla: no text involved (default); r+d: the model reads the word "
                             "itself; d+t: use the given / OCR text")
    parser.add_argument("--text", help="the word's text (only for --mode d+t on a single word)")
    parser.add_argument("--segmenter", default="doctr",
                        help="word finder for --page: doctr (default) or tesseract")
    parser.add_argument("-o", "--output",
                        help=f"output file; format by suffix ({', '.join(ink_writers.names())}). "
                             "Without it, JSON is printed to stdout")
    parser.add_argument("--figure", help="also save the 3-panel result figure here (--page only)")
    parser.add_argument("--model", help="checkpoint: a .zip path or URL, a SavedModel folder, or a "
                                        "Hugging Face repo id (default: the authors' small-p-cpu.zip)")
    parser.add_argument("--batch-size", type=int, default=32, help="words per model call")
    return parser


def main(argv: Sequence[str] | None = None, *,
         kit_factory: Callable[..., InkSight] = InkSight.from_pretrained) -> int:
    """Run the CLI. ``kit_factory`` is injectable so tests need no model."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.text and args.page:
        parser.error("--text only applies to a single word; with --page the segmenter's OCR text is used")
    if args.figure and not args.page:
        parser.error("--figure needs --page")

    try:
        kit = kit_factory(args.model, segmenter=args.segmenter, batch_size=args.batch_size)
        if args.page:
            result = kit.convert_page(args.image, mode=args.mode)
        else:
            result = kit.convert_word(args.image, mode=args.mode, text=args.text)

        if args.output:
            write_ink(result.ink, args.output)
        else:
            print(JSONWriter().to_string(result.ink))
        if args.figure:
            from .visualization import plot_page_result
            plot_page_result(result, save_to=args.figure)
    except (InkSightError, OSError) as exc:
        print(f"inksight: {exc}", file=sys.stderr)
        return 1

    print(f"{len(result.ink)} strokes from {len(result.words)} word(s) "
          f"in {result.elapsed_seconds:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
