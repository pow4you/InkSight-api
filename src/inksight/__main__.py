"""Command line: ``python -m inksight <command>`` (installed as the ``inksight`` script).

Commands::

    inksight vectorize page.png -m page -o page.svg
    inksight words page.png -o page.words.json
    inksight build --docker | --local | --bundle model/small-p-cpu
    inksight info
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__, available_methods, get_vectorizer_class
from .build import build_docker_image, bundle_model, prepare_local
from .bundle import bundle_info
from .drawing import VectorDrawing
from .errors import InksightError, VectorizationError
from .prompts import InferenceMode
from .runners import DockerRunner, LocalRunner, WslRunner
from .words import available_detectors, get_detector
from .worker import TOOL


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return args.handler(args)
    except VectorizationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (InksightError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


# --------------------------------------------------------------------------- commands
def _cmd_vectorize(args: argparse.Namespace) -> int:
    options = {"runner": args.runner, "mode": args.mode, "batch_size": args.batch_size,
               "invert": args.invert, "timeout": args.timeout, "workdir": args.workdir}
    if args.no_fallback:
        options["fallback_mode"] = None
    inputs = {}
    if args.method_class.name == "page":
        options["word_detector"] = args.detector
        if args.words:
            inputs["words"] = args.words
        if args.text:
            print("note: --text is only used by the word method; pages use the detector's text",
                  file=sys.stderr)
    else:
        if args.text:
            inputs["text"] = args.text
        if args.words:
            print("note: --words is only used by the page method", file=sys.stderr)

    vectorizer = args.method_class(**options)
    if len(args.image) == 1:
        drawing = vectorizer.vectorize(args.image[0], **inputs)
        _write(drawing, Path(args.output) if args.output else Path(args.image[0]).with_suffix(".svg"))
        return 0
    # Several images: one model call for all of them (words batched across images).
    if inputs.get("words"):
        raise ValueError("--words applies to a single image")
    out_dir = Path(args.output) if args.output else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    drawings = vectorizer.vectorize_many(args.image, [dict(inputs) for _ in args.image])
    for image, drawing in zip(args.image, drawings):
        default = Path(image).with_suffix(".svg")
        _write(drawing, out_dir / default.name if out_dir else default)
    return 0


def _write(drawing: "VectorDrawing", output: Path) -> None:
    if output.suffix.lower() == ".json":
        output.write_text(json.dumps({"width": drawing.width, "height": drawing.height,
                                      "text": drawing.text,
                                      "strokes": [s.round(3).tolist() for s in drawing]}),
                          encoding="utf-8")
    else:
        drawing.save(output)
    summary = f"{output}: {len(drawing)} strokes, {drawing.num_points} points"
    if "elapsed" in drawing.metadata:
        summary += (f" ({drawing.metadata['elapsed']:.1f} s via "
                    f"{drawing.metadata.get('runner', '?')} runner)")
    if drawing.text:
        summary += f"\ntext: {drawing.text}"
    print(summary)


def _cmd_words(args: argparse.Namespace) -> int:
    detector = get_detector(args.detector)
    words = detector.detect(args.image)
    output = Path(args.output) if args.output else Path(args.image).with_suffix(".words.json")
    words.save(output)
    print(f"{output}: {words!r}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    if args.bundle:
        manifest = bundle_model(args.bundle)
        print(f"bundled {manifest['name']}: {len(manifest['parts'])} parts, "
              f"{manifest['size'] / 1e6:.0f} MB")
    elif args.local:
        path = prepare_local(model_dir=args.model_dir, tokenizer=args.tokenizer)
        print(f"local model ready: {path}")
    else:
        runner = DockerRunner(gpus="all" if args.gpu else None)
        image = build_docker_image(runner.image, build_args=runner.build_args, quiet=False)
        print(f"backend image ready: {image}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    print(f"inksight {__version__}")
    print("methods:")
    for name in available_methods():
        cls = get_vectorizer_class(name)
        print(f"  {name:<6} {cls.title}")
    print(f"word detectors: {', '.join(available_detectors())}")

    info = bundle_info()
    print("bundled model:")
    print(f"  {info['name']} ({info['size'] / 1e6:.0f} MB compressed, sha256 {info['sha256'][:12]})"
          if info else "  missing")
    local, docker = LocalRunner(), DockerRunner()
    print("local runner:")
    print(f"  {TOOL:<14} {local.find(TOOL) or 'not available (TensorFlow not installed)'}")
    print("docker runner:")
    if not docker.daemon_running():
        print("  Docker is not reachable")
    else:
        state = "built" if docker.image_exists() else "not built yet (built on first use)"
        print(f"  image {docker.image}: {state}")
    if sys.platform == "win32":
        wsl = WslRunner()
        print("wsl / rocm runner:")
        print(f"  ready: {wsl.describe()}" if wsl.is_available()
              else f"  not available ({wsl.python} with TensorFlow not found in any WSL distro; "
                   "see scripts/setup-rocm-wsl.sh)")
    return 0


# --------------------------------------------------------------------------- parser
def _method(name: str) -> type:
    try:
        return get_vectorizer_class(name)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inksight", description="Handwriting images to pen "
                                     "strokes with the InkSight model (Mitrevski et al., 2025).")
    parser.add_argument("--version", action="version", version=f"inksight {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    vec = commands.add_parser("vectorize", help="convert an image into strokes (SVG or JSON)")
    vec.add_argument("image", nargs="+", help="input image(s) (PNG, JPEG, ...); several are "
                                                 "derendered in one model call")
    vec.add_argument("-m", "--method", dest="method_class", type=_method,
                     default=get_vectorizer_class("page"),
                     help=f"one of: {', '.join(available_methods())} (default: page)")
    vec.add_argument("-o", "--output", help="output .svg or .json (default: image name with .svg); "
                                            "with several images, an output directory")
    vec.add_argument("--mode", default="r+d", choices=[m.alias for m in InferenceMode],
                     help="r+d: the model reads the word itself (default); vanilla: no text "
                          "involved; d+t: derender the given / detected text")
    vec.add_argument("--no-fallback", action="store_true",
                     help="do not retry empty words with the vanilla prompt")
    vec.add_argument("--text", help="word: the word's text (for --mode d+t)")
    vec.add_argument("-w", "--words", help="page: use this word-box JSON instead of the detector")
    vec.add_argument("-d", "--detector", default="doctr",
                     help=f"page: word detector, one of: {', '.join(available_detectors())} "
                          "(default: doctr)")
    vec.add_argument("--batch-size", type=int, default=32, help="words per forward pass (default: 32)")
    vec.add_argument("--invert", action="store_true", help="writing is light on dark")
    vec.add_argument("--runner", choices=["auto", "local", "docker", "docker-gpu", "docker-server",
                                                "docker-gpu-server", "wsl", "rocm"], default="auto",
                     help="where the model runs (default: auto)")
    vec.add_argument("--timeout", type=float, help="abort the model after this many seconds (docker)")
    vec.add_argument("--workdir", help="keep intermediate files and the model log here")
    vec.set_defaults(handler=_cmd_vectorize)

    words = commands.add_parser("words", help="detect word boxes (page stage 1) into a JSON file")
    words.add_argument("image")
    words.add_argument("-o", "--output", help="output JSON (default: <image>.words.json)")
    words.add_argument("-d", "--detector", default="doctr",
                       help=f"one of: {', '.join(available_detectors())} (default: doctr)")
    words.set_defaults(handler=_cmd_words)

    build = commands.add_parser("build", help="prepare a backend, or (re)create the model bundle")
    what = build.add_mutually_exclusive_group()
    what.add_argument("--docker", action="store_true", help="build the Docker image (default)")
    what.add_argument("--local", action="store_true",
                      help="unpack the bundled model (and rewrite its tokenizer if needed) now")
    what.add_argument("--bundle", metavar="SAVED_MODEL_DIR",
                      help="compress a SavedModel folder into the package")
    build.add_argument("--gpu", action="store_true",
                       help="--docker: build the CUDA image used by --runner docker-gpu")
    build.add_argument("--model-dir", help="--local: use this SavedModel instead of the bundle")
    build.add_argument("--tokenizer", choices=["auto", "tensorflow-text", "sentencepiece"],
                       default="auto", help="--local: tokenizer route (default: auto)")
    build.set_defaults(handler=_cmd_build)

    info = commands.add_parser("info", help="show methods, the model and which backends are ready")
    info.set_defaults(handler=_cmd_info)
    return parser


if __name__ == "__main__":
    sys.exit(main())
