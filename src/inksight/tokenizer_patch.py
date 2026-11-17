"""Running the released graph without ``tensorflow-text`` (e.g. on Windows).

The InkSight SavedModel tokenizes its text input and output with three
SentencePiece ops - ``SentencepieceOp``, ``SentencepieceTokenizeOp`` and
``SentencepieceDetokenizeOp`` - that only the ``tensorflow-text`` package
registers, and that package has no Windows wheels after version 2.10.  Nothing
else in the graph needs it.

So when ``tensorflow-text`` is missing, a copy of the graph is rewritten once:

* every ``SentencepieceOp`` (which creates the tokenizer resource) becomes a
  string constant,
* the tokenize / detokenize ops become ``EagerPyFunc`` calls into the
  ``sentencepiece`` Python package - the same C++ library, available on every
  platform - using the tokenizer model embedded in the graph, with the same
  arguments (``nbest_size``, ``alpha``, ``add_bos``, ``add_eos``, ``reverse``)
  the original ops received.

The rewritten ``saved_model.pb`` is stored next to the original as
``<model folder>.sentencepiece/`` with the weights hard-linked (copied where
hard links are impossible), and reused afterwards.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .errors import BackendNotAvailableError, MissingDependencyError
from .paths import user_data_dir

logger = logging.getLogger("inksight")

#: The ops ``tensorflow-text`` would have to provide.
SENTENCEPIECE_OPS = frozenset({"SentencepieceOp", "SentencepieceTokenizeOp",
                               "SentencepieceDetokenizeOp"})

#: ``EagerPyFunc`` tokens under which the replacements are registered.
TOKENIZE_TOKEN = "inksight_sentencepiece_tokenize"
DETOKENIZE_TOKEN = "inksight_sentencepiece_detokenize"

#: Name of the tokenizer model written next to the rewritten graph.
SP_MODEL_FILE = "sentencepiece.model"

#: Bump when the rewrite changes, so stale rewritten copies are redone.
_PATCH_VERSION = "1"

_registered: List[Any] = []      # strong references: TF's registry only keeps weak ones


# --------------------------------------------------------------------------- inspection
def _read_saved_model(model_dir: Path) -> Any:
    from tensorflow.core.protobuf import saved_model_pb2

    proto = saved_model_pb2.SavedModel()
    proto.ParseFromString((model_dir / "saved_model.pb").read_bytes())
    return proto


def missing_ops(model_dir: "os.PathLike[str] | str") -> Set[str]:
    """Ops the graph uses that this TensorFlow does not provide."""
    from tensorflow.python.framework import op_def_registry

    graph = _read_saved_model(Path(model_dir)).meta_graphs[0].graph_def
    functions = {f.signature.name for f in graph.library.function}
    used = {n.op for n in graph.node}
    used.update(n.op for f in graph.library.function for n in f.node_def)
    return {op for op in used if op not in functions and op_def_registry.get(op) is None}


# --------------------------------------------------------------------------- rewrite
def patch_saved_model(model_dir: "os.PathLike[str] | str", out_dir: "os.PathLike[str] | str") -> Path:
    """Write a copy of ``model_dir`` whose SentencePiece ops call the Python package."""
    from tensorflow.core.framework import types_pb2

    model_dir, out_dir = Path(model_dir), Path(out_dir)
    proto = _read_saved_model(model_dir)
    graph = proto.meta_graphs[0].graph_def
    sp_model = b""

    for function in graph.library.function:
        rename: Dict[str, str] = {}
        for node in function.node_def:
            if node.op == "SentencepieceOp":
                sp_model = node.attr["model"].s or sp_model
                node.op = "Const"
                node.ClearField("attr")
                node.attr["dtype"].type = types_pb2.DT_STRING
                node.attr["value"].tensor.dtype = types_pb2.DT_STRING
                node.attr["value"].tensor.tensor_shape.SetInParent()
                node.attr["value"].tensor.string_val.append(b"sentencepiece")
                rename[f"{node.name}:handle:0"] = f"{node.name}:output:0"
            elif node.op == "SentencepieceTokenizeOp":
                out_type, splits = _attr_type(node, "out_type", types_pb2.DT_INT32), \
                    _attr_type(node, "Tsplits", types_pb2.DT_INT64)
                if out_type != types_pb2.DT_INT32:
                    raise BackendNotAvailableError("the graph tokenizes into strings; "
                                                   "this needs tensorflow-text")
                # handle, input, nbest_size, alpha, add_bos, add_eos, reverse
                _to_py_func(node, TOKENIZE_TOKEN,
                            [types_pb2.DT_STRING, types_pb2.DT_STRING, types_pb2.DT_INT32,
                             types_pb2.DT_FLOAT, types_pb2.DT_BOOL, types_pb2.DT_BOOL,
                             types_pb2.DT_BOOL],
                            [types_pb2.DT_INT32, splits])
                rename[f"{node.name}:output_values:0"] = f"{node.name}:output:0"
                rename[f"{node.name}:output_splits:0"] = f"{node.name}:output:1"
            elif node.op == "SentencepieceDetokenizeOp":
                splits = _attr_type(node, "Tsplits", types_pb2.DT_INT64)
                # handle, input_values, input_splits, add_bos, add_eos, reverse
                _to_py_func(node, DETOKENIZE_TOKEN,
                            [types_pb2.DT_STRING, types_pb2.DT_INT32, splits, types_pb2.DT_BOOL,
                             types_pb2.DT_BOOL, types_pb2.DT_BOOL],
                            [types_pb2.DT_STRING])
        if rename:
            for node in function.node_def:
                for index, name in enumerate(node.input):
                    if name in rename:
                        node.input[index] = rename[name]
            for key, value in list(function.ret.items()):
                if value in rename:
                    function.ret[key] = rename[value]

    if not sp_model:
        raise BackendNotAvailableError(f"{model_dir}: no SentencePiece model found in the graph")

    staging = out_dir.with_name(out_dir.name + f".{os.getpid()}.tmp")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "variables").mkdir(parents=True)
    (staging / "saved_model.pb").write_bytes(proto.SerializeToString())
    (staging / SP_MODEL_FILE).write_bytes(sp_model)
    for extra in ("fingerprint.pb",):
        if (model_dir / extra).is_file():
            shutil.copy2(model_dir / extra, staging / extra)
    for source in (model_dir / "variables").iterdir():
        _link_or_copy(source, staging / "variables" / source.name)
    (staging / "PATCHED").write_text(_stamp(model_dir), encoding="utf-8")
    shutil.rmtree(out_dir, ignore_errors=True)
    staging.replace(out_dir)
    return out_dir


def _attr_type(node: Any, name: str, default: int) -> int:
    return node.attr[name].type if name in node.attr else default


def _to_py_func(node: Any, token: str, tin: List[int], tout: List[int]) -> None:
    data = [name for name in node.input if not name.startswith("^")]
    if len(data) != len(tin):
        raise BackendNotAvailableError(f"unexpected inputs on {node.op} {node.name!r}: {data}")
    node.op = "EagerPyFunc"
    node.ClearField("attr")
    node.attr["token"].s = token.encode()
    node.attr["is_async"].b = False
    node.attr["Tin"].list.type.extend(tin)
    node.attr["Tout"].list.type.extend(tout)


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _stamp(model_dir: Path) -> str:
    stat = (model_dir / "saved_model.pb").stat()
    return f"{_PATCH_VERSION} {stat.st_size} {int(stat.st_mtime)}"


def patched_model_dir(model_dir: "os.PathLike[str] | str") -> Path:
    """The rewritten copy of ``model_dir``, creating it on first use.

    It lives next to the original (``<folder>.sentencepiece``) or, if that
    location is read-only, under the per-user data directory.
    """
    model_dir = Path(model_dir).resolve()
    candidates = [model_dir.with_name(model_dir.name + ".sentencepiece"),
                  user_data_dir() / "patched" /
                  hashlib.sha256(str(model_dir).encode()).hexdigest()[:16]]
    stamp = _stamp(model_dir)
    for candidate in candidates:
        marker = candidate / "PATCHED"
        if marker.is_file() and marker.read_text(encoding="utf-8") == stamp:
            return candidate
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            logger.info("inksight: rewriting the graph's SentencePiece ops into %s", candidate)
            return patch_saved_model(model_dir, candidate)
        except OSError as exc:
            last_error = exc
    raise BackendNotAvailableError(f"could not write a patched copy of {model_dir}: {last_error}")


# --------------------------------------------------------------------------- runtime
def register_sentencepiece(sp_model: bytes) -> None:
    """Register the Python implementations of the two rewritten ops."""
    try:
        import sentencepiece
    except ImportError as exc:
        raise MissingDependencyError(
            "running the model without tensorflow-text needs the 'sentencepiece' package: "
            "pip install sentencepiece") from exc
    import numpy as np
    import tensorflow as tf
    from tensorflow.python.ops import script_ops

    processor = sentencepiece.SentencePieceProcessor(model_proto=sp_model)

    def tokenize(handle, strings, nbest_size, alpha, add_bos, add_eos, reverse):
        nbest, sample = int(nbest_size.numpy()), float(alpha.numpy())
        options = dict(add_bos=bool(add_bos.numpy()), add_eos=bool(add_eos.numpy()),
                       reverse=bool(reverse.numpy()))
        if nbest not in (0, 1):
            options.update(enable_sampling=True, nbest_size=nbest, alpha=sample)
        pieces = [processor.encode(s.decode("utf-8"), **options) for s in strings.numpy().reshape(-1)]
        splits = np.cumsum([0] + [len(p) for p in pieces]).astype(np.int64)
        values = np.fromiter((i for p in pieces for i in p), dtype=np.int32, count=int(splits[-1]))
        return tf.constant(values), tf.constant(splits)

    def detokenize(handle, values, splits, add_bos, add_eos, reverse):
        ids, bounds = values.numpy().tolist(), splits.numpy().tolist()
        flip = bool(reverse.numpy())
        texts = []
        for start, stop in zip(bounds[:-1], bounds[1:]):
            row = ids[start:stop]
            texts.append(processor.decode(row[::-1] if flip else row).encode("utf-8"))
        return tf.constant(texts, dtype=tf.string)

    _registered.clear()
    for token, function, outputs in ((TOKENIZE_TOKEN, tokenize, [tf.int32, tf.int64]),
                                     (DETOKENIZE_TOKEN, detokenize, [tf.string])):
        eager = script_ops.EagerFunc(function, outputs, False)
        _registered.append(eager)
        script_ops._py_funcs._funcs[token] = eager


def load_patched(tf: Any, model_dir: "os.PathLike[str] | str") -> Tuple[Any, Path]:
    """``tf.saved_model.load`` on the rewritten copy, with the replacements registered."""
    patched = patched_model_dir(model_dir)
    register_sentencepiece((patched / SP_MODEL_FILE).read_bytes())
    return tf.saved_model.load(str(patched)), patched
