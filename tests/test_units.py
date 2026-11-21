"""Unit tests that need neither TensorFlow, nor Docker, nor an OCR engine."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import inksight
from inksight import (InferenceMode, InvalidImageError, ModelJob, ModelVectorizer, Runner,
                      RunResult, VectorDrawing, VectorizationError, Vectorizer, WordDetector, Words)
from inksight import bundle, worker
from inksight.errors import BackendBuildError
from inksight.images import load_grayscale, load_rgb, prepare_word
from inksight.tokens import decode, detokenize, recognized_text
from inksight.words import expand_box


# ------------------------------------------------------------------ image loading
class TestLoadImages:
    def test_all_input_forms_agree(self, tmp_path):
        rgb = np.full((6, 8, 3), 255, np.uint8)
        rgb[2:4, 1:7] = (10, 20, 30)
        path = tmp_path / "drawing.png"
        Image.fromarray(rgb).save(path)

        from_array = load_rgb(rgb)
        assert from_array.shape == (6, 8, 3) and from_array.dtype == np.uint8
        np.testing.assert_array_equal(from_array, load_rgb(path))
        np.testing.assert_array_equal(from_array, load_rgb(str(path)))
        np.testing.assert_array_equal(from_array, load_rgb(Image.fromarray(rgb)))
        assert load_grayscale(path).shape == (6, 8)

    def test_array_dtypes(self):
        gray = np.array([[0, 128, 255]], np.uint8)
        np.testing.assert_array_equal(load_rgb(gray)[..., 0], gray)
        np.testing.assert_array_equal(load_rgb(gray / 255.0)[..., 1], gray)
        np.testing.assert_array_equal(load_rgb(np.array([[False, True]]))[..., 2], [[0, 255]])
        np.testing.assert_array_equal(load_rgb(np.array([[0, 65535]], np.uint16))[..., 0], [[0, 255]])

    def test_transparency_becomes_white_paper(self):
        rgba = np.zeros((2, 2, 4), np.uint8)
        rgba[0, 0] = (0, 0, 0, 255)
        np.testing.assert_array_equal(load_grayscale(rgba), [[0, 255], [255, 255]])

    @pytest.mark.parametrize("bad", [np.zeros((2, 2, 2)), np.zeros(5), np.zeros((0, 4)), 42,
                                     np.array([[np.nan]])])
    def test_rejects_non_images(self, bad):
        with pytest.raises(InvalidImageError):
            load_rgb(bad)

    def test_missing_file(self, tmp_path):
        with pytest.raises(InvalidImageError, match="not found"):
            load_rgb(tmp_path / "nope.png")


class TestPrepareWord:
    def test_wide_image_is_centred_on_black(self):
        rgb = np.full((100, 400, 3), 255, np.uint8)
        pixels, transform = prepare_word(rgb)
        assert pixels.shape == (224, 224, 3)
        assert transform.ratio == pytest.approx(0.56) and transform.pad_y == (224 - 56) // 2
        assert pixels[0, 0].tolist() == [0, 0, 0] and pixels[112, 112].tolist() == [255, 255, 255]

    def test_round_trip_and_crop_origin(self):
        rgb = np.zeros((300, 500, 3), np.uint8)
        _, transform = prepare_word(rgb, (100.7, 50.2, 300, 150))       # truncated like the notebook
        assert (transform.origin_x, transform.origin_y) == (100, 50)
        points = np.array([[100.0, 50.0], [300.0, 150.0], [123.4, 99.9]])
        np.testing.assert_allclose(transform.model_to_source(transform.source_to_model(points)), points)
        np.testing.assert_allclose(transform.source_to_model(points[:2]), [[0, 56], [224, 168]])

    def test_box_outside_the_image(self):
        with pytest.raises(InvalidImageError):
            prepare_word(np.zeros((10, 10, 3), np.uint8), (20, 20, 30, 30))


# ------------------------------------------------------------------ model output
class TestTokens:
    # 450 starts a stroke; 0..224 are x, 225..449 are y + 225 (notebook layout).
    def test_detokenize(self):
        strokes = detokenize([450, 10, 235, 11, 236, 450, 5, 230])
        assert len(strokes) == 2
        np.testing.assert_array_equal(strokes[0], [[10, 10], [11, 11]])
        np.testing.assert_array_equal(strokes[1], [[5, 5]])

    def test_skips_dangling_and_out_of_range(self):
        strokes = detokenize([450, 10, 450, 3, 4, 7, 232, 9])       # dangling x, y out of range
        assert len(strokes) == 1
        np.testing.assert_array_equal(strokes[0], [[7, 7]])
        with pytest.raises(ValueError):
            detokenize([451])

    def test_decode_text(self):
        raw = "<extra_id_0> hello <ink_token_450> <ink_token_1> <ink_token_226> <ink_token_2> <ink_token_227>"
        strokes, text = decode(raw)
        assert text == "hello" and len(strokes) == 1 and strokes[0].shape == (2, 2)
        assert recognized_text("<extra_id_0> <ink_token_450>") == ""

    def test_prompts(self):
        assert InferenceMode.coerce("r+d").prompt() == "Recognize and derender."
        assert InferenceMode.coerce("VANILLA") is InferenceMode.VANILLA
        assert InferenceMode.DERENDER_WITH_TEXT.prompt(" hi ") == "Derender the ink: hi"
        with pytest.raises(ValueError):
            InferenceMode.DERENDER_WITH_TEXT.prompt(None)
        with pytest.raises(ValueError):
            InferenceMode.coerce("nope")


# --------------------------------------------------------------------- drawings
class TestVectorDrawing:
    def test_svg_round_trip_keeps_dots(self, tmp_path):
        strokes = [np.array([[0.0, 0.0], [3.0, 4.0]]), np.array([[5.0, 5.0]])]
        drawing = VectorDrawing(strokes=strokes, width=10, height=8, method="demo")
        again = VectorDrawing.from_svg(drawing.save(tmp_path / "out.svg", stroke="red"))
        assert (again.width, again.height) == (10, 8) and len(again) == 2
        for a, b in zip(again, drawing):
            np.testing.assert_allclose(a, b, atol=1e-3)

    def test_measures(self):
        drawing = VectorDrawing([np.array([[0.0, 0.0], [3.0, 4.0]])], 5, 5, metadata={"text": "hi"})
        assert drawing.num_points == 2 and drawing.text == "hi"
        np.testing.assert_allclose(drawing.lengths(), [5.0])


# ------------------------------------------------------------------------ words
class TestWords:
    def test_json_round_trip(self, tmp_path):
        words = Words(boxes=[[1, 2, 30, 40], [50, 2, 90, 40.5]], texts=("a", None), scores=[0.9, np.nan])
        loaded = Words.load(words.save(tmp_path / "w.json"))
        np.testing.assert_allclose(loaded.boxes, words.boxes)
        assert loaded.texts == ("a", None) and np.isnan(loaded.scores[1])

    def test_coerce(self, tmp_path):
        from_array = Words.coerce(np.array([[0, 0, 5, 5]]))
        assert len(from_array) == 1 and from_array.texts == (None,)
        assert Words.coerce(from_array) is from_array
        assert Words.coerce([{"box": [0, 0, 2, 2], "text": "x"}]).texts == ("x",)
        assert len(Words.coerce(np.zeros((0, 4)))) == 0
        with pytest.raises(ValueError):
            Words.coerce(np.zeros((3, 2)))
        with pytest.raises(ValueError):
            Words(boxes=[[5, 5, 1, 1]])

    def test_expand_box_like_the_notebook(self):
        assert expand_box([10, 10, 30, 20], 0.1, 100, 100) == [8, 9, 32, 21]
        assert expand_box([0, 0, 100, 50], 0.5, 100, 50) == [0, 0, 100, 50]

    def test_detector_registry(self):
        assert {"doctr", "paddleocr", "tesseract", "whole"} <= set(inksight.available_detectors())
        assert isinstance(inksight.get_detector("paddle"), inksight.PaddleOCRWordDetector)
        assert len(inksight.get_detector("whole").detect(np.zeros((4, 6), np.uint8))) == 1
        with pytest.raises(ValueError, match="unknown word detector"):
            inksight.get_detector("nope")


# ------------------------------------------------------------------------ bundle
class TestBundle:
    def _fake_model(self, root: Path) -> Path:
        model = root / "fake"
        (model / "variables").mkdir(parents=True)
        (model / "saved_model.pb").write_bytes(b"graph" * 100)
        (model / "variables" / "variables.data-00000-of-00001").write_bytes(np.random.default_rng(0).bytes(50_000))
        (model / "variables" / "variables.index").write_bytes(b"index")
        return model

    def test_compress_split_and_unpack(self, tmp_path):
        model = self._fake_model(tmp_path)
        manifest = bundle.bundle_model(model, tmp_path / "bundle", preset=0, part_size=20_000)
        assert len(manifest["parts"]) >= 3
        assert all(part["size"] <= 20_000 for part in manifest["parts"])

        unpacked = bundle.extract_model(tmp_path / "home", manifest=tmp_path / "bundle" / "MANIFEST.json")
        assert unpacked.name == bundle.MODEL_NAME
        for name in manifest["files"]:
            assert (unpacked / name).read_bytes() == (model / name).read_bytes()
        # a second call reuses the copy
        assert bundle.extract_model(tmp_path / "home", manifest=tmp_path / "bundle" / "MANIFEST.json") == unpacked

    def test_damaged_archive_is_refused(self, tmp_path):
        model = self._fake_model(tmp_path)
        manifest = bundle.bundle_model(model, tmp_path / "bundle", preset=0, part_size=20_000)
        info = json.loads((tmp_path / "bundle" / "MANIFEST.json").read_text())
        info["sha256"] = "0" * 64
        (tmp_path / "bundle" / "MANIFEST.json").write_text(json.dumps(info))
        with pytest.raises(BackendBuildError, match="damaged"):
            bundle.extract_model(tmp_path / "home", manifest=tmp_path / "bundle" / "MANIFEST.json")
        assert not any((tmp_path / "home").iterdir())          # nothing half-unpacked left behind

    def test_package_carries_the_model(self):
        info = bundle.bundle_info()
        assert info is not None and info["name"] == "small-p-cpu"
        assert all(part["size"] < 100 * 10 ** 6 for part in info["parts"])


# ------------------------------------------------------- hierarchy & extension
class FakeRunner(Runner):
    """Pretends to be the model program: answers every word with fixed tokens."""

    name = "fake"

    def __init__(self, raw="<extra_id_0> hi <ink_token_450> <ink_token_10> <ink_token_285>",
                 returncode=0, write_output=True):
        self.raw, self.returncode, self.write_output, self.calls = raw, returncode, write_output, []

    def is_available(self, tool):
        return True

    def run(self, tool, args, workdir, timeout=None):
        workdir = Path(workdir)
        job = json.loads((workdir / args[0]).read_text())
        self.calls.append((tool, list(args), sorted(p.name for p in workdir.iterdir()), job))
        if self.write_output:
            outputs = [{"raw": self.raw, "prompt": p, "fallback": False} for p in job["prompts"]]
            (workdir / "output.json").write_text(json.dumps({"outputs": outputs, "tokenizer": "fake"}))
        return RunResult([tool, *args], self.returncode, "log line", 0.25)


class EchoVectorizer(ModelVectorizer):
    """A third method added from outside the package, as a user would."""

    name = "echo-test"
    aliases = ("echo-alias",)

    def _stage(self, rgb, **inputs):
        return ModelJob(Words(boxes=[[0, 0, rgb.shape[1], rgb.shape[0]]]), metadata={"shape": rgb.shape})


class TestHierarchy:
    def test_builtin_methods_are_registered(self):
        assert {"page", "word"} <= set(inksight.available_methods())
        assert inksight.get_vectorizer_class("word") is inksight.InkSightWord
        assert inksight.get_vectorizer_class("INKSIGHT") is inksight.InkSightPage
        for cls in (inksight.InkSightWord, inksight.InkSightPage):
            assert issubclass(cls, ModelVectorizer) and issubclass(cls, Vectorizer)
        with pytest.raises(ValueError, match="unknown vectorization method"):
            inksight.get_vectorizer("nope")

    def test_subclass_registers_and_runs_through_the_template(self):
        assert inksight.get_vectorizer_class("echo-alias") is EchoVectorizer
        runner = FakeRunner()
        vectorizer = inksight.get_vectorizer("echo-test", runner=runner)

        drawing = vectorizer.vectorize(np.full((224, 448, 3), 255, np.uint8))

        tool, args, staged, job = runner.calls[0]
        assert tool == "inksight-model" and args == ["job.json"]
        assert staged == ["job.json", "word_000.png"]                  # staged before the run
        assert job["prompts"] == ["Recognize and derender."]
        assert job["fallback_prompts"] == ["Derender the ink."]
        assert drawing.method == "echo-test" and (drawing.width, drawing.height) == (448, 224)
        # ratio 0.5, padding 56 px on top: model (10.5, 60.5) -> source (21, 9)
        np.testing.assert_allclose(drawing[0], [[21.0, 9.0]])
        assert drawing.text == "hi"
        assert drawing.metadata["runner"] == "fake" and drawing.metadata["shape"] == (224, 448, 3)
        assert vectorizer.last_run.output == "log line"

    def test_preprocessing_reaches_the_model(self, tmp_path):
        vectorizer = EchoVectorizer(runner=FakeRunner(), invert=True, workdir=tmp_path)
        vectorizer.vectorize(np.zeros((224, 224), np.uint8))
        staged = np.asarray(Image.open(tmp_path / "word_000.png"))
        assert staged.min() == 255                                  # inverted: black became white
        assert (tmp_path / "log.txt").read_text() == "log line"

    @pytest.mark.parametrize("runner", [FakeRunner(returncode=3), FakeRunner(write_output=False)])
    def test_model_failure_raises_with_log(self, runner):
        with pytest.raises(VectorizationError) as raised:
            EchoVectorizer(runner=runner).vectorize(np.zeros((2, 2), np.uint8))
        assert raised.value.log == "log line"

    def test_word_modes(self):
        runner = FakeRunner()
        inksight.InkSightWord(runner=runner, mode="d+t").vectorize(np.zeros((4, 4)), text="hey")
        assert runner.calls[-1][3]["prompts"] == ["Derender the ink: hey"]
        inksight.InkSightWord(runner=runner, mode="vanilla", fallback_mode=None).vectorize(np.zeros((4, 4)))
        assert runner.calls[-1][3]["fallback_prompts"] is None
        with pytest.raises(ValueError, match="needs the word's text"):
            inksight.InkSightWord(runner=runner, mode="d+t").vectorize(np.zeros((4, 4)))

    def test_page_uses_given_words_without_the_detector(self, tmp_path):
        class NoDetector(WordDetector):
            def detect(self, image):
                raise AssertionError("detector must not run when words are supplied")

        runner = FakeRunner()
        page = inksight.InkSightPage(runner=runner, word_detector=NoDetector(), workdir=tmp_path)
        drawing = page.vectorize(np.full((50, 200), 255, np.uint8),
                                 words=np.array([[0, 0, 100, 50], [100, 0, 200, 50]]))
        assert runner.calls[0][2] == ["job.json", "word_000.png", "word_001.png"]
        assert len(drawing) == 2 and [w["strokes"] for w in drawing.metadata["words"]] == [(0, 1), (1, 2)]
        assert drawing[1][0, 0] > 100                               # second word lands in its box
        assert drawing.metadata["detector"] == "given"

    def test_page_with_no_words(self):
        class Empty(WordDetector):
            def detect(self, image):
                return Words(boxes=np.zeros((0, 4)))

        runner = FakeRunner()
        drawing = inksight.InkSightPage(runner=runner, word_detector=Empty()).vectorize(np.zeros((9, 9)))
        assert len(drawing) == 0 and not runner.calls

    def test_word_rejects_unknown_inputs(self):
        with pytest.raises(TypeError, match="words"):
            inksight.InkSightWord(runner=FakeRunner()).vectorize(np.zeros((2, 2)), words=[[0, 0, 1, 1]])


# ----------------------------------------------------------- the model program
class FakeSession:
    def __init__(self, answers):
        self.answers, self.calls = answers, []
        self.model_dir, self.tokenizer, self.load_seconds = Path("m"), "fake", 0.0

    def generate(self, images, prompts, batch_size=32):
        self.calls.append(list(prompts))
        return [self.answers[p] for p in prompts]


class TestWorker:
    def _job(self, tmp_path, prompts, fallback):
        for i in range(len(prompts)):
            Image.new("RGB", (224, 224)).save(tmp_path / f"w{i}.png")
        (tmp_path / "job.json").write_text(json.dumps({
            "images": [f"w{i}.png" for i in range(len(prompts))], "prompts": prompts,
            "fallback_prompts": fallback}))

    def test_fallback_retries_only_empty_words_in_one_batch(self, tmp_path, monkeypatch):
        session = FakeSession({"A": "<ink_token_450> <ink_token_1> <ink_token_226>", "B": "nothing",
                               "F": "<ink_token_450> <ink_token_2> <ink_token_227>"})
        monkeypatch.setattr(worker.ModelSession, "get", classmethod(lambda cls, *a, **k: session))
        monkeypatch.setattr(worker, "resolve_model_dir", lambda model_dir=None: Path("m"))
        self._job(tmp_path, ["A", "B", "B"], ["F", "F", "F"])

        assert worker.main(["job.json"], workdir=tmp_path) == 0
        result = json.loads((tmp_path / "output.json").read_text())
        assert session.calls == [["A", "B", "B"], ["F", "F"]]
        assert [o["fallback"] for o in result["outputs"]] == [False, True, True]

    def test_local_runner_runs_in_process_and_captures_the_log(self, tmp_path, monkeypatch):
        session = FakeSession({"A": "<ink_token_450>"})
        monkeypatch.setattr(worker.ModelSession, "get", classmethod(lambda cls, *a, **k: session))
        monkeypatch.setattr(worker, "resolve_model_dir", lambda model_dir=None: Path("m"))
        monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: object())
        self._job(tmp_path, ["A"], None)

        result = inksight.LocalRunner().run("inksight-model", ["job.json"], tmp_path)
        assert result.returncode == 0 and "derendered 1 word" in result.output
        assert logging.getLogger("inksight.model").handlers == []

    def test_bad_job_fails_cleanly(self, tmp_path):
        (tmp_path / "job.json").write_text(json.dumps({"images": ["a.png"], "prompts": []}))
        assert worker.main(["job.json"], workdir=tmp_path) == 1


def test_image_name_tracks_the_sources():
    name = inksight.DockerRunner().image
    assert name.startswith("inksight-backend:") and len(name.split(":")[1]) == 12


def test_gpu_docker_runner_uses_a_cuda_image_and_passes_the_gpus(monkeypatch, tmp_path):
    monkeypatch.delenv("INKSIGHT_DOCKER_GPUS", raising=False)
    cpu, gpu = inksight.DockerRunner(), inksight.get_runner("docker-gpu", "inksight-model")
    assert gpu.gpus == "all" and "and-cuda" in gpu.build_args["TF_PACKAGE"]
    assert cpu.image != gpu.image                                  # separate tags

    commands = []
    monkeypatch.setattr(gpu, "ensure_image", lambda: gpu.image)
    monkeypatch.setattr("subprocess.run", lambda command, **kw: commands.append(command)
                        or type("Done", (), {"returncode": 0, "stdout": ""})())
    gpu.run("inksight-model", ["job.json"], tmp_path)
    assert commands[0][commands[0].index("--gpus") + 1] == "all"

    monkeypatch.setenv("INKSIGHT_DOCKER_GPUS", "device=0")
    assert inksight.DockerRunner().gpus == "device=0"


class TestPersistentServer:
    """The serve protocol, with the server in a thread instead of a container."""

    def _serve_in_thread(self, monkeypatch, spool, answers):
        import threading
        session = FakeSession(answers)
        monkeypatch.setattr(worker.ModelSession, "get", classmethod(lambda cls, *a, **k: session))
        monkeypatch.setattr(worker, "resolve_model_dir", lambda model_dir=None: Path("m"))
        thread = threading.Thread(target=worker.serve, args=(spool,), kwargs={"poll": 0.01},
                                  daemon=True)
        thread.start()
        return session, thread

    def test_runner_sends_every_call_to_one_server(self, tmp_path, monkeypatch):
        spool = tmp_path / "spool"
        session, thread = self._serve_in_thread(monkeypatch, spool, {
            "Recognize and derender.": "<ink_token_450> <ink_token_1> <ink_token_226>"})
        runner = inksight.DockerRunner(persistent=True)
        starts = []

        def fake_start():
            starts.append(1)
            runner._spool, runner._server = spool, "fake-server"
            return "fake-server"

        monkeypatch.setattr(runner, "start_server", fake_start)
        monkeypatch.setattr(runner, "_server_running", lambda: True)
        word = inksight.InkSightWord(runner=runner)
        first = word.vectorize(np.full((20, 40), 255, np.uint8))
        second = word.vectorize(np.full((20, 40), 255, np.uint8))
        assert len(first) == len(second) == 1 and len(starts) == 2       # start_server per call ...
        assert len(session.calls) == 2                                   # ... one model session
        assert "derendered 1 word" in word.last_run.output
        assert [p.name for p in spool.iterdir() if p.is_dir()] == []     # job dirs cleaned up

        (spool / worker.STOP).touch()
        thread.join(5)
        assert not thread.is_alive()

    def test_a_bad_job_does_not_stop_the_server(self, tmp_path, monkeypatch):
        import time
        spool = tmp_path / "spool"
        _, thread = self._serve_in_thread(monkeypatch, spool, {})
        bad = spool / "bad"
        bad.mkdir(parents=True)
        (bad / "job.json").write_text(json.dumps({"images": ["a.png"], "prompts": []}))
        (bad / worker.READY).touch()
        for _ in range(500):
            if (bad / worker.DONE).exists():
                break
            time.sleep(0.01)
        assert (bad / "exitcode").read_text() == "1" and thread.is_alive()
        (spool / worker.STOP).touch()
        thread.join(5)

    def test_server_runner_names(self, monkeypatch):
        monkeypatch.delenv("INKSIGHT_DOCKER_PERSISTENT", raising=False)
        assert inksight.get_runner("docker-server", "inksight-model").persistent
        both = inksight.get_runner("docker-gpu-server", "inksight-model")
        assert both.persistent and both.gpus == "all"
        assert not inksight.DockerRunner().persistent
        monkeypatch.setenv("INKSIGHT_DOCKER_PERSISTENT", "1")
        assert inksight.DockerRunner().persistent


def test_vectorize_many_sends_all_pages_in_one_call(tmp_path):
    runner = FakeRunner()
    page = inksight.InkSightPage(runner=runner, word_detector="whole")
    drawings = page.vectorize_many(
        [np.full((50, 200), 255, np.uint8), np.full((30, 30), 255, np.uint8), np.zeros((9, 9))],
        [{"words": np.array([[0, 0, 100, 50], [100, 0, 200, 50]])}, {}, {"words": np.zeros((0, 4))}])
    assert len(runner.calls) == 1                                    # one model invocation
    assert len(runner.calls[0][3]["images"]) == 3                    # 2 + 1 + 0 words
    assert [len(d) for d in drawings] == [2, 1, 0]
    assert [(d.width, d.height) for d in drawings] == [(200, 50), (30, 30), (9, 9)]
    assert drawings[0][1][0, 0] > 100 and drawings[1][0][0, 0] < 30  # strokes land in their own page
    assert drawings[0].metadata["batched_images"] == 3
    with pytest.raises(ValueError, match="input dicts"):
        page.vectorize_many([np.zeros((4, 4))], [{}, {}])


class TestScheduling:
    """Word ordering and batch padding in ModelSession.generate (no TensorFlow needed)."""

    class _Signature:
        def __init__(self):
            self.batches = []

        def __call__(self, **inputs):
            texts = [t.decode() for t in inputs["input_text"].numpy()]
            self.batches.append(texts)
            answer = np.array([[f"out:{t}".encode()] for t in texts], dtype=object)
            return {"output_0": type("Tensor", (), {"numpy": lambda self: answer})()}

    def _session(self, on_accelerator):
        tf = pytest.importorskip("tensorflow")
        session = worker.ModelSession.__new__(worker.ModelSession)
        session._tf, session._signature, session.on_accelerator = tf, self._Signature(), on_accelerator
        return session

    @staticmethod
    def _word(ink):
        image = np.zeros((224, 224, 3), np.uint8)
        image[50:150, :] = 255                         # paper inside black padding
        image[60:60 + ink, 10:20] = 0                  # ink, 10 px per row
        return image

    def test_padded_size(self):
        assert [worker.padded_size(n, 32) for n in (1, 2, 3, 5, 17, 32)] == [1, 2, 4, 8, 32, 32]
        assert worker.padded_size(20, 24) == 24

    def test_ink_estimate_ignores_padding(self):
        assert worker.ink_estimate(self._word(0)) == 0
        assert worker.ink_estimate(self._word(7)) == 70

    def test_sorting_groups_by_length_and_restores_order(self):
        session = self._session(on_accelerator=False)
        inks = [80, 1, 60, 2, 70]
        out = session.generate([self._word(i) for i in inks], [f"p{i}" for i in inks], batch_size=2)
        assert out == [f"out:p{i}" for i in inks]                    # original order
        assert session._signature.batches == [["p1", "p2"], ["p60", "p70"], ["p80"]]

    def test_padding_on_accelerators_only(self):
        gpu = self._session(on_accelerator=True)
        assert gpu.generate([self._word(1)] * 3, ["a", "b", "c"], batch_size=32) == ["out:a", "out:b", "out:c"]
        assert gpu._signature.batches == [["a", "b", "c", "c"]]      # padded to 4
        cpu = self._session(on_accelerator=False)
        cpu.generate([self._word(1)] * 3, ["a", "b", "c"], batch_size=32)
        assert cpu._signature.batches == [["a", "b", "c"]]


class TestWslRunner:
    def test_paths_and_command(self, monkeypatch, tmp_path):
        from inksight.runners.wsl import to_wsl_path
        assert to_wsl_path(r"C:\Users\me\x y") == "/mnt/c/Users/me/x y"
        runner = inksight.WslRunner("Ubuntu-24.04", "/opt/py/bin/python", model_dir="/opt/model")
        command = runner.command(["job.json"], cwd=Path(r"D:\work\job1"))
        assert command[:5] == ["wsl.exe", "-d", "Ubuntu-24.04", "--cd", "/mnt/d/work/job1"]
        assert "INKSIGHT_MODEL_DIR=/opt/model" in command
        assert any(c.startswith("PYTHONPATH=/mnt/") and c.endswith("/src") for c in command)
        assert command[-4:] == ["/opt/py/bin/python", "-m", "inksight.worker", "job.json"]

    def test_runner_names(self, monkeypatch):
        monkeypatch.delenv("INKSIGHT_WSL_PERSISTENT", raising=False)
        assert not inksight.get_runner("wsl", "inksight-model").persistent
        assert inksight.get_runner("rocm", "inksight-model").persistent

    def test_persistent_mode_uses_the_spool_protocol(self, tmp_path, monkeypatch):
        spool = tmp_path / "spool"
        session, thread = TestPersistentServer()._serve_in_thread(monkeypatch, spool, {
            "Recognize and derender.": "<ink_token_450> <ink_token_1> <ink_token_226>"})
        runner = inksight.WslRunner(persistent=True)
        monkeypatch.setattr(runner, "start_server", lambda: setattr(runner, "_spool", spool))
        drawing = inksight.InkSightWord(runner=runner).vectorize(np.full((20, 40), 255, np.uint8))
        assert len(drawing) == 1 and len(session.calls) == 1
        (spool / worker.STOP).touch()
        thread.join(5)


def test_rocm_tensorflow_keeps_xla_autotuning(monkeypatch):
    monkeypatch.setenv("INKSIGHT_ROCM", "1")
    assert worker.tensorflow_is_rocm()
    assert not any("autotune_level" in flag for flag in worker.XLA_FLAGS_ROCM)
    monkeypatch.setenv("INKSIGHT_ROCM", "0")
    assert not worker.tensorflow_is_rocm()
