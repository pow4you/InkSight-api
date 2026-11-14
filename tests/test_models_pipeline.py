from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from conftest import FakeBackend, FixedSegmenter, encode
from inksight_api import (InferenceMode, InkSight, InkSightModel, PageInkPipeline, PromptError,
                          WordInkPipeline)
from inksight_api.models import TFSavedModelBackend
from inksight_api.segmentation import WholeImageSegmenter


def test_model_defaults_and_recognized_text():
    backend = FakeBackend()
    model = InkSightModel(backend)
    out = model.derender_one(Image.new("RGB", (224, 224)))
    assert backend.calls == [["Recognize and derender."]]
    assert out.mode is InferenceMode.RECOGNIZE_AND_DERENDER and not out.used_fallback
    assert out.recognized_text == "word" and len(out.ink) == 1
    assert model.derender([]) == []


def test_fallback_is_one_batched_retry_for_empty_results_only():
    def responder(image, prompt):
        if prompt == "Recognize and derender.":
            return "" if image.getpixel((0, 0)) == (1, 1, 1) else encode([[(1, 1), (2, 2)]])
        return encode([[(9, 9), (8, 8)]])  # vanilla works

    backend = FakeBackend(responder)
    images = [Image.new("RGB", (224, 224), c) for c in [(0, 0, 0), (1, 1, 1), (0, 0, 0), (1, 1, 1)]]
    outs = InkSightModel(backend).derender(images)
    assert [o.used_fallback for o in outs] == [False, True, False, True]
    assert len(backend.calls) == 2 and backend.calls[1] == ["Derender the ink."] * 2
    assert outs[1].mode is InferenceMode.VANILLA and outs[1].ink[0][0] == (9.0, 9.0)


def test_fallback_can_be_disabled_and_is_skipped_for_same_mode():
    backend = FakeBackend(lambda i, p: "")
    model = InkSightModel(backend, fallback_mode=None)
    assert model.derender_one(Image.new("RGB", (8, 8))).ink.is_empty and len(backend.calls) == 1
    backend2 = FakeBackend(lambda i, p: "")
    InkSightModel(backend2).derender([Image.new("RGB", (8, 8))], mode="vanilla")
    assert len(backend2.calls) == 1


def test_text_guided_mode():
    backend = FakeBackend()
    model = InkSightModel(backend)
    model.derender([Image.new("RGB", (8, 8))] * 2, mode="d+t", texts=["foo", "bar"])
    assert backend.calls[0] == ["Derender the ink: foo", "Derender the ink: bar"]
    with pytest.raises(PromptError):
        model.derender([Image.new("RGB", (8, 8))], mode="d+t")
    with pytest.raises(ValueError):
        model.derender([Image.new("RGB", (8, 8))], texts=["a", "b"])


def test_page_pipeline_maps_ink_back_into_source_pixels(page_image, two_words):
    # The fake always draws a point at the model-frame centre (112, 112), so after mapping
    # back it must land on the centre of each word's box.
    result = PageInkPipeline(InkSightModel(FakeBackend()), FixedSegmenter(two_words)).run(page_image)
    assert len(result.words) == 2 and result.text == "word word"
    for word, wb in zip(result.words, two_words):
        cx, cy = wb.box.left + wb.box.width / 2, wb.box.top + wb.box.height / 2
        assert word.ink[0][1] == pytest.approx((cx, cy), abs=1.0)
        assert word.model_input.size == (224, 224) and word.output.ink[0][1] == (112.0, 112.0)
    assert len(result.ink) == 2 and result.elapsed_seconds >= 0
    assert result.annotated_image().size == page_image.size


def test_page_pipeline_passes_ocr_text_in_d_plus_t_mode(page_image, two_words):
    backend = FakeBackend()
    PageInkPipeline(InkSightModel(backend), FixedSegmenter(two_words), mode="d+t").run(page_image)
    assert backend.calls[0] == ["Derender the ink: hello", "Derender the ink: world"]


def test_empty_segmentation_skips_the_model(page_image):
    backend = FakeBackend()
    result = PageInkPipeline(InkSightModel(backend), FixedSegmenter([])).run(page_image)
    assert result.words == [] and result.ink.is_empty and backend.calls == []


def test_texts_override_and_length_check(page_image, two_words):
    pipe = PageInkPipeline(InkSightModel(FakeBackend()), FixedSegmenter(two_words))
    assert [w.ocr_text for w in pipe.run(page_image, texts=["a", "b"]).words] == ["a", "b"]
    with pytest.raises(ValueError):
        pipe.run(page_image, texts=["only one"])


def test_word_pipeline_is_whole_image_and_accepts_text(tmp_path):
    path = tmp_path / "w.png"
    Image.new("RGB", (300, 100), "white").save(path)
    backend = FakeBackend()
    result = WordInkPipeline(InkSightModel(backend), mode="d+t").run(path, text="hi")
    assert backend.calls[0] == ["Derender the ink: hi"] and len(result.words) == 1
    assert isinstance(WordInkPipeline(InkSightModel(backend)).segmenter, WholeImageSegmenter)


def test_pipeline_stages_are_overridable(page_image, two_words):
    class SkipFirstWord(PageInkPipeline):
        def _segment(self, image):
            return super()._segment(image)[1:]

    result = SkipFirstWord(InkSightModel(FakeBackend()), FixedSegmenter(two_words)).run(page_image)
    assert [w.ocr_text for w in result.words] == ["world"]


def test_facade_word_and_page(page_image, two_words):
    kit = InkSight(InkSightModel(FakeBackend()), segmenter=FixedSegmenter(two_words))
    assert len(kit.convert_page(page_image).words) == 2
    assert len(kit.convert_page(page_image, mode="vanilla").words) == 2
    assert len(kit.convert_word(page_image).words) == 1


def test_facade_mode_does_not_leak_between_calls(page_image, two_words):
    backend = FakeBackend()
    kit = InkSight(InkSightModel(backend), segmenter=FixedSegmenter(two_words))
    kit.convert_page(page_image, mode="vanilla")
    kit.convert_page(page_image)
    assert backend.calls[0][0] == "Derender the ink." and backend.calls[1][0] == "Recognize and derender."


def test_facade_builds_named_segmenter_once(page_image):
    kit = InkSight(InkSightModel(FakeBackend()), segmenter="whole")
    kit.convert_page(page_image)
    first = kit._segmenter_cache["whole"]
    kit.convert_page(page_image)
    assert kit._segmenter_cache["whole"] is first


# --- the TensorFlow backend, exercised through a stand-in `tf` namespace ---------------------
def _fake_tf():
    return SimpleNamespace(
        string="string",
        constant=lambda values, dtype=None: list(values),
        reshape=lambda t, shape: t,
        stack=lambda items: np.stack(items),
        io=SimpleNamespace(encode_jpeg=lambda arr: np.array([arr.shape[0]])),  # tag = image height
    )


def test_tf_backend_batches_and_decodes():
    seen = []

    def signature(**kwargs):
        seen.append({k: (np.shape(v)) for k, v in kwargs.items()})
        prompts = kwargs["input_text"]
        out = np.array([[f"{p}|{i}".encode()] for i, p in enumerate(prompts)], dtype=object)
        return {"output_0": SimpleNamespace(numpy=lambda: out)}

    backend = TFSavedModelBackend(signature, tf=_fake_tf(), batch_size=2)
    images = [Image.new("RGB", (4, 4 + i)) for i in range(5)]
    outputs = backend.generate(images, [f"p{i}" for i in range(5)])
    assert outputs == ["p0|0", "p1|1", "p2|0", "p3|1", "p4|0"]
    assert [s["input_text"] for s in seen] == [(2,), (2,), (1,)]
    assert seen[0]["image/encoded"] == (2, 1)  # (batch, 1) like the notebook
    with pytest.raises(ValueError):
        backend.generate(images, ["only one"])
