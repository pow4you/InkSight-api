import numpy as np
import pytest

from conftest import TOK, encode
from inksight_api import (BoundingBox, CropTransform, InferenceMode, Ink, InkTokenError,
                          PromptError, Stroke)


class TestStrokeAndInk:
    def test_notebook_compatible_access(self):
        s = Stroke([(1, 2), (3, 4), (5, 6)])
        assert len(s) == 3 and s[1] == (3.0, 4.0)
        assert list(s.x) == [1, 3, 5] and list(s.y) == [2, 4, 6]
        assert list(s) == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
        ink = Ink([s, Stroke()])
        assert len(ink) == 2 and ink.strokes[0] is s and ink[0] is s

    def test_immutable(self):
        s = Stroke([(0, 0), (1, 1)])
        with pytest.raises(ValueError):
            s.points[0, 0] = 5
        src = np.zeros((2, 2))
        s2 = Stroke(src)
        src[0, 0] = 9
        assert s2[0] == (0.0, 0.0)  # copied

    def test_bad_shape(self):
        with pytest.raises(ValueError):
            Stroke(np.zeros((3, 3)))

    def test_segments_shape(self):
        assert Stroke([(0, 0), (1, 0), (1, 1)]).segments.shape == (2, 2, 2)
        assert Stroke([(0, 0)]).segments.shape == (0, 2, 2)
        assert Stroke().bounds() is None
        assert Stroke([(0, 0), (3, 4)]).length() == 5.0

    def test_transforms_return_new_objects_and_keep_text(self):
        ink = Ink([Stroke([(0, 0), (2, 4)])], text="hi", metadata={"a": "b"})
        moved = ink.scaled(2).translated(1, 1)
        assert moved[0].points.tolist() == [[1, 1], [5, 9]]
        assert ink[0].points.tolist() == [[0, 0], [2, 4]]  # untouched
        assert moved.text == "hi" and moved.metadata == {"a": "b"}
        assert ink.scaled(2, 3)[0].points.tolist() == [[0, 0], [4, 12]]

    def test_merge_bounds_polylines(self):
        a, b = Ink([Stroke([(0, 0), (1, 1)])]), Ink([Stroke([(5, 5), (6, 7)])])
        merged = Ink.merge([a, b], text="ab")
        assert len(merged) == 2 and merged.text == "ab" and merged.point_count == 4
        assert merged.bounds() == (0.0, 0.0, 6.0, 7.0)
        assert len(merged.to_polylines()) == 2
        assert Ink().is_empty and Ink().bounds() is None


class TestPromptsAndTokens:
    def test_prompts_match_model_card(self):
        assert InferenceMode.VANILLA.build_prompt() == "Derender the ink."
        assert InferenceMode.RECOGNIZE_AND_DERENDER.build_prompt() == "Recognize and derender."
        assert InferenceMode.DERENDER_WITH_TEXT.build_prompt(" hello ") == "Derender the ink: hello"

    def test_d_plus_t_needs_text(self):
        with pytest.raises(PromptError):
            InferenceMode.DERENDER_WITH_TEXT.build_prompt(None)
        with pytest.raises(PromptError):
            InferenceMode.DERENDER_WITH_TEXT.build_prompt("  ")

    def test_aliases(self):
        assert InferenceMode.from_alias("r+d") is InferenceMode.RECOGNIZE_AND_DERENDER
        assert InferenceMode.from_alias("D+T") is InferenceMode.DERENDER_WITH_TEXT
        assert InferenceMode.from_alias("vanilla") is InferenceMode.VANILLA
        assert InferenceMode.from_alias(InferenceMode.VANILLA) is InferenceMode.VANILLA
        with pytest.raises(PromptError):
            InferenceMode.from_alias("nope")

    def test_vocabulary_layout(self):
        assert (TOK.tokens_per_dimension, TOK.vocabulary_size, TOK.start_token) == (225, 451, 450)

    def test_roundtrip(self):
        strokes = [[(0, 0), (10, 20), (224, 224)], [(5, 6), (7, 8)]]
        ink = TOK.decode(encode(strokes, text="hello"))
        assert [list(s) for s in ink] == [[(float(x), float(y)) for x, y in s] for s in strokes]
        assert ink.text == "hello"

    def test_no_text_means_none(self):
        assert TOK.decode(encode([[(1, 2), (3, 4)]])).text is None

    def test_same_semantics_as_notebook_detokenize(self):
        start, off = TOK.start_token, TOK.tokens_per_dimension
        # stroke 1: one good pair, then a dangling x right before the next stroke start
        # stroke 2: a good pair, then a pair whose "y" is really an x token (out of range -> dropped)
        tokens = [start, 10, 20 + off, 30, start, 5, 6 + off, 7, 8]
        ink = TOK.detokenize(tokens)
        assert [list(s) for s in ink] == [[(10.0, 20.0)], [(5.0, 6.0)]]
        assert TOK.detokenize([]).is_empty
        assert TOK.detokenize([start, start]).is_empty

    def test_out_of_vocabulary_raises(self):
        with pytest.raises(InkTokenError):
            TOK.detokenize([TOK.vocabulary_size])
        with pytest.raises(ValueError):  # also a ValueError, like the notebook
            TOK.detokenize([-1])


class TestGeometry:
    def test_box_expansion_clips(self):
        b = BoundingBox(10, 10, 100, 50).expanded(0.1, bounds=(105, 200))
        assert (b.left, b.top, b.right, b.bottom) == (0, 5, 105, 65)
        assert BoundingBox(1.9, 2.9, 3.9, 4.9).to_int() == BoundingBox(1, 2, 3, 4)

    def test_transform_roundtrip(self):
        t = CropTransform(ratio=2.24, pad_x=0, pad_y=56, origin_x=100, origin_y=50)
        pts = np.array([[100.0, 50.0], [150.0, 75.0], [200.0, 100.0]])
        assert np.allclose(t.model_to_source(t.source_to_model(pts)), pts)
        assert np.allclose(t.source_to_model(pts)[1], [112, 112])
