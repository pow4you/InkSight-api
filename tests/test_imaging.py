import numpy as np
import pytest
from PIL import Image

from inksight_api import BoundingBox, PadColor, SquarePadPreprocessor
from inksight_api.imaging import draw_boxes, load_image, resize_to_max_side


def test_load_image_accepts_all_sources(tmp_path):
    img = Image.new("RGBA", (10, 8), (255, 0, 0, 255))
    path = tmp_path / "a.png"
    img.save(path)
    for source in (path, str(path), path.read_bytes(), bytearray(path.read_bytes()), img):
        out = load_image(source)
        assert out.mode == "RGB" and out.size == (10, 8)
    with pytest.raises(TypeError):
        load_image(123)  # type: ignore[arg-type]


def test_wide_image_is_centred_with_black_padding():
    prep = SquarePadPreprocessor().prepare(Image.new("RGB", (448, 112), "white"))
    arr = np.asarray(prep.image)
    assert prep.image.size == (224, 224)
    assert prep.transform.ratio == 0.5 and (prep.transform.pad_x, prep.transform.pad_y) == (0, 84)
    assert arr[0, 0].tolist() == [0, 0, 0] and arr[112, 112].tolist() == [255, 255, 255]


def test_tall_image_and_white_padding():
    prep = SquarePadPreprocessor(pad_color=PadColor.WHITE).prepare(Image.new("RGB", (50, 200), "black"))
    assert (prep.transform.pad_x, prep.transform.pad_y) == (84, 0)  # 50 * 1.12 = 56 wide
    assert np.asarray(prep.image)[100, 0].tolist() == [255, 255, 255]


def test_corner_average_padding():
    prep = SquarePadPreprocessor(pad_color=PadColor.CORNER_AVERAGE).prepare(Image.new("RGB", (448, 112), (10, 20, 30)))
    assert np.asarray(prep.image)[0, 0].tolist() == [10, 20, 30]


def test_box_crop_records_origin():
    img = Image.new("RGB", (400, 300), "white")
    prep = SquarePadPreprocessor().prepare(img, BoundingBox(100, 50, 100, 50))
    t = prep.transform
    assert (t.origin_x, t.origin_y) == (100, 50) and t.ratio == pytest.approx(2.24)
    assert (t.pad_x, t.pad_y) == (0, 56)


def test_degenerate_box_does_not_crash():
    SquarePadPreprocessor().prepare(Image.new("RGB", (400, 300)), BoundingBox(10, 10, 1, 400))


def test_helpers():
    big = Image.new("RGB", (4000, 1000))
    assert resize_to_max_side(big, 2000).size == (2000, 500)
    small = Image.new("RGB", (10, 10))
    assert resize_to_max_side(small) is small
    marked = draw_boxes(Image.new("RGB", (50, 50), "white"), [BoundingBox(5, 5, 20, 20)])
    assert np.asarray(marked)[5, 5].tolist() == [255, 0, 0]
