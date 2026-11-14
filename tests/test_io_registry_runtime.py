import xml.etree.ElementTree as ET

import pytest

from conftest import FakeBackend
from inksight_api import (BoundingBox, Ink, InkSight, InkSightModel, MissingDependencyError,
                          RegistryError, Stroke, read_ink, write_ink)
from inksight_api.io import InkMLReader, InkReader, SVGWriter, ink_readers, ink_writers
from inksight_api.models import XLA_FLAGS, configure_tensorflow
from inksight_api.registry import Registry
from inksight_api.segmentation import WordBox, WordSegmenter, segmenters
from inksight_api.utils import require

SAMPLE = Ink([Stroke([(1.5, 2.25), (3, 4)]), Stroke([(10, 10), (11, 12), (13, 9)])], text="hi")


def test_inkml_roundtrip_with_text(tmp_path):
    path = write_ink(SAMPLE, tmp_path / "a.inkml")
    back = read_ink(path)
    assert back.text == "hi" and back.metadata["textField"] == "hi"
    assert [list(s) for s in back] == [list(s) for s in SAMPLE]
    assert InkMLReader().read_annotations(path) == {"textField": "hi"}


def test_inkml_reads_the_notebook_layout(tmp_path):
    path = tmp_path / "x.inkml"
    path.write_text(
        '<ink xmlns="http://www.w3.org/2003/InkML"><annotation type="textField">yo</annotation>'
        '<trace>1,2 3,4 5,6</trace><trace> </trace><trace>7,8</trace></ink>')
    ink = read_ink(path)
    assert len(ink) == 2 and ink.text == "yo" and list(ink[0]) == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]


def test_svg_export(tmp_path):
    path = write_ink(SAMPLE, tmp_path / "a.svg", stroke_width=2, size=(224, 224))
    root = ET.parse(path).getroot()
    paths = root.findall(".//{http://www.w3.org/2000/svg}path")
    assert len(paths) == 2 and paths[0].get("d") == "M 1.5 2.25 L 3 4"
    assert root.get("viewBox") == "0 0 224 224"
    fitted = SVGWriter().to_string(SAMPLE)  # fits to bounds + padding
    assert 'viewBox="-2.5 -1.75' in fitted
    assert "M 5 5 L 5 5" in SVGWriter().to_string(Ink([Stroke([(5, 5)])]))  # lone point -> dot
    ET.fromstring(SVGWriter().to_string(Ink()))  # empty ink is still valid XML


def test_format_dispatch_errors(tmp_path):
    with pytest.raises(RegistryError, match="cannot read"):
        read_ink(tmp_path / "x.unknown")
    with pytest.raises(RegistryError, match="cannot write"):
        write_ink(SAMPLE, tmp_path / "x.unknown")


def test_registry_basics_and_extension():
    reg: Registry = Registry("thing")

    @reg.register("a", "alias")
    class A:
        pass

    assert reg.get("ALIAS") is A and isinstance(reg.create("a"), A) and "a" in reg
    assert reg.names() == ["a", "alias"] and len(reg) == 2

    with pytest.raises(RegistryError):
        reg.register("a")(type("B", (), {}))
    reg.register("a")(A)  # re-registering the same class is fine
    reg.register("a", replace=True)(type("C", (), {}))
    with pytest.raises(RegistryError, match="registered: a, alias"):
        reg.get("missing")
    with pytest.raises(ValueError):
        reg.register()
    reg.unregister("alias")
    assert "alias" not in reg


def test_builtin_registrations():
    assert {"tesseract", "doctr", "whole"} <= set(segmenters.names())
    assert ".inkml" in ink_readers and {".inkml", ".svg"} <= set(ink_writers.names())


def test_custom_components_plug_in(page_image):
    @segmenters.register("test-grid")
    class GridSegmenter(WordSegmenter):
        def segment(self, image):
            return [WordBox(BoundingBox(0, 0, 100, 100)), WordBox(BoundingBox(100, 0, 100, 100))]

    @ink_readers.register(".fake")
    class FakeReader(InkReader):
        def read(self, path):
            return Ink([Stroke([(0, 0), (1, 1)])])

    try:
        kit = InkSight(InkSightModel(FakeBackend()), segmenter="test-grid")
        assert len(kit.convert_page(page_image).words) == 2
        assert len(read_ink("whatever.fake")) == 1
    finally:  # keep the global registries clean for other tests
        segmenters.unregister("test-grid")
        ink_readers.unregister(".fake")


def test_configure_tensorflow_is_idempotent():
    env: dict[str, str] = {"XLA_FLAGS": "--foo=1"}
    assert configure_tensorflow(env) is True
    assert env["XLA_FLAGS"].startswith("--foo=1 ") and all(f in env["XLA_FLAGS"] for f in XLA_FLAGS)
    snapshot = dict(env)
    assert configure_tensorflow(env) is False and env == snapshot
    fresh: dict[str, str] = {}
    configure_tensorflow(fresh)
    assert fresh["XLA_FLAGS"] == " ".join(XLA_FLAGS)


def test_require_gives_actionable_error():
    assert require("json").__name__ == "json"
    with pytest.raises(MissingDependencyError, match=r'inksight-api\[tf\]') as info:
        require("definitely_not_installed_xyz", extra="tf")
    assert isinstance(info.value, ImportError) and info.value.__cause__ is not None
