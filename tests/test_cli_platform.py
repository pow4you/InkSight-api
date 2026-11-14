import json

import pytest
from PIL import Image

from conftest import FakeBackend, FixedSegmenter
from inksight_api import Ink, InkSight, InkSightModel, Stroke, read_ink, write_ink
from inksight_api.cli import main


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "in.png"
    Image.new("RGB", (400, 300), "white").save(path)
    return str(path)


def fake_factory(two_words, calls):
    def factory(source=None, **kwargs):
        calls.append({"source": source, **kwargs})
        backend = FakeBackend()
        calls.append(backend)
        return InkSight(InkSightModel(backend), segmenter=FixedSegmenter(two_words))
    return factory


def test_json_roundtrip(tmp_path):
    ink = Ink([Stroke([(1.23456, 2), (3, 4)])], text="hi")
    back = read_ink(write_ink(ink, tmp_path / "a.json"))
    assert back.text == "hi" and list(back[0]) == [(1.235, 2.0), (3.0, 4.0)]


def test_cli_word_prints_json_and_defaults_to_vanilla(image_path, two_words, capsys):
    calls = []
    assert main([image_path], kit_factory=fake_factory(two_words, calls)) == 0
    out = capsys.readouterr()
    data = json.loads(out.out)
    assert len(data["strokes"]) == 1 and len(data["strokes"][0]) == 3
    assert calls[0] == {"source": None, "segmenter": "doctr", "batch_size": 32}
    assert calls[1].calls == [["Derender the ink."]]
    assert "1 strokes from 1 word(s)" in out.err


def test_cli_page_writes_file_by_suffix(image_path, two_words, tmp_path, capsys):
    target = tmp_path / "out" / "page.svg"
    code = main([image_path, "--page", "--mode", "r+d", "-o", str(target), "--segmenter", "tesseract"],
                kit_factory=fake_factory(two_words, []))
    assert code == 0 and target.read_text().count("<path") == 2
    assert capsys.readouterr().out == ""  # nothing on stdout when writing a file


def test_cli_text_guided_word(image_path, two_words):
    calls = []
    main([image_path, "--mode", "d+t", "--text", "hello"], kit_factory=fake_factory(two_words, calls))
    assert calls[1].calls == [["Derender the ink: hello"]]


def test_cli_reports_errors_without_traceback(image_path, two_words, tmp_path, capsys):
    factory = fake_factory(two_words, [])
    assert main([image_path, "--mode", "d+t"], kit_factory=factory) == 1  # d+t without text
    assert "needs the word's text" in capsys.readouterr().err
    assert main([str(tmp_path / "missing.png")], kit_factory=factory) == 1
    assert main([image_path, "-o", str(tmp_path / "x.unknown")], kit_factory=factory) == 1
    with pytest.raises(SystemExit):
        main([image_path, "--page", "--text", "x"], kit_factory=factory)


def test_cli_passes_model_source(image_path, two_words):
    calls = []
    main([image_path, "--model", "small-p-cpu.zip"], kit_factory=fake_factory(two_words, calls))
    assert calls[0]["source"] == "small-p-cpu.zip"
