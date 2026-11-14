"""InkML reader/writer (the format of the InkSight dataset release)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ..core.ink import Ink, Stroke
from .base import InkReader, InkWriter, ink_readers, ink_writers

INKML_NS = "http://www.w3.org/2003/InkML"
_NS = {"inkml": INKML_NS}


def format_number(value: float) -> str:
    """Compact decimal text: at most 3 decimals, no trailing zeros."""
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


@ink_readers.register(".inkml")
class InkMLReader(InkReader):
    """Read ``<trace>`` elements (``"x,y x,y ..."``) as strokes.

    The ``<annotation type=...>`` entries land in ``ink.metadata`` and the
    ``textField`` annotation (the dataset's OCR / recognised text) in ``ink.text``.
    """

    def read(self, path: str | Path) -> Ink:
        root = ET.parse(path).getroot()
        strokes = []
        for trace in root.findall("inkml:trace", _NS):
            tokens = (trace.text or "").split()
            if tokens:
                strokes.append(Stroke([tuple(map(float, t.split(","))) for t in tokens]))
        annotations = self._annotations(root)
        return Ink(strokes, text=annotations.get("textField"), metadata=annotations)

    def read_annotations(self, path: str | Path) -> dict[str, str | None]:
        """Only the annotations (notebook: ``parse_inkml_annotations``)."""
        return self._annotations(ET.parse(path).getroot())

    @staticmethod
    def _annotations(root: ET.Element) -> dict[str, str | None]:
        return {a.get("type"): a.text for a in root.findall(f".//{{{INKML_NS}}}annotation")}


@ink_writers.register(".inkml")
class InkMLWriter(InkWriter):
    """Write a minimal InkML file that :class:`InkMLReader` reads back."""

    def write(self, ink: Ink, path: str | Path) -> Path:
        ET.register_namespace("", INKML_NS)
        root = ET.Element(f"{{{INKML_NS}}}ink")
        fields = dict(ink.metadata)
        if ink.text is not None:
            fields["textField"] = ink.text
        for key, value in fields.items():
            ET.SubElement(root, f"{{{INKML_NS}}}annotation", {"type": key}).text = value
        for stroke in ink:
            trace = ET.SubElement(root, f"{{{INKML_NS}}}trace")
            trace.text = " ".join(f"{format_number(x)},{format_number(y)}" for x, y in stroke)
        ET.indent(root)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
        return path
