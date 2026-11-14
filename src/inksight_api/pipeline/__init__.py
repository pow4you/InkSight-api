"""Pipelines that wire segmentation, preprocessing and the model together."""

from .pipeline import InkPipeline, PageInkPipeline, WordInkPipeline
from .results import PipelineResult, WordResult

__all__ = ["InkPipeline", "PageInkPipeline", "PipelineResult", "WordInkPipeline", "WordResult"]
