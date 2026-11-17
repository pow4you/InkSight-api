"""Shared fixtures.  Tests that run the model skip themselves when no runner is
available, so the plain unit tests always run.

``INKSIGHT_RUNNER=docker`` (or ``local``) forces the runner the model tests use.
"""

from __future__ import annotations

import pytest

from inksight.errors import BackendNotAvailableError
from inksight.paths import EXAMPLE_IMAGE, EXAMPLE_WORD
from inksight.runners import get_runner
from inksight.worker import TOOL


@pytest.fixture(scope="session")
def example_word():
    return EXAMPLE_WORD


@pytest.fixture(scope="session")
def example_page():
    return EXAMPLE_IMAGE


@pytest.fixture(scope="session")
def model_runner():
    """A runner that can execute the model, or skip."""
    try:
        runner = get_runner("auto", TOOL)
    except BackendNotAvailableError as exc:
        pytest.skip(f"no way to run the model: {exc}")
    return runner
