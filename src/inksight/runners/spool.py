"""Client side of ``inksight-model --serve``: hand a job to a running server and wait.

The server (see :func:`inksight.worker.serve`) watches a spool directory.  A
job is a sub-directory holding the staged inputs; it is picked up once a
``.ready`` marker appears, and finished when ``.done`` does (next to
``output.json``, ``log.txt`` and ``exitcode``).  Runners that keep a server
alive - in a Docker container, in WSL, ... - only differ in how they start it
and how they tell whether it is still running.
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Callable, Optional, Tuple

from ..errors import VectorizationError

#: Seconds between liveness checks of the server while waiting.
LIVENESS_INTERVAL = 5.0


class ServerStopped(Exception):
    """The server went away before finishing the job."""


def submit_job(spool: Path, workdir: Path, *, timeout: Optional[float],
               alive: Callable[[], bool], what: str = "inksight-model",
               poll: float = 0.05) -> Tuple[int, str]:
    """Copy ``workdir`` into ``spool`` as a job, wait for it, copy ``output.json`` back.

    Args:
        spool: The directory the server watches.
        workdir: Staged inputs (``job.json`` and the word images).
        timeout: Seconds to wait at most, or None.
        alive: Returns False once the server has stopped (checked every few seconds).
        what: Program name for messages.

    Returns:
        The job's exit code and log.

    Raises:
        VectorizationError: The job did not finish within ``timeout``.
        ServerStopped: The server stopped while the job was pending.
    """
    job_dir = spool / uuid.uuid4().hex[:12]
    started = time.perf_counter()
    shutil.copytree(workdir, job_dir)
    (job_dir / ".ready").touch()
    checked = time.monotonic()
    try:
        while not (job_dir / ".done").exists():
            if timeout is not None and time.perf_counter() - started > timeout:
                raise VectorizationError(f"'{what}' did not finish within {timeout:g} s")
            if time.monotonic() - checked > LIVENESS_INTERVAL:
                checked = time.monotonic()
                if not alive():
                    raise ServerStopped()
            time.sleep(poll)
        if (job_dir / "output.json").is_file():
            shutil.copy2(job_dir / "output.json", workdir / "output.json")
        code = int((job_dir / "exitcode").read_text().strip() or 1)
        log = (job_dir / "log.txt").read_text(encoding="utf-8", errors="replace")
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)
    return code, log
