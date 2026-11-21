"""Run the model inside WSL - e.g. on an AMD GPU through ROCm, from Windows.

TensorFlow's Windows wheels are CPU-only, and Docker Desktop has no AMD GPU
passthrough.  AMD's ROCm stack does run inside WSL 2 (Radeon RX 7900 / 9000
series and the matching Radeon PRO cards), with an official TensorFlow 2.20
wheel - so the GPU route on Windows with an AMD card is: TensorFlow in a WSL
distro, the model program started there through ``wsl.exe``.
``python -m inksight build --rocm`` (``inksight/scripts/setup-rocm-wsl.sh``) prepares such a distro.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path, PureWindowsPath
from typing import Dict, IO, List, Optional, Sequence

from ..errors import BackendNotAvailableError, VectorizationError
from ..paths import PACKAGE_DIR
from .base import Runner, RunResult
from .spool import ServerStopped, submit_job

#: Seconds allowed for the availability probe.
_PROBE_TIMEOUT = 60

#: Runs ``inksight.worker`` from the package folder given as the first argument,
#: without adding anything else to ``sys.path``.
_BOOTSTRAP = (
    "import importlib.util, sys; d = sys.argv[1]; "
    "s = importlib.util.spec_from_file_location('inksight', d + '/__init__.py', "
    "submodule_search_locations=[d]); m = importlib.util.module_from_spec(s); "
    "sys.modules['inksight'] = m; s.loader.exec_module(m); "
    "sys.argv = ['inksight-model'] + sys.argv[2:]; "
    "from inksight.worker import cli; cli()"
)


def to_wsl_path(path: "os.PathLike[str] | str") -> str:
    """``C:\\Users\\me\\x`` -> ``/mnt/c/Users/me/x`` (drive paths only; WSL's default mount)."""
    windows = PureWindowsPath(os.path.abspath(path))
    if not windows.drive or not windows.drive.endswith(":"):
        raise BackendNotAvailableError(f"cannot reach {path} from WSL (not on a drive letter)")
    return "/mnt/" + windows.drive[0].lower() + "/" + "/".join(windows.parts[1:])


class WslRunner(Runner):
    """Executes ``inksight-model`` with a Python interpreter inside a WSL distro.

    The package's own source is used from the Windows side (via ``/mnt/c/...``),
    so only TensorFlow, sentencepiece, NumPy and Pillow have to be installed in
    the distro.  Without ``model_dir`` the bundled model is unpacked once into
    the distro's own ``~/.local/share/inksight`` (a Linux filesystem, which
    loads far faster than ``/mnt/c``).

    Args:
        distro: WSL distribution (``wsl -d``).  Default: ``$INKSIGHT_WSL_DISTRO``,
            else the first distro whose ``python`` can import TensorFlow - WSL's
            default distribution first, then the others.  Docker Desktop's own
            ``docker-desktop`` distros are never chosen (installing Docker Desktop
            often makes one of them the default).
        python: Interpreter inside the distro.  Default: ``$INKSIGHT_WSL_PYTHON``,
            else ``/opt/inksight-rocm/bin/python`` (what the setup script creates).
        user: Linux user to run as (``wsl -u``); default: the distro's default user.
        model_dir: SavedModel folder *as a path inside WSL*.
        persistent: Keep one ``inksight-model --serve`` process running in WSL
            and send every call to it, so the model is loaded and XLA-compiled
            once (with the GPU that is the difference between ~35 s and ~2 s
            for a word).  Default: ``$INKSIGHT_WSL_PERSISTENT`` (``1`` = on),
            else off.
        idle_timeout: Seconds without work after which a persistent server exits.
        env: Extra environment variables for the program.
        wsl: The ``wsl.exe`` to call.
    """

    name = "wsl"

    def __init__(self, distro: Optional[str] = None, python: Optional[str] = None, *,
                 user: Optional[str] = None, model_dir: Optional[str] = None,
                 persistent: Optional[bool] = None, idle_timeout: Optional[float] = 900,
                 env: Optional[Dict[str, str]] = None, wsl: str = "wsl.exe") -> None:
        self.distro = distro or os.environ.get("INKSIGHT_WSL_DISTRO") or None
        self.python = python or os.environ.get("INKSIGHT_WSL_PYTHON") or "/opt/inksight-rocm/bin/python"
        self.user = user
        self.model_dir = model_dir
        if persistent is None:
            persistent = os.environ.get("INKSIGHT_WSL_PERSISTENT", "") not in ("", "0", "false")
        self.persistent = persistent
        self.idle_timeout = idle_timeout
        self.env = dict(env or {})
        self.wsl = wsl
        self._available: Optional[bool] = None
        self._server: Optional[subprocess.Popen] = None
        self._server_log: Optional[IO[bytes]] = None
        self._spool: Optional[Path] = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------------- command
    def command(self, program_args: Sequence[str], cwd: Optional[Path] = None) -> List[str]:
        """The ``wsl.exe ...`` command line that runs the model program with ``program_args``."""
        command = [self.wsl]
        if self.distro:
            command += ["-d", self.distro]
        if self.user:
            command += ["-u", self.user]
        if cwd is not None:
            command += ["--cd", to_wsl_path(cwd)]
        variables = {"TF_CPP_MIN_LOG_LEVEL": "2", **self.env}
        if self.model_dir:
            variables["INKSIGHT_MODEL_DIR"] = self.model_dir
        # Load the package by its own folder.  Putting its parent on PYTHONPATH
        # would, for an installed copy, expose the whole Windows site-packages
        # (Windows builds of numpy, Pillow, ...) to the Linux interpreter.
        command += ["--", "env", *(f"{k}={v}" for k, v in variables.items()),
                    self.python, "-c", _BOOTSTRAP, to_wsl_path(PACKAGE_DIR), *program_args]
        return command

    # ----------------------------------------------------------- availability
    def is_available(self, tool: str = "inksight-model") -> bool:
        """WSL exists here and a distro's Python can import TensorFlow (probed once).

        Without an explicit distro this also settles which distro is used.
        """
        if tool != "inksight-model" or sys.platform != "win32" or shutil.which(self.wsl) is None:
            return False
        if self._available is None:
            if self.distro:
                self._available = self._probe(self.distro)
            else:
                for candidate in self._candidate_distros():
                    if self._probe(candidate):
                        self.distro, self._available = candidate, True
                        break
                else:
                    self._available = False
        return self._available

    def ensure_available(self) -> None:
        """Raise a clear error up front instead of a failure deep inside a run."""
        if not self.is_available():
            where = f"WSL distro {self.distro!r}" if self.distro else "any WSL distro"
            raise BackendNotAvailableError(
                f"{self.python} with TensorFlow was not found in {where}. Run "
                "`python -m inksight build --rocm`, or set INKSIGHT_WSL_DISTRO / "
                "INKSIGHT_WSL_PYTHON (distros: "
                f"{', '.join(self.list_distros()) or 'none'}).")

    def list_distros(self) -> List[str]:
        """Installed WSL distributions, WSL's default first."""
        try:
            listed = subprocess.run([self.wsl, "--list", "--quiet"], stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, timeout=_PROBE_TIMEOUT).stdout
        except (OSError, subprocess.TimeoutExpired):
            return []
        # wsl.exe writes UTF-16 to pipes unless WSL_UTF8=1 is set.
        text = listed.decode("utf-16-le", errors="ignore") if b"\0" in listed else \
            listed.decode("utf-8", errors="ignore")
        return [line.strip() for line in text.replace("\0", "").splitlines() if line.strip()]

    def _candidate_distros(self) -> List[str]:
        return [name for name in self.list_distros() if not name.lower().startswith("docker-desktop")]

    def _probe(self, distro: str) -> bool:
        probe = [self.wsl, "-d", distro, *(["-u", self.user] if self.user else []), "--",
                 self.python, "-c",
                 "import importlib.util as u, sys; sys.exit(u.find_spec('tensorflow') is None)"]
        try:
            return subprocess.run(probe, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  timeout=_PROBE_TIMEOUT).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            return False

    # -------------------------------------------------------------------- run
    def run(self, tool: str, args: Sequence[str], workdir: Path,
            timeout: Optional[float] = None) -> RunResult:
        if tool != "inksight-model":
            raise BackendNotAvailableError(f"the WSL runner only runs 'inksight-model', not '{tool}'")
        self.ensure_available()
        if self.persistent:
            return self._run_on_server(tool, args, Path(workdir), timeout)
        command = self.command([*map(str, args)], cwd=Path(workdir))
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, timeout=timeout, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                       errors="replace")
        except FileNotFoundError as exc:
            raise BackendNotAvailableError(f"'{self.wsl}' is not installed") from exc
        except subprocess.TimeoutExpired as exc:
            partial = exc.output if isinstance(exc.output, str) else ""
            raise VectorizationError(f"'{tool}' did not finish within {timeout:g} s", partial) from exc
        return RunResult(command, completed.returncode, completed.stdout or "",
                         time.perf_counter() - started)

    # --------------------------------------------------------- persistent mode
    def start_server(self) -> None:
        """Start the persistent ``--serve`` process now (it loads the model)."""
        with self._lock:
            if self._server is not None and self._server.poll() is None:
                return
            self._stop_server_locked()
            self._spool = Path(tempfile.mkdtemp(prefix="inksight-spool-"))
            program = ["--serve", to_wsl_path(self._spool)]
            if self.idle_timeout is not None:
                program += ["--idle-timeout", f"{self.idle_timeout:g}"]
            self._server_log = open(self._spool.with_name(self._spool.name + ".log"), "wb")
            try:
                self._server = subprocess.Popen(self.command(program), stdin=subprocess.DEVNULL,
                                                stdout=self._server_log, stderr=subprocess.STDOUT)
            except FileNotFoundError as exc:
                self._stop_server_locked()
                raise BackendNotAvailableError(f"'{self.wsl}' is not installed") from exc
            atexit.register(self.close)

    def close(self) -> None:
        """Stop the persistent process (if any) and remove its spool directory."""
        with self._lock:
            self._stop_server_locked()

    def __enter__(self) -> "WslRunner":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _stop_server_locked(self) -> None:
        if self._server is not None:
            if self._spool is not None:
                (self._spool / ".stop").touch()
            try:
                self._server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server.kill()
            self._server = None
        if self._server_log is not None:
            name = self._server_log.name
            self._server_log.close()
            self._server_log = None
            Path(name).unlink(missing_ok=True)
        if self._spool is not None:
            shutil.rmtree(self._spool, ignore_errors=True)
            self._spool = None

    def _server_output(self) -> str:
        if self._server_log is None:
            return ""
        self._server_log.flush()
        text = Path(self._server_log.name).read_text(encoding="utf-8", errors="replace")
        return "\n".join(text.splitlines()[-40:])

    def _run_on_server(self, tool: str, args: Sequence[str], workdir: Path,
                       timeout: Optional[float]) -> RunResult:
        self.start_server()
        spool = self._spool
        assert spool is not None
        command = self.command(["--serve", to_wsl_path(spool)])          # for messages only
        started = time.perf_counter()
        alive = lambda: self._server is not None and self._server.poll() is None  # noqa: E731
        try:
            code, log = submit_job(spool, workdir, timeout=timeout, alive=alive, what=tool)
        except VectorizationError:
            self.close()
            raise
        except ServerStopped:
            output = self._server_output()
            self.close()
            return RunResult(command, 1, "the inksight server in WSL stopped:\n" + output,
                             time.perf_counter() - started)
        return RunResult(command, code, log, time.perf_counter() - started)

    def describe(self) -> str:
        where = f"WSL ({self.distro or 'default distro'}, {self.python})"
        return where + (" persistent" if self.persistent else "")
