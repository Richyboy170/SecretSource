"""What one run of the tool cost: wall time, CPU time and peak RAM.

    import runcost
    ...
    stats = runcost.measure()          # numbers, for the manifest
    runcost.report(stats)              # prints them, at most once per run

The clocks start when this module is imported, not when `measure()` is called,
so a run is timed from its first line rather than from wherever someone
remembered to start a stopwatch. Import it early for that reason.

Peak RAM is the operating system's own process-lifetime high-water mark, not a
sample: memory that was allocated and freed before the run ended still shows up,
and no sampling thread is needed to catch it. `psutil` reports the same number,
but this is forty lines of `ctypes` against a tool that ships two requirements,
so it stays dependency-free — and where a platform will not answer, the fields
come back `None` and print as `n/a` rather than raising.

What is NOT counted: other processes. `time.process_time()` is this process
only, across all its threads. Work done elsewhere on your behalf — Ollama
answering an expansion request in the semantic tool, an API server doing the
search — is invisible here, so a run that looks cheap was not necessarily cheap.

This file is duplicated verbatim in `source-semantic-tool/`, for the same reason
that tool's download helpers are copies rather than imports: each folder has to
be movable on its own.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

# Import-time, so the measurement covers argument parsing, module loading and
# everything after it. perf_counter for wall (monotonic, like the rate limiters);
# process_time for CPU, which counts every thread of this process — including the
# native ones a torch model spawns inside encode().
_WALL_START = time.perf_counter()
_CPU_START = time.process_time()

_REPORTED = False


# --------------------------------------------------------------------------- #
# Resident memory, per platform
# --------------------------------------------------------------------------- #

def _windows_memory() -> tuple[int, int]:
    """(peak, current) working set, from K32GetProcessMemoryInfo."""
    import ctypes
    from ctypes import wintypes

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    # The K32-prefixed export lives in kernel32 itself on Windows 7 and later,
    # which avoids the psapi.dll-versus-psapi-forwarder mess entirely.
    query = kernel32.K32GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]
    query.restype = wintypes.BOOL

    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError(ctypes.get_last_error(), "K32GetProcessMemoryInfo failed")
    return counters.PeakWorkingSetSize, counters.WorkingSetSize


def _linux_memory() -> tuple[int, int]:
    """(peak, current) resident set, from /proc/self/status.

    Read from /proc rather than resource.getrusage(), whose ru_maxrss unit is
    kilobytes here and bytes on macOS — an ambiguity that silently reports the
    wrong figure by a factor of 1024.
    """
    found: dict[str, int] = {}
    with open("/proc/self/status", encoding="ascii") as fh:
        for line in fh:
            name, _, rest = line.partition(":")
            if name in ("VmHWM", "VmRSS"):
                found[name] = int(rest.split()[0]) * 1024
    return found["VmHWM"], found["VmRSS"]


def _memory() -> tuple[int | None, int | None]:
    """(peak, current) resident bytes, or Nones where the platform will not say.

    Never raises. A missing counter degrades the report to `n/a`, the same way
    the semantic tool runs without Ollama instead of failing: this is
    instrumentation, and instrumentation must not be able to fail a run.
    """
    try:
        if sys.platform == "win32":
            return _windows_memory()
        if sys.platform.startswith("linux"):
            return _linux_memory()
        if sys.platform == "darwin":
            import resource  # noqa: PLC0415 — absent on Windows, so not top-level
            # ru_maxrss is bytes on macOS. There is no cheap current-RSS here.
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, None
    except Exception:
        pass
    return None, None


# --------------------------------------------------------------------------- #

def _seconds(value: float) -> str:
    """Two decimals under ten seconds, none above: a 0.04s argument error and a
    191s batch run both want their significant digits, and neither wants the
    other's."""
    return f"{value:.2f}s" if value < 10 else f"{value:.0f}s"


def _bytes(size: int | None) -> str:
    if size is None:
        return "n/a"
    mb = size / 1_048_576
    return f"{mb / 1024:.2f} GB" if mb >= 1024 else f"{mb:.0f} MB"


@dataclass
class Usage:
    wall_seconds: float
    cpu_seconds: float
    # Of ONE core. Above 100 means several were busy at once (torch encodes on
    # every core it can find); far below 100 is the normal shape of this tool,
    # which spends most of a run waiting on other people's APIs.
    cpu_percent: float | None
    peak_rss_bytes: int | None
    rss_bytes: int | None

    def as_dict(self) -> dict:
        return {
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "cpu_percent": self.cpu_percent,
            "peak_rss_bytes": self.peak_rss_bytes,
            "rss_bytes": self.rss_bytes,
        }

    def line(self) -> str:
        """The one line a run prints.

        Prefixed "run cost", not "usage" — argparse already owns that word on a
        command line, and two different `usage:` lines in one failed run is a
        confusing thing to read.
        """
        cpu = f"{_seconds(self.cpu_seconds)} CPU"
        if self.cpu_percent is not None:
            cpu += f" ({self.cpu_percent:.0f}% of one core)"
        return (f"run cost: {_seconds(self.wall_seconds)} wall  {cpu}  "
                f"peak RAM {_bytes(self.peak_rss_bytes)}"
                f"  (now {_bytes(self.rss_bytes)})")


def measure() -> Usage:
    """Everything spent since this module was imported."""
    # Rounded first, then divided, so the percentage is the one a reader gets
    # back by dividing the two fields the manifest actually stores.
    wall = round(time.perf_counter() - _WALL_START, 3)
    cpu = round(time.process_time() - _CPU_START, 3)
    peak, current = _memory()
    return Usage(
        wall_seconds=wall,
        cpu_seconds=cpu,
        cpu_percent=round(100 * cpu / wall, 1) if wall > 0 else None,
        peak_rss_bytes=peak,
        rss_bytes=current,
    )


def report(stats: Usage | None = None) -> Usage:
    """Print the run-cost line, at most once per run.

    A runner prints the same snapshot it wrote to the manifest, then calls this
    again from a `finally` to cover the paths that never got that far — a bad
    command line, a dry run, Ctrl-C. The second call is a no-op, so those paths
    report exactly once and a normal run does not report twice.

    "Once" is per process, not per call site: the flag is module state, so a
    caller that has already reported will get silence from anywhere else.

    Printing cannot fail the run either. A `finally` block that raised would
    replace whatever exception sent it there — and this line is worth less than
    the traceback it would swallow. Piping the tool into `head` is enough to hit
    that (`BrokenPipeError`), so the write is guarded rather than trusted.
    """
    global _REPORTED
    stats = stats or measure()
    if not _REPORTED:
        _REPORTED = True
        try:
            print(stats.line())
        except Exception:
            pass
    return stats
