"""
Shared concurrency helper for the eval harnesses (run_full_eval.py --
Phase 1, run_full_eval_v2.py -- Phase 2).

Local Ollama calls are I/O-bound (the model runs server-side; the client
just waits on the HTTP response), so a thread pool is enough -- no
multiprocessing needed. Default is still fully sequential (max_workers=1),
so existing single-threaded invocations of either eval script are
unaffected unless --workers/--models is passed explicitly.

True parallel throughput also depends on the Ollama SERVER's own
concurrency settings (OLLAMA_NUM_PARALLEL, OLLAMA_MAX_LOADED_MODELS env vars
on the machine running `ollama serve`) -- without raising those, concurrent
requests to the same model just queue server-side instead of running
side-by-side. See docs/phase2/parallel-execution.md.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_rows_parallel(
    rows: List[T],
    worker_fn: Callable[[int, T], R],
    max_workers: int = 1,
) -> List[R]:
    """Apply worker_fn(index, row) to every row, optionally concurrently.

    Returns results in the SAME ORDER as `rows`, regardless of completion
    order or worker count -- callers can treat this exactly like
    `[worker_fn(i, r) for i, r in enumerate(rows)]` regardless of
    max_workers. worker_fn is expected to catch its own exceptions and
    encode failures in its return value (matching the existing
    try/except-and-record pattern already used per-row in the eval
    scripts); an exception that escapes worker_fn will surface via
    Future.result() and abort the whole run, same as it would in a plain
    sequential loop.
    """
    if max_workers <= 1 or len(rows) <= 1:
        return [worker_fn(i, row) for i, row in enumerate(rows)]

    results: List[Optional[R]] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker_fn, i, row): i for i, row in enumerate(rows)}
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
    return results  # type: ignore[return-value]


class LockedCheckpointWriter:
    """Thread-safe wrapper around the append-and-flush-per-row checkpoint
    pattern already used by run_full_eval.py / run_full_eval_v2.py. Safe to
    share across worker threads submitted to the same ThreadPoolExecutor."""

    def __init__(self, path):
        self._file = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write_row(self, row: dict) -> None:
        line = json.dumps(row) + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()

    def close(self) -> None:
        self._file.close()
