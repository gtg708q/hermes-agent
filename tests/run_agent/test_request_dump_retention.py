"""Cross-process request-dump retention regression tests."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path


def _write_dump_worker(logs_dir_raw: str, index: int) -> None:
    from agent.agent_runtime_helpers import _write_bounded_request_dump

    logs_dir = Path(logs_dir_raw)
    payload = {
        "timestamp": str(index),
        "fingerprint": f"fp-{index}",
        "request": {"body": {"index": index}},
    }
    _write_bounded_request_dump(
        logs_dir=logs_dir,
        dump_file=logs_dir / f"request_dump_worker_{index}.json",
        payload=payload,
        max_files=2,
        max_bytes=4096,
        ttl_days=7,
        dedup_seconds=0,
    )


def test_request_dump_count_cap_is_cross_process_safe(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(target=_write_dump_worker, args=(str(tmp_path), index))
        for index in range(8)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    dumps = list(tmp_path.glob("request_dump_*.json"))
    assert len(dumps) == 2
    assert all(json.loads(path.read_text(encoding="utf-8")) for path in dumps)
