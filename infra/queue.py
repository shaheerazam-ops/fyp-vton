"""
Minimal queue abstraction — decouples the API tier (fast, stateless,
CPU) from the inference workers (slow, GPU-bound). Backed by Redis
here; swap for SQS/PubSub in other clouds without touching API/worker
code, since both only call `enqueue_job` / `consume_jobs`.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import redis

_client = redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
QUEUE_KEY = "vton:try_on_jobs"


def enqueue_job(job: dict) -> None:
    _client.rpush(QUEUE_KEY, json.dumps(job))


def consume_jobs(block_timeout_s: int = 5) -> Iterator[dict]:
    """Blocking generator — yields jobs as they arrive, one worker loop per process."""
    while True:
        item = _client.blpop(QUEUE_KEY, timeout=block_timeout_s)
        if item is None:
            continue
        _, payload = item
        yield json.loads(payload)
