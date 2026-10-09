import logging
import time
import uuid
from collections import Counter, deque

from .persistence import digest

logger = logging.getLogger("raglab")


class Telemetry:
    def __init__(self):
        self.started = time.monotonic()
        self.counts: Counter = Counter()
        self.traces: deque = deque(maxlen=200)

    def trace(self, owner, question, version, stages, degradation, run_id=None):
        trace_id = uuid.uuid4().hex
        self.traces.append(
            {
                "trace_id": trace_id,
                "owner": owner,
                "query_digest": digest(question),
                "index_version": version,
                "stages_ms": stages,
                "degradation": list(degradation),
                "run_id": run_id,
            }
        )
        self.counts["answers"] += 1
        if degradation:
            self.counts["degraded"] += 1
        logger.info("answer trace=%s run=%s degraded=%s", trace_id, run_id, bool(degradation))
        return trace_id

    def metrics(self):
        # 指标名只来自服务端代码，不接受任意的身份/问题/ID 标签。
        names = (
            "answers",
            "degraded",
            "runs_submitted",
            "runs_completed",
            "runs_cancelled",
            "timeouts",
            "errors",
        )
        return "\n".join(f"raglab_{name}_total {self.counts[name]}" for name in names) + "\n"

    def for_owner(self, owner):
        return [
            {k: v for k, v in x.items() if k != "owner"} for x in self.traces if x["owner"] == owner
        ]
