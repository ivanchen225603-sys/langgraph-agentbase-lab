"""进程内的依赖熔断器，带与代次（generation）绑定的半开票据。"""

import time
import uuid
from dataclasses import dataclass

from .contracts import AgentError


@dataclass(frozen=True)
class CircuitTicket:
    generation: int
    probe: str | None = None


class ToolCircuit:
    """每次逻辑执行一张票据，而非每次重试或每次缓存命中一张。

    访问在执行器的事件循环上同步进行。状态有意保持进程内；尝试次数与截止时间
    仍记录在持久化账本（ledger）中。
    """

    def __init__(self, failures: int, cooldown: float):
        self.threshold, self.cooldown = failures, cooldown
        self.failures = 0
        self.generation = 0
        self.open_until: float | None = None
        self.probe: str | None = None

    def acquire(self) -> CircuitTicket:
        if self.open_until is None:
            return CircuitTicket(self.generation)
        if time.monotonic() < self.open_until or self.probe is not None:
            raise AgentError("tool_circuit_open", "tool dependency circuit is open", 503)
        self.probe = uuid.uuid4().hex
        return CircuitTicket(self.generation, self.probe)

    def finish(self, ticket: CircuitTicket, outcome: str):
        if ticket.generation != self.generation or ticket.probe != self.probe:
            return
        if ticket.probe is not None:
            self.generation += 1
            self.probe = None
            if outcome == "success":
                self.failures = 0
                self.open_until = None
            elif outcome == "failure":
                self.open_until = time.monotonic() + self.cooldown
            # 取消/认领/Schema/策略类失败会释放探针并保留已流逝的冷却时间，
            # 从而允许稍后发起新的探针。
            return
        if outcome == "success":
            self.failures = 0
        elif outcome == "failure":
            self.failures += 1
            if self.failures >= self.threshold:
                self.generation += 1
                self.open_until = time.monotonic() + self.cooldown
