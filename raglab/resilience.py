import asyncio
import time
from collections import OrderedDict

from .errors import LabError, TransientError


def remaining(deadline: float) -> float:
    value = deadline - time.time()
    if value <= 0:
        raise LabError("deadline_exceeded", "original total deadline exceeded", 408)
    return value


class Circuit:
    def __init__(self, threshold=3, cooldown=2.0):
        self.threshold, self.cooldown = threshold, cooldown
        self.failures, self.opened, self.probing = 0, 0.0, False
        self.generation = 0

    def enter(self):
        if self.failures < self.threshold:
            return (self.generation, False)
        if time.monotonic() - self.opened < self.cooldown or self.probing:
            raise LabError("circuit_open", "dependency circuit is open", 503)
        self.probing = True
        return (self.generation, True)

    def success(self, ticket=None):
        if ticket is not None and ticket[0] != self.generation:
            return
        self.failures, self.probing = 0, False

    def fail(self, ticket=None):
        if ticket is not None and ticket[0] != self.generation:
            return
        self.failures += 1
        self.opened, self.probing = time.monotonic(), False
        if self.failures >= self.threshold:
            self.generation += 1

    def cancel(self, ticket):
        if ticket[0] == self.generation and ticket[1]:
            self.probing = False


def transient(exc: Exception) -> bool:
    if isinstance(exc, TransientError):
        return True
    # 仅识别 provider 明确声明的瞬时错误；校验/认证类错误一律视为永久错误。
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    return isinstance(
        exc, (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
    )


async def guarded(call, deadline: float, circuit: Circuit, *, safe=True, attempts=2):
    ticket = circuit.enter()
    try:
        for attempt in range(attempts if safe else 1):
            try:
                async with asyncio.timeout(remaining(deadline)):
                    value = await call()
                circuit.success(ticket)
                return value
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not transient(exc):
                    raise
                if attempt + 1 >= (attempts if safe else 1):
                    raise
                await asyncio.sleep(min(0.02 * 2**attempt, remaining(deadline)))
        raise AssertionError("unreachable")
    except asyncio.CancelledError:
        circuit.cancel(ticket)
        raise
    except Exception as exc:
        if transient(exc):
            circuit.fail(ticket)
        else:
            circuit.cancel(ticket)
        raise


class AnswerCache:
    def __init__(self, capacity: int, max_flights: int = 10):
        self.capacity = capacity
        self.max_flights = max_flights
        self.values: OrderedDict[str, dict] = OrderedDict()
        self.flights: dict[str, asyncio.Task] = {}
        self.hits = 0

    async def get_or_compute(self, key, factory):
        if key in self.values:
            self.hits += 1
            self.values.move_to_end(key)
            return self.values[key]
        if key in self.flights:
            return await asyncio.shield(self.flights[key])
        if len(self.flights) >= self.max_flights:
            raise LabError("query_backpressure", "bounded query flight queue is full", 429)
        task = asyncio.create_task(factory())
        self.flights[key] = task
        try:
            value = await task
            self.values[key] = value
            while len(self.values) > self.capacity:
                self.values.popitem(last=False)
            return value
        finally:
            self.flights.pop(key, None)
