"""同步 SQL 外观层：原声明事务结束后不允许继续使用。"""

import math
import sqlite3

from .contracts import AgentError


class AttemptTransaction:
    """受信任钩子拿到的是这个受限接口，而不是原始连接。

    SQL 授权本身已限制可用的表与语句。这道额外防护处理 SQLite 冲突解析或
    触发器导致的隐式 ROLLBACK。它不是针对受信任 Python 访问宿主私有对象的沙箱。
    """

    def __init__(self, db: sqlite3.Connection):
        self._db = db
        self._lost = False
        self._enabled = True

    def _active(self):
        if self._lost or not self._enabled:
            raise AgentError("attempt_commit", "original claim transaction ended", 503)
        if not self._db.in_transaction:
            self._lost = True
        if self._lost:
            raise AgentError("attempt_commit", "original claim transaction ended", 503)

    def _invalidate(self):
        self._enabled = False

    def _parameters(self, parameters):
        iterator = iter(parameters)
        while True:
            self._active()
            try:
                item = next(iterator)
            except StopIteration:
                return
            # 受信任的参数生成器自身也可能执行 SQL。那次调用一旦结束事务，
            # 就绝不能再把下一项交给 SQLite。
            self._active()
            yield self._row(item)

    def _row(self, parameters):
        # sqlite 的适配器/__conform__ 可能在准备阶段与 SQLite 写入之间执行 Python。
        # 只有纯内建标量才允许进入驱动。
        self._active()
        row: tuple | dict
        if type(parameters) in {tuple, list}:
            row = tuple(parameters)
            values = row
        elif type(parameters) is dict and all(type(key) is str for key in parameters):
            row = dict(parameters)
            values = tuple(row.values())
        else:
            raise AgentError("attempt_commit", "SQL parameters must be plain rows", 503)
        if len(values) > 256 or any(
            type(value) not in {type(None), bool, int, float, str, bytes}
            or type(value) is float
            and not math.isfinite(value)
            or (type(value), sqlite3.PrepareProtocol) in sqlite3.adapters
            for value in values
        ):
            raise AgentError("attempt_commit", "SQL parameters must be plain finite scalars", 503)
        self._active()
        return row

    def _operation(self, method, *args):
        self._active()
        try:
            return method(*args)
        finally:
            if not self._db.in_transaction:
                self._lost = True

    @property
    def in_transaction(self):
        self._active()
        return True

    @property
    def total_changes(self):
        self._active()
        return self._db.total_changes

    def cursor(self):
        self._active()
        return AttemptCursor(self, self._db.cursor())

    def execute(self, sql, parameters=()):
        return self.cursor().execute(sql, parameters)

    def executemany(self, sql, parameters):
        return self.cursor().executemany(sql, parameters)

    def executescript(self, *args):
        raise AgentError("attempt_commit", "scripts cannot control the claim transaction", 503)

    def commit(self):
        raise AgentError("attempt_commit", "hook cannot commit the claim transaction", 503)

    def rollback(self):
        raise AgentError("attempt_commit", "hook cannot roll back the claim transaction", 503)


class AttemptCursor:
    def __init__(self, connection: AttemptTransaction, cursor: sqlite3.Cursor):
        self._connection, self._cursor = connection, cursor

    @property
    def connection(self):
        return self._connection

    @property
    def rowcount(self):
        self._connection._active()
        return self._cursor.rowcount

    @property
    def lastrowid(self):
        self._connection._active()
        return self._cursor.lastrowid

    @property
    def description(self):
        self._connection._active()
        return self._cursor.description

    def execute(self, sql, parameters=()):
        self._connection._operation(self._cursor.execute, sql, self._connection._row(parameters))
        return self

    def executemany(self, sql, parameters):
        self._connection._operation(
            self._cursor.executemany, sql, self._connection._parameters(parameters)
        )
        return self

    def executescript(self, *args):
        return self._connection.executescript(*args)

    def fetchone(self):
        return self._connection._operation(self._cursor.fetchone)

    def fetchmany(self, size=1):
        return self._connection._operation(self._cursor.fetchmany, size)

    def fetchall(self):
        return self._connection._operation(self._cursor.fetchall)

    def __iter__(self):
        return self

    def __next__(self):
        self._connection._active()
        return next(self._cursor)

    def close(self):
        self._cursor.close()
