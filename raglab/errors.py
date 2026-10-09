class LabError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.status, self.retryable = code, message, status, retryable


class TransientError(LabError):
    def __init__(self, message: str = "temporary dependency failure"):
        super().__init__("transient", message, 503, True)


class UnknownEffect(LabError):
    def __init__(self):
        super().__init__(
            "needs_reconciliation", "effect or compensation is unknown; inspect receipt", 409
        )
