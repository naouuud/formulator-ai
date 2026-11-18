"""Error class and internal error codes for agent."""

LLM_CONFIG_ERROR = "LLM_CONFIG_ERROR"
BAD_REQUEST_ERROR = "BAD_REQUEST_ERROR"
STRUCTURED_COMPLETION_ERROR = "STRUCTURED_COMPLETION_ERROR"


class AgentError(Exception):
    def __init__(self, error_code: str, detail: str):
        self.error_code = error_code
        self.detail = detail
        super().__init__(f"{error_code}: {detail}")


class LLMConfigError(AgentError):
    """Raised when required configuration is missing or malformed."""

    def __init__(self, detail):
        super().__init__(LLM_CONFIG_ERROR, detail)


class BadRequestError(AgentError):
    """Raised when a bad (malformed) request is sent to the LLM."""

    def __init__(self, detail):
        super().__init__(BAD_REQUEST_ERROR, detail)


class StructuredCompletionError(AgentError):
    """Raised when no valid structured output was produced within the budget."""

    def __init__(self, detail):
        super().__init__(STRUCTURED_COMPLETION_ERROR, detail)
