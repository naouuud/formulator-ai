INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
BAD_REQUEST = "BAD_REQUEST"


class APIError(Exception):
    def __init__(self, internal_error_code: str, title: str, status: int, detail: str):
        self.internal_error_code = internal_error_code
        self.title = title
        self.status = status
        self.detail = detail
        super().__init__(f"{internal_error_code}: {detail}")
