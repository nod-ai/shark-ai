from enum import Enum, auto


class ResponseErrorCodes(Enum):
    QUEUE_FULL = "QUEUE_FULL"
    INVALID_REQUEST_ARGS = "INVALID_REQUEST_ARGS"
