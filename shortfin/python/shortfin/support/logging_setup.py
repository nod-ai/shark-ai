# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import os
import sys

from _shortfin import lib as _sfl

_LOG_FUNCTIONS = {
    logging.DEBUG: _sfl.log_debug,
    logging.INFO: _sfl.log_info,
    logging.WARNING: _sfl.log_warn,
    logging.ERROR: _sfl.log_error,
    logging.CRITICAL: _sfl.log_error,
}

logger = logging.getLogger("shortfin")
logger.propagate = False

SHORTFIN_APPS_LOG_LEVEL = getattr(
    logging, os.environ.get("SHORTFIN_APPS_LOG_LEVEL", "WARNING")
)


class NativeHandler(logging.Handler):
    def emit(self, record):
        formatted = self.format(record)
        f = _LOG_FUNCTIONS.get(record.levelno)
        if f is not None:
            f(formatted)


class NativeFormatter(logging.Formatter):
    def __init__(self):
        super().__init__("[%(filename)s:%(lineno)d] %(message)s")


native_handler = NativeHandler()
native_handler.setFormatter(NativeFormatter())

logger.setLevel(SHORTFIN_APPS_LOG_LEVEL)
logger.addHandler(native_handler)


def configure_main_logger(module_suffix: str = "__main__") -> logging.Logger:
    """Configures logging from a main entrypoint.
    Returns a logger that can be used for the main module itself.
    """
    logging.root.addHandler(native_handler)
    logging.root.setLevel(SHORTFIN_APPS_LOG_LEVEL)
    main_module = sys.modules["__main__"]
    return logging.getLogger(f"{main_module.__package__}.{module_suffix}")
