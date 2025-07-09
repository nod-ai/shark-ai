# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from _shortfin import lib as _sfl

# Set up logging.
import shortfin.support.logging_setup as _logging_setup

# Most classes from the native "local" namespace are aliased to the top
# level of the public API.
BaseProgramParameters = _sfl.local.BaseProgramParameters
CompletionEvent = _sfl.local.CompletionEvent
Device = _sfl.local.Device
Fiber = _sfl.local.Fiber
Message = _sfl.local.Message
Node = _sfl.local.Node
Process = _sfl.local.Process
Program = _sfl.local.Program
ProgramFunction = _sfl.local.ProgramFunction
ProgramIsolation = _sfl.local.ProgramIsolation
ProgramInvocation = _sfl.local.ProgramInvocation
ProgramInvocationFuture = _sfl.local.ProgramInvocationFuture
ProgramModule = _sfl.local.ProgramModule
Queue = _sfl.local.Queue
QueueReader = _sfl.local.QueueReader
QueueWriter = _sfl.local.QueueWriter
ScopedDevice = _sfl.local.ScopedDevice
StaticProgramParameters = _sfl.local.StaticProgramParameters
System = _sfl.local.System
SystemBuilder = _sfl.local.SystemBuilder
VoidFuture = _sfl.local.VoidFuture
Worker = _sfl.local.Worker

# Array is auto-imported.
from . import array

# LLM components.
llm = _sfl.llm

# System namespaces.
from . import amdgpu
from . import host

__all__ = [
    "BaseProgramParameters",
    "CompletionEvent",
    "Device",
    "Fiber",
    "Message",
    "Node",
    "Program",
    "ProgramFunction",
    "ProgramInvocation",
    "ProgramInvocationFuture",
    "ProgramModule",
    "Queue",
    "QueueReader",
    "QueueWriter",
    "ScopedDevice",
    "StaticProgramParameters",
    "System",
    "SystemBuilder",
    "VoidFuture",
    "Worker",
    # LLM components.
    "llm",
    # System namespaces.
    "amdgpu",
    "host",
]
