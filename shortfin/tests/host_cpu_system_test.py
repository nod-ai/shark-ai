# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import pytest
import re
import sys

import shortfin as sf


def test_create_host_cpu_system_defaults():
    sc = sf.host.CPUSystemBuilder()
    with sc.create_system() as ls:
        print(f"DEFAULT LOCAL SYSTEM:", ls)
        print("\n".join(repr(d) for d in ls.devices))
        assert len(ls.devices) > 0


@pytest.mark.skip(
    reason="IREE UNIMPLEMENTED: threadless donate-only executor mode not yet implemented with hostcpu_topology_max_group_count=2"
)
def test_create_host_cpu_system_topology_nodes_all():
    sc = sf.host.CPUSystemBuilder(
        hostcpu_topology_nodes="all", hostcpu_topology_max_group_count=2
    )
    with sc.create_system() as ls:
        print(f"NODES ALL LOCAL SYSTEM:", ls)
        print("\n".join(repr(d) for d in ls.devices))
        assert len(ls.devices) > 0


def test_create_host_cpu_system_topology_nodes_explicit():
    sc = sf.host.CPUSystemBuilder(
        hostcpu_topology_nodes="0,0", hostcpu_topology_max_group_count=2
    )
    with sc.create_system() as ls:
        print(f"NODES EXPLICIT LOCAL SYSTEM:", ls)
        print("\n".join(repr(d) for d in ls.devices))
        assert len(ls.devices) == 2


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Only detecting 1 device, check config setup from env vars?",
)
def test_create_host_cpu_system_env_vars():
    os.environ["SHORTFIN_HOSTCPU_TOPOLOGY_NODES"] = "0,0"
    os.environ["SHORTFIN_HOSTCPU_TOPOLOGY_MAX_GROUP_COUNT"] = "2"
    sc = sf.host.CPUSystemBuilder()
    with sc.create_system() as ls:
        print(f"ENV VARS LOCAL SYSTEM:", ls)
        print("\n".join(repr(d) for d in ls.devices))
        assert len(ls.devices) == 2


def test_create_host_cpu_system_allocators():
    pytest.skip("Setting allocators triggers LSAN leak. See #443")
    sc = sf.host.CPUSystemBuilder(hostcpu_allocators="caching;debug")
    assert sc.hostcpu_allocator_specs == ["caching", "debug"]
    with sc.create_system() as ls:
        pass


def test_create_host_cpu_system_unsupported_option():
    sc = sf.host.CPUSystemBuilder(unsupported="foobar")
    with pytest.raises(
        ValueError, match="Specified options were not used: unsupported"
    ):
        sc.create_system()


def test_system_ctor():
    with sf.System(
        "hostcpu", hostcpu_topology_nodes="0,0", hostcpu_topology_max_group_count=2
    ) as ls:
        print(f"NODES EXPLICIT LOCAL SYSTEM:", ls)
        print("\n".join(repr(d) for d in ls.devices))
        assert len(ls.devices) == 2


def test_system_ctor_unknown_type():
    with pytest.raises(
        ValueError,
        match=re.escape("System type 'NOTDEFINED' not known (available: hostcpu"),
    ):
        sf.System("NOTDEFINED")


def test_system_ctor_undef_error():
    with pytest.raises(ValueError, match="Specified options were not used: undef"):
        sf.System("hostcpu", undef=1)


def test_system_ctor_undef_warn():
    with sf.System("hostcpu", validate_undef=False, undef=1) as ls:
        ...
