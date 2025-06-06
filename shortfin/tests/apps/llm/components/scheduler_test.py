# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from shortfin_apps.llm.components.scheduler import Scheduler


class FakeBatcher:
    def __init__(self):
        self.msgs = []

    def submit(self, x):
        self.msgs.append(x)

    def pop(self):
        msgs = self.msgs
        self.msgs = []
        return msgs


def reserve_helper(scheduler, *, rid, count):
    batcher = FakeBatcher()
    scheduler.reserve_workitem(rid=rid, count=count, batcher=batcher)
    assert scheduler.handle_scheduler(batcher.pop()[0]) == True


def release_helper(scheduler, *, rid, count):
    batcher = FakeBatcher()
    scheduler.release_workitem(rid=rid, count=count, batcher=batcher)
    assert scheduler.handle_scheduler(batcher.pop()[0]) == True


def make_workload(rids):
    workload = {}
    running = 0
    for rid in rids:
        count = rids[rid]
        workload[rid] = [f"Task{i + running}" for i in range(count)]
        running = running + count

    return workload


# Checked that strobing will return workload when detected
def test_scheduler_unreserved_strobed():
    ideal_batch_size = 32
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    workload = make_workload({0: 4})
    to_schedule = scheduler.should_execute(pending=workload, strobe=0)
    assert len(to_schedule) == 0

    to_schedule = scheduler.should_execute(pending=workload, strobe=1)
    assert len(to_schedule) == 0

    to_schedule = scheduler.should_execute(pending=workload, strobe=2)
    assert len(to_schedule) == 1
    assert to_schedule[0] == workload[0]


# Check that a full ideal set is returned
def test_scheduler_unreserved_full():
    ideal_batch_size = 4
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    workload = make_workload({0: 4})
    to_schedule = scheduler.should_execute(pending=workload, strobe=0)
    assert len(to_schedule) == 1
    assert to_schedule[0] == to_schedule[0]


# Check that a subset is returned if overfilled
def test_scheduler_unreserved_overfull():
    ideal_batch_size = 3
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    pending = make_workload({0: 4})
    to_schedule = scheduler.should_execute(pending=pending, strobe=0)
    assert len(to_schedule) == 1
    assert to_schedule[0] == pending[0][:3]

    # Check we fill as many ideal batches as possible:
    pending = make_workload({0: 10})
    to_schedule = scheduler.should_execute(pending=pending, strobe=0)
    assert len(to_schedule) == 3
    assert to_schedule[0] == pending[0][:3]
    assert to_schedule[1] == pending[0][3:6]
    assert to_schedule[2] == pending[0][6:9]


# Check that if there is a reservation only the unreserved are executed:
def test_scheduler_unreserved_with_reservation():
    ideal_batch_size = 8
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    reserve_helper(scheduler, rid=1, count=5)

    pending = make_workload({0: 4, 1: 4})
    to_schedule = scheduler.should_execute(pending=pending, strobe=0)
    assert len(to_schedule) == 0

    to_schedule = scheduler.should_execute(pending=pending, strobe=2)
    assert len(to_schedule) == 1
    assert to_schedule[0] == pending[0]


# Check if reserved and all passed to execute the whole job:
def test_scheduler_reserved_basic():
    ideal_batch_size = 32
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    reserve_helper(scheduler, rid=0, count=5)
    workload = make_workload({0: 4})
    to_schedule = scheduler.should_execute(pending=workload, strobe=0)
    assert to_schedule == []

    to_schedule = scheduler.should_execute(pending=workload, strobe=2)
    assert to_schedule == []

    workload = make_workload({0: 5})
    to_schedule = scheduler.should_execute(pending=workload, strobe=2)
    assert len(to_schedule) == 1
    assert to_schedule[0] == workload[0]

    release_helper(scheduler, rid=0, count=1)
    workload = make_workload({0: 4})
    to_schedule = scheduler.should_execute(pending=workload, strobe=2)
    assert len(to_schedule) == 1
    assert to_schedule[0] == workload[0]


# Check if reserved and all passed to execute the whole job.
def test_scheduler_reserved_extra():
    ideal_batch_size = 7
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    reserve_helper(scheduler, rid=0, count=5)
    workload = make_workload({0: 5, 1: 3})
    to_schedule = scheduler.should_execute(pending=workload, strobe=2)
    assert len(to_schedule) == 1
    assert to_schedule[0] == workload[0]


# Reserve a job at that exceeds the max size, shhould be split between jobs.
def test_scheduler_reserved_too_big():
    ideal_batch_size = 5
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    reserve_helper(scheduler, rid=0, count=7)
    workload = make_workload({0: 7})
    to_schedule = scheduler.should_execute(pending=workload, strobe=2)
    assert len(to_schedule) == 2
    assert to_schedule[0] == workload[0][:5]
    assert to_schedule[1] == workload[0][5:]


# Check two reservations fall into the same bucket:
def test_scheduler_reserved_two_shared():
    ideal_batch_size = 10
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)
    batcher = FakeBatcher()

    # Include two separate reservations for scheduler
    reserve_helper(scheduler, rid=0, count=5)
    reserve_helper(scheduler, rid=1, count=5)

    # Check without full on either we do not submit
    workload = {j: [f"Task{i * 2 + j}" for i in range(4)] for j in range(2)}
    to_schedule = scheduler.should_execute(pending=workload, strobe=0)
    assert to_schedule == []

    # Check with single RID full we still do not submit:
    workload = {0: [f"Task{i}" for i in range(5)]}
    to_schedule = scheduler.should_execute(pending=workload, strobe=2)
    assert to_schedule == []

    # Check we submit with both full
    workload = {j: [f"Task{i * 2 + j}" for i in range(5)] for j in range(2)}
    to_schedule = scheduler.should_execute(pending=workload, strobe=2)
    assert sorted(to_schedule[0]) == [f"Task{i}" for i in range(10)]


# Check that if we exceed the ideal size we put into separate buckets
def test_scheduler_reserved_two_separate():
    ideal_batch_size = 9
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)
    batcher = FakeBatcher()

    # Include two separate reservations for scheduler
    reserve_helper(scheduler, rid=0, count=5)
    reserve_helper(scheduler, rid=1, count=5)

    # Check without full on either we do not submit
    workload = make_workload({0: 4, 1: 4})
    to_schedule = scheduler.should_execute(pending=workload, strobe=0)
    assert to_schedule == []

    # Check with single RID full we still do not submit:
    workload = make_workload({0: 5, 1: 4})
    to_schedule = scheduler.should_execute(pending=workload, strobe=2)
    assert to_schedule[0] == workload[0]

    # Check we submit with both full
    workload = {j: [f"Task{i + j * 5}" for i in range(5)] for j in range(2)}
    to_schedule = scheduler.should_execute(pending=workload, strobe=2)
    assert sorted(to_schedule[0]) == workload[0]
    assert sorted(to_schedule[1]) == workload[1]
