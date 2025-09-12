# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import logging

from shortfin_apps.llm.components.scheduler import ChunkScheduler, WorkloadBuilder


logger = logging.getLogger(__name__)


class FakeBatcher:
    def __init__(self):
        self.msgs = []

    def submit(self, x):
        self.msgs.append(x)

    def pop(self):
        msgs = self.msgs
        self.msgs = []
        return msgs


class FakeTask:
    def __init__(self, rid, instance_id):
        self.rid = rid
        self.instance_id = instance_id

    def __eq__(self, other) -> bool:
        return self.rid == other.rid and self.instance_id == other.instance_id

    def __hash__(self):
        return hash((self.rid, self.instance_id))


def reserve_helper(scheduler, *, rid, count):
    batcher = FakeBatcher()
    scheduler.reserve_workload(rid=rid, count=count, batcher=batcher)
    assert scheduler.handle_scheduler(batcher.pop()[0]) == True


def make_workload(rids):
    workload = {}
    running = 0
    for rid in rids:
        count = rids[rid]
        workload[rid] = [
            FakeTask(rid=rid, instance_id=i + running) for i in range(count)
        ]
        running = running + count

    return workload


def schedule_workload(scheduler, workload):
    scheduler._ready = []
    for rid in workload:
        for task in workload[rid]:
            scheduler.schedule_job(task)


class TestChunkScheduler:

    # Check that entire request is returned if only one chunk
    def test_single_chunks(self):
        ideal_batch_size = 4
        scheduler = ChunkScheduler(ideal_batch_size=ideal_batch_size)

        workload = {0: 1, 1: 1, 2: 1, 3: 1}
        workload = make_workload(workload)
        schedule_workload(scheduler, workload)

        to_schedule = scheduler.should_execute(strobe=2)
        assert len(to_schedule[0]) == 4
        assert to_schedule[0] == workload[0] + workload[1] + workload[2] + workload[3]

        # No more work to schedule
        to_schedule = scheduler.should_execute(strobe=2)
        assert len(to_schedule) == 0

    def test_multi_chunks(self):
        ideal_batch_size = 4
        scheduler = ChunkScheduler(ideal_batch_size=ideal_batch_size)

        workload = {}
        for i in range(4):
            workload[i] = i + 1
        workload = make_workload(workload)
        schedule_workload(scheduler, workload)

        to_schedule = scheduler.should_execute(strobe=0)
        assert len(to_schedule[0]) == 4

        # Schedule first chunk of each request
        assert to_schedule[0] == [
            workload[0][0],
            workload[1][0],
            workload[2][0],
            workload[3][0],
        ]

        # handle_completed NOT called yet
        to_schedule = scheduler.should_execute(strobe=0)
        assert len(to_schedule) == 0
        to_schedule = scheduler.should_execute(strobe=1)
        assert len(to_schedule) == 0
        to_schedule = scheduler.should_execute(strobe=2)
        assert len(to_schedule) == 0

        for i in range(4):
            scheduler.handle_completed(i)

        to_schedule = scheduler.should_execute(strobe=0)
        to_schedule = scheduler.should_execute(strobe=1)
        to_schedule = scheduler.should_execute(strobe=2)
        assert len(to_schedule[0]) == 3
        assert to_schedule[0] == [
            workload[1][1],
            workload[2][1],
            workload[3][1],
        ]

        for i in [1, 2, 3]:
            scheduler.handle_completed(i)

        to_schedule = scheduler.should_execute(strobe=0)
        to_schedule = scheduler.should_execute(strobe=1)
        to_schedule = scheduler.should_execute(strobe=2)
        assert len(to_schedule[0]) == 2
        assert to_schedule[0] == [
            workload[2][2],
            workload[3][2],
        ]

        for i in [2, 3]:
            scheduler.handle_completed(i)

        to_schedule = scheduler.should_execute(strobe=0)
        to_schedule = scheduler.should_execute(strobe=1)
        to_schedule = scheduler.should_execute(strobe=2)
        assert len(to_schedule[0]) == 1
        assert to_schedule[0] == [
            workload[3][3],
        ]

        scheduler.handle_completed(3)

        to_schedule = scheduler.should_execute(strobe=0)
        to_schedule = scheduler.should_execute(strobe=1)
        to_schedule = scheduler.should_execute(strobe=2)

        assert len(to_schedule) == 0
