# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import logging

from shortfin_apps.llm.components.scheduler import Scheduler, WorkloadBuilder


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


# Checked that strobing will return workload when detected
def test_scheduler_unreserved_strobed():
    ideal_batch_size = 32
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    workload = make_workload({0: 4})
    schedule_workload(scheduler, workload)

    to_schedule = scheduler.should_execute(strobe=0)
    assert len(to_schedule) == 0

    to_schedule = scheduler.should_execute(strobe=1)
    assert len(to_schedule) == 0

    to_schedule = scheduler.should_execute(strobe=2)
    assert len(to_schedule) == 1
    assert to_schedule[0] == workload[0]


# Check that a full ideal set is returned
def test_scheduler_unreserved_full():
    ideal_batch_size = 4
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    workload = make_workload({0: 4})
    schedule_workload(scheduler, workload)

    to_schedule = scheduler.should_execute(strobe=0)
    assert len(to_schedule) == 1
    assert to_schedule[0] == to_schedule[0]


# Check that a subset is returned if overfilled
def test_scheduler_unreserved_overfull():
    ideal_batch_size = 3
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    workload = make_workload({0: 4})
    schedule_workload(scheduler, workload)

    to_schedule = scheduler.should_execute(strobe=0)
    assert len(to_schedule) == 1
    assert to_schedule[0] == workload[0][:3]

    # Check we fill as many ideal batches as possible:
    workload = make_workload({0: 10})
    schedule_workload(scheduler, workload)

    to_schedule = scheduler.should_execute(strobe=0)
    assert len(to_schedule) == 3
    assert to_schedule[0] == workload[0][:3]
    assert to_schedule[1] == workload[0][3:6]
    assert to_schedule[2] == workload[0][6:9]


# Check that if there is a reservation only the unreserved are executed:
def test_scheduler_unreserved_with_reservation():
    ideal_batch_size = 8
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    reserve_helper(scheduler, rid=1, count=5)

    workload = make_workload({0: 4, 1: 4})
    schedule_workload(scheduler, workload)

    to_schedule = scheduler.should_execute(strobe=0)
    assert len(to_schedule) == 0

    to_schedule = scheduler.should_execute(strobe=2)
    assert len(to_schedule) == 1
    assert to_schedule[0] == workload[0]


# Check if reserved and all passed to execute the whole job:
def test_scheduler_reserved_basic():
    ideal_batch_size = 32
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    reserve_helper(scheduler, rid=0, count=5)
    workload = make_workload({0: 4})
    schedule_workload(scheduler, workload)

    to_schedule = scheduler.should_execute(strobe=0)
    assert to_schedule == []

    to_schedule = scheduler.should_execute(strobe=2)
    assert to_schedule == []

    workload = make_workload({0: 5})
    schedule_workload(scheduler, workload)

    to_schedule = scheduler.should_execute(strobe=2)
    assert len(to_schedule) == 1
    assert to_schedule[0] == workload[0]

    reserve_helper(scheduler, rid=0, count=4)
    workload = make_workload({0: 4})
    schedule_workload(scheduler, workload)

    to_schedule = scheduler.should_execute(strobe=2)
    assert len(to_schedule) == 1
    assert to_schedule[0] == workload[0]


# Check if reserved and all passed to execute the whole job.
def test_scheduler_reserved_extra():
    ideal_batch_size = 7
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    reserve_helper(scheduler, rid=0, count=5)
    workload = make_workload({0: 5, 1: 3})
    schedule_workload(scheduler, workload)

    to_schedule = scheduler.should_execute(strobe=2)
    assert len(to_schedule) == 1
    assert to_schedule[0] == workload[0] + workload[1][:2]


# Reserve a job at that exceeds the max size, should be split between jobs.
def test_scheduler_reserved_too_big():
    ideal_batch_size = 5
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    reserve_helper(scheduler, rid=0, count=7)
    workload = make_workload({0: 7})
    schedule_workload(scheduler, workload)

    to_schedule = scheduler.should_execute(strobe=2)
    assert len(to_schedule) == 2
    assert to_schedule[0] == workload[0][:5]
    assert to_schedule[1] == workload[0][5:]


# Check two reservations fall into the same bucket:
def test_scheduler_reserved_two_shared():
    ideal_batch_size = 10
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    # Include two separate reservations for scheduler
    reserve_helper(scheduler, rid=0, count=5)
    reserve_helper(scheduler, rid=1, count=5)

    # Check without full on either we do not submit
    workload = {
        j: [FakeTask(rid=j, instance_id=i * 2 + j) for i in range(4)] for j in range(2)
    }
    schedule_workload(scheduler, workload)
    to_schedule = scheduler.should_execute(strobe=0)
    assert to_schedule == []

    # Check with single RID full we still do not submit:
    workload = {0: [FakeTask(rid=0, instance_id=i) for i in range(5)]}
    schedule_workload(scheduler, workload)
    to_schedule = scheduler.should_execute(strobe=2)
    assert to_schedule == []

    # Check we submit with both full
    workload = {j: [FakeTask(rid=j, instance_id=i) for i in range(5)] for j in range(2)}
    schedule_workload(scheduler, workload)
    to_schedule = scheduler.should_execute(strobe=2)
    combined_workloads = workload[0] + workload[1]
    assert to_schedule[0] == combined_workloads


# Check that if we exceed the ideal size we put into separate buckets
def test_scheduler_reserved_two_separate():
    ideal_batch_size = 9
    scheduler = Scheduler(ideal_batch_size=ideal_batch_size)

    # Include two separate reservations for scheduler
    reserve_helper(scheduler, rid=0, count=5)
    reserve_helper(scheduler, rid=1, count=5)

    # Check without full on either we do not submit
    workload = make_workload({0: 4, 1: 4})
    schedule_workload(scheduler, workload)
    to_schedule = scheduler.should_execute(strobe=0)
    assert to_schedule == []

    # Check with single RID full we still do not submit:
    workload = make_workload({0: 5, 1: 4})
    schedule_workload(scheduler, workload)
    to_schedule = scheduler.should_execute(strobe=2)
    assert to_schedule[0] == workload[0]

    # Check we submit with both full
    workload = {
        j: [FakeTask(rid=j, instance_id=i + j * 5) for i in range(5)] for j in range(2)
    }
    schedule_workload(scheduler, workload)
    to_schedule = scheduler.should_execute(strobe=2)
    assert to_schedule[0] == workload[0]
    assert to_schedule[1] == workload[1]


class TestWorkloadBuilder:
    def setup_method(self):
        self.ideal_batch_size = 4
        self.workload_builder = WorkloadBuilder(ideal_batch_size=self.ideal_batch_size)

    def test_workload_builder_less_than_ideal(self):
        job = ["Task1", "Task2"]
        self.workload_builder.add_work(job)
        assert self.workload_builder.get_jobs() == [job]
        assert self.workload_builder.available() == 2

    def test_workload_builder_more_than_ideal(self):
        job = ["Task1", "Task2", "Task3", "Task4", "Task5"]
        self.workload_builder.add_work(job)

        assert self.workload_builder.get_jobs() == [
            job[: self.ideal_batch_size],
            job[self.ideal_batch_size :],
        ]
        assert self.workload_builder.available() == 3

    def test_workload_builder_exactly_ideal(self):
        job = ["Task1", "Task2", "Task3", "Task4"]
        self.workload_builder.add_work(job)
        assert self.workload_builder.get_jobs() == [job]
        assert self.workload_builder.available() == 0

    def test_workload_builder_two_small_jobs(self):
        job1 = ["Task1"]
        job2 = ["Task2"]
        self.workload_builder.add_work(job1)
        assert self.workload_builder.available() == 3
        assert self.workload_builder.get_jobs() == [job1]
        print(job1)
        self.workload_builder.add_work(job2)
        assert self.workload_builder.available() == 2
        print(job1)
        assert self.workload_builder.get_jobs() == [job1 + job2]

    def test_workload_builder_multiple_jobs_exact_size(self):
        job = [f"Task{i + 1}" for i in range(12)]
        self.workload_builder.add_work(job)

        assert self.workload_builder.get_jobs() == [job[0:4], job[4:8], job[8:12]]
        assert self.workload_builder.available() == 0

    def test_workload_builder_multiple_jobs_smaller_size(self):
        """
        Jobs are added unbroken until there is enough room to fully break up an
        incoming job. This happens with the 4th job (Task10) in the test."""
        jobs = make_workload({i: 3 for i in range(6)})
        expected = [
            # row 0
            [FakeTask(rid=0, instance_id=i) for i in range(3)]
            + [FakeTask(rid=3, instance_id=9)],
            # row 1
            [FakeTask(rid=1, instance_id=i) for i in range(3, 6)]
            + [FakeTask(rid=3, instance_id=10)],
            # row 2
            [FakeTask(rid=2, instance_id=i) for i in range(6, 9)]
            + [FakeTask(rid=3, instance_id=11)],
            # row 3
            [FakeTask(rid=4, instance_id=i) for i in range(12, 15)],
            # row 4
            [FakeTask(rid=5, instance_id=i) for i in range(15, 18)],
        ]
        for i in range(len(jobs)):
            self.workload_builder.add_work(jobs[i])
        assert self.workload_builder.get_jobs() == expected
        assert self.workload_builder.available() == 2

    def test_workload_builder_multiple_jobs(self):
        jobs = make_workload({0: 2, 1: 4, 2: 1, 3: 5})
        expected = [
            [
                FakeTask(rid=0, instance_id=0),
                FakeTask(rid=0, instance_id=1),
                FakeTask(rid=2, instance_id=6),
                FakeTask(rid=3, instance_id=11),
            ],
            [
                FakeTask(rid=1, instance_id=2),
                FakeTask(rid=1, instance_id=3),
                FakeTask(rid=1, instance_id=4),
                FakeTask(rid=1, instance_id=5),
            ],
            [
                FakeTask(rid=3, instance_id=7),
                FakeTask(rid=3, instance_id=8),
                FakeTask(rid=3, instance_id=9),
                FakeTask(rid=3, instance_id=10),
            ],
        ]
        for i in range(len(jobs)):
            self.workload_builder.add_work(jobs[i])

        assert self.workload_builder.get_jobs() == expected
        assert self.workload_builder.available() == 0
