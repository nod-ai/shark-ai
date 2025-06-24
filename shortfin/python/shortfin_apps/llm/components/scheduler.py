# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import itertools
import logging
import shortfin as sf
import threading

logger = logging.getLogger(__name__)


class NewWorkItem(sf.Message):
    def __init__(self, *, count: int, rid: int):
        super().__init__()
        self.count = count
        self.rid = rid


class DoneWorkItem(sf.Message):
    def __init__(self, *, count: int, rid: int):
        super().__init__()
        self.count = count
        self.rid = rid


class Workgroup:
    def __init__(self, *, wid: int, max_size: int):
        self._wid = wid
        self._members = {}
        self._size = 0
        self._max_size = max_size
        self._strobe = None

    @property
    def wid(self):
        return self._wid

    @property
    def size(self):
        return self._size

    @property
    def members(self):
        return set(self._members.keys())

    def get_members(self, members):
        return set(members) & set(self._members.keys())

    def is_member(self, rid):
        return rid in self._members

    def member_count(self, rid):
        return self._members[rid]

    def is_empty(self):
        return self._size == 0

    def can_add(self, count):
        return self._size + count <= self._max_size

    def add(self, *, rid, count):
        if rid in self._members:
            new_size = self._members[rid] + count
            self._members[rid] = new_size
        else:
            self._members[rid] = count

        self._size = self._size + count

    def remove(self, *, rid, count):
        assert rid in self._members
        assert count <= self._members[rid]

        self._size = self._size - count
        new_count = self._members[rid] - count
        if new_count == 0:
            self._members.pop(rid)
        else:
            self._members[rid] = new_count

    def schedule(self, *, pending, strobe: int):
        pending = [pending[rid] for rid in pending if rid in self._members]
        pending = list(itertools.chain(*pending))
        target_size = sum(self._members[rid] for rid in self._members)

        # Not all workgroup items are ready.
        if len(pending) < target_size:
            return None

        return pending


class WorkloadBuilder:
    def __init__(self, *, ideal_batch_size):
        self._queues = []
        self._ideal_batch_size = ideal_batch_size
        self._occupancy = 0

    def add_work(self, job):
        while len(job) > self._ideal_batch_size:
            self._occupancy += self._ideal_batch_size
            self._queues.append(job[: self._ideal_batch_size])

            job = job[self._ideal_batch_size :]

        # Place into existing jobs if here is available space:
        worksize = len(job)
        if worksize <= self.available():
            for queue in self._queues:
                available = self._ideal_batch_size - len(queue)
                if available > 0:
                    self._occupancy += available
                    queue.extend(job[available:])
                    job = job[available:]

                if len(job) == 0:
                    break
            return

        # Create a new job for the workload
        self._occupancy += len(job)
        self._queues.append(job)

    def get_scheduled(self):
        return set(itertools.chain(*self._queues))

    def get_jobs(self):
        return self._queues

    def available(self):
        return len(self._queues) * self._ideal_batch_size - self._occupancy


class Scheduler:
    def __init__(self, *, ideal_batch_size):
        self._ideal_batch_size = ideal_batch_size
        self._unreserved_strobe = None
        self._wid = 0
        self._preferred_groups = 1

        # Mapping from RID to the corresponding workgroup ID
        self._workgroup_placement = {}

        # Mapping from workgroup ID to the Workgroup tracker:
        self._workgroups = {}

    def should_execute(self, pending, strobe):
        workload_builder = WorkloadBuilder(ideal_batch_size=self._ideal_batch_size)
        returned_jobs = []

        # Split out reserved and unreserved jobs:
        reserved = {
            rid: pending[rid] for rid in pending if rid in self._workgroup_placement
        }
        unreserved = list(
            itertools.chain(
                *[
                    pending[rid]
                    for rid in pending
                    if rid not in self._workgroup_placement
                ]
            )
        )

        # Schedule all jobs known to the reservation system
        for workgroup_id in self._workgroups.keys():
            workgroup = self._workgroups[workgroup_id]
            to_schedule = workgroup.schedule(pending=reserved, strobe=strobe)
            if to_schedule is not None:
                workload_builder.add_work(to_schedule)

        # Slot any unreserved work into empty ideal space
        if len(unreserved) > 0 and workload_builder.available() > 0:
            available = workload_builder.available()
            workload_builder.add_work(unreserved[:available])
            unreserved = unreserved[available:]

        # Dispatch ideal batch size if we accumulated enough:
        while len(unreserved) >= self._ideal_batch_size:
            job_size = 0
            new_job = unreserved[: self._ideal_batch_size]
            unreserved = unreserved[self._ideal_batch_size :]
            workload_builder.add_work(new_job)
            self._unreserved_strobe = None

        # If we have remaining unreserved jobs
        if len(unreserved) > 0:
            # Schedule the strobe for a future follow up:
            if self._unreserved_strobe is None:
                self._unreserved_strobe = strobe
            # If we strobed previously we should add the remaining work:
            elif strobe - self._unreserved_strobe > 1:
                self._unreserved_strobe = None
                workload_builder.add_work(unreserved)

        return workload_builder.get_jobs()

    def _schedule(self, *, rid, count):
        if rid in self._workgroup_placement:
            wid = self._workgroup_placement[rid]
            workgroup = self._workgroups[wid]
            if workgroup.can_add(count):
                workgroup.add(rid=rid, count=count)
                return
            existing = workgroup.member_count(rid=rid)
            workgroup.remove(rid=rid, count=existing)
            count = count + existing

            self._workgroup_placement.pop(rid)
            if workgroup.is_empty():
                self._workgroups.pop(wid)

        def schedule_new():
            self._wid = self._wid + 1
            wid = self._wid

            wg = Workgroup(wid=wid, max_size=self._ideal_batch_size)
            wg.add(rid=rid, count=count)
            self._workgroups[wid] = wg
            self._workgroup_placement[rid] = wid

        # Guarantee there are two workgroups and schedule full count:
        if len(self._workgroups) < self._preferred_groups:
            schedule_new()
            return

        # Search for a workgroup with space
        workgroup_sel = None
        for wid in self._workgroups.keys():
            workgroup = self._workgroups[wid]

            if workgroup.can_add(count):
                workgroup_sel = workgroup
                break

        # Schedule if no home found:
        if workgroup_sel is None:
            schedule_new()
            return

        workgroup_sel.add(count=count, rid=rid)
        self._workgroup_placement[rid] = workgroup_sel.wid

    def _release(self, *, rid, count):
        assert rid in self._workgroup_placement

        wid = self._workgroup_placement[rid]
        workgroup = self._workgroups[wid]

        workgroup.remove(rid=rid, count=count)
        if workgroup.is_empty():
            self._workgroups.pop(wid)

        remove = True
        for wid in self._workgroups:
            workgroup = self._workgroups[wid]
            if workgroup.is_member(rid=rid):
                remove = False
                break

        if remove:
            self._workgroup_placement.pop(rid)

    def handle_scheduler(self, msg):
        if isinstance(msg, NewWorkItem):
            self._schedule(rid=msg.rid, count=msg.count)
            return True

        if isinstance(msg, DoneWorkItem):
            self._release(rid=msg.rid, count=msg.count)
            return True

        return False

    def reserve_workitem(self, *, batcher, count, rid):
        batcher.submit(NewWorkItem(count=count, rid=rid))

    def release_workitem(self, *, batcher, count, rid):
        batcher.submit(DoneWorkItem(count=count, rid=rid))
