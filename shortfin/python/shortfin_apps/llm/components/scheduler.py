# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
import itertools
import logging
from typing import Dict, List
import shortfin as sf

from .invocation import LlmTaskInput

logger = logging.getLogger(__name__)


class UpdateWorkload(sf.Message):
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

    def has_member(self, rid):
        return rid in self._members

    def member_count(self, rid):
        return self._members[rid]

    def is_empty(self):
        return self._size == 0

    def can_add(self, count):
        return self._size + count <= self._max_size

    def remove(self, *, rid):
        if rid in self._members:
            old_count = self._members[rid]
            self._members.pop(rid)
            self._size = self._size - old_count

    def resize(self, *, rid, count):
        if count == 0:
            self.remove(rid=rid)
            return

        old_count = 0 if rid not in self._members else self._members[rid]
        self._members[rid] = count
        self._size = self._size + count - old_count

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
        if len(job) <= self.available():
            for queue in self._queues:
                available = self._ideal_batch_size - len(queue)
                if available > 0:
                    needed = min(available, len(job))
                    self._occupancy += needed
                    queue.extend(job[:needed])
                    job = job[needed:]

                if len(job) == 0:
                    break
            return

        # Create a new job for the workload
        self._occupancy += len(job)
        self._queues.append(job.copy())

    def get_scheduled(self):
        return set(itertools.chain(*self._queues))

    def get_jobs(self):
        return self._queues

    def available(self):
        return len(self._queues) * self._ideal_batch_size - self._occupancy


class AbstractScheduler(ABC):
    def __init__(self, *, ideal_batch_size: int) -> None:
        self._ideal_batch_size = ideal_batch_size
        self._unreserved_strobe = None
        self._wid = 0
        self._preferred_groups = 1

        self.pending: List[LlmTaskInput] = []

        # Mapping from RID to the corresponding workgroup ID
        self._workgroup_placement = {}

        # Mapping from workgroup ID to the Workgroup tracker:
        self._workgroups = {}

    @abstractmethod
    def schedule_job(self, task: LlmTaskInput):
        pass

    @abstractmethod
    def should_execute(self, *args, **kwargs) -> List[List[LlmTaskInput]]:
        pass

    @abstractmethod
    def handle_scheduler(self, msg) -> bool:
        pass

    @abstractmethod
    def reserve_workload(self, *, batcher, count, rid):
        pass

    @abstractmethod
    def handle_completed(self, rid: str) -> bool:
        pass

    def _group_jobs(
        self, rid_map: Dict[str, List[LlmTaskInput]], strobe
    ) -> WorkloadBuilder:
        workload_builder = WorkloadBuilder(ideal_batch_size=self._ideal_batch_size)

        # Split out reserved and unreserved jobs:
        reserved = {
            rid: rid_map[rid] for rid in rid_map if rid in self._workgroup_placement
        }
        unreserved = list(
            itertools.chain(
                *[
                    rid_map[rid]
                    for rid in rid_map
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

        return workload_builder

    def _schedule_reservation(self, *, rid, count):
        if rid in self._workgroup_placement:
            wid = self._workgroup_placement[rid]
            workgroup = self._workgroups[wid]
            existing = workgroup.member_count(rid=rid)
            if workgroup.can_add(count - existing):
                workgroup.resize(rid=rid, count=count)
                return

            # If we cannot fit the workgroup in the existing dispatch we need to redistribute:
            workgroup.remove(rid=rid)
            self._workgroup_placement.pop(rid)
            if workgroup.is_empty():
                self._workgroups.pop(wid)

        def schedule_new():
            self._wid = self._wid + 1
            wid = self._wid

            wg = Workgroup(wid=wid, max_size=self._ideal_batch_size)
            wg.resize(rid=rid, count=count)
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

        workgroup_sel.resize(count=count, rid=rid)
        self._workgroup_placement[rid] = workgroup_sel.wid

    def _remove(self, *, rid):
        if rid not in self._workgroup_placement:
            return

        wid = self._workgroup_placement[rid]
        workgroup = self._workgroups[wid]

        workgroup.remove(rid=rid)
        if workgroup.is_empty():
            self._workgroups.pop(wid)

        for wid in self._workgroups:
            workgroup = self._workgroups[wid]
            if workgroup.has_member(rid=rid):
                break

        self._workgroup_placement.pop(rid)


class Scheduler(AbstractScheduler):
    def __init__(self, *, ideal_batch_size):
        self._ready: List[LlmTaskInput] = []
        super().__init__(ideal_batch_size=ideal_batch_size)

    def schedule_job(self, task: LlmTaskInput):
        self._ready.append(task)

    def should_execute(self, strobe) -> List[List[LlmTaskInput]]:
        pending = self._ready
        self._ready = []
        if len(pending) == 0:
            return []

        # Determine the requested requests these jobs are for
        rids = set([j.rid for j in pending])

        # Group jobs together under their rid
        rid_map = {rid: [] for rid in rids}
        for j in pending:
            rid_map[j.rid].append(j)

        workload_builder = self._group_jobs(rid_map=rid_map, strobe=strobe)

        pending = [
            item for item in pending if item not in workload_builder.get_scheduled()
        ]
        self._ready = pending

        return workload_builder.get_jobs()

    def handle_scheduler(self, msg) -> bool:
        if isinstance(msg, UpdateWorkload):
            if msg.count == 0:
                self._remove(rid=msg.rid)
                return True

            self._schedule_reservation(rid=msg.rid, count=msg.count)
            return True

        return False

    def reserve_workload(self, *, batcher, count, rid):
        batcher.submit(UpdateWorkload(count=count, rid=rid))

    def handle_completed(self, rid: str) -> bool:
        return True


class ChunkScheduler(AbstractScheduler):
    def __init__(self, *, ideal_batch_size):
        self._pending: Dict[str, List[LlmTaskInput]] = {}
        self._ready: List[LlmTaskInput] = []
        super().__init__(ideal_batch_size=ideal_batch_size)

    def schedule_job(self, task: LlmTaskInput):
        if self._pending.get(task.rid) is None:
            self._ready.append(task)
            self._pending[task.rid] = []
        else:
            self._pending[task.rid].append(task)

    def should_execute(self, strobe) -> List[List[LlmTaskInput]]:
        jobs = self._ready
        self._ready = []
        if len(jobs) == 0:
            return []

        # Determine the requested requests these jobs are for
        rids = set([j.rid for j in jobs])

        # Group jobs together under their rid
        rid_map = {rid: [] for rid in rids}
        for j in jobs:
            rid_map[j.rid].append(j)

        workload_builder = self._group_jobs(rid_map=rid_map, strobe=strobe)

        jobs = [item for item in jobs if item not in workload_builder.get_scheduled()]
        self._ready = jobs

        return workload_builder.get_jobs()

    def handle_scheduler(self, msg) -> bool:
        if isinstance(msg, UpdateWorkload):
            if msg.count == 0:
                self._remove(rid=msg.rid)
                return True

            self._schedule_reservation(rid=msg.rid, count=msg.count)
            return True

        return False

    def reserve_workload(self, *, batcher, count, rid):
        batcher.submit(UpdateWorkload(count=count, rid=rid))

    def handle_completed(self, rid: str) -> bool:
        if len(self._pending[rid]) == 0:
            del self._pending[rid]
            return True

        next_chunk = self._pending[rid].pop(0)
        self._ready.append(next_chunk)
        return False
