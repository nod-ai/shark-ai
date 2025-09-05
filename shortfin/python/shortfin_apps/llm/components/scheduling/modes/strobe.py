import itertools
from typing import Dict

from ..abstract import AbstractScheduler
from ..config import SchedulerConfig
from ..workload import WorkloadBuilder, Workgroup, UpdateWorkload


class StrobeScheduler(AbstractScheduler):
    def __init__(self, config: SchedulerConfig):
        self._runtime = config.runtime
        self._ideal_batch_size = config.ideal_batch_size
        self._unreserved_strobe = None
        self._wid = 0
        self._preferred_groups = 1

        # Mapping from RID to the corresponding workgroup ID
        self._workgroup_placement: Dict[str, int] = {}

        # Mapping from workgroup ID to the Workgroup tracker:
        self._workgroups: Dict[int, Workgroup] = {}

    def should_execute(self, pending):
        workload_builder = WorkloadBuilder(ideal_batch_size=self._ideal_batch_size)

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
            to_schedule = workgroup.schedule(pending=reserved)
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
        strobe = self._runtime.get_tick()
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

    def handle_message(self, message):
        if isinstance(message, UpdateWorkload):
            if message.count == 0:
                self._remove(rid=message.rid)
                return True

            self._schedule(rid=message.rid, count=message.count)
            return True

        return False

    def reserve_workload(self, *, count, rid):
        self._runtime.submit_workload(count=count, rid=rid)
