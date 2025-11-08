# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from abc import ABC, abstractmethod
import itertools
import logging
import math
from typing import Dict, List, Set
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


class ExtendAttentionScheduler(AbstractScheduler):
    """Scheduler for extend-attention batching with dynamic chunking.

    This scheduler manages requests that are dynamically chunked based on the
    number of active requests and a token budget. Each request is processed
    in chunks, with chunk sizes calculated to maximize GPU utilization while
    respecting page alignment constraints.
    """

    def __init__(self, *, token_budget: int, block_seq_stride: int):
        # Pass dummy ideal_batch_size to parent - not used in extend attention
        super().__init__(ideal_batch_size=1)
        self._block_seq_stride = block_seq_stride
        self._token_budget = token_budget
        # Track active requests (full task inputs with all tokens)
        self._active_requests: Dict[str, LlmTaskInput] = {}
        # Track current position (token offset) for each request
        self._request_positions: Dict[str, int] = {}
        # Track the chunk size used for each request in the last execution
        self._last_chunk_sizes: Dict[str, int] = {}
        # Track requests that are currently executing to prevent double-scheduling
        self._in_flight: Set[str] = set()

    def schedule_job(self, task: LlmTaskInput):
        """Add a request to the scheduler.

        The task contains all tokens for the request. We'll dynamically chunk it
        at scheduling time based on the number of active requests.
        """
        rid = task.rid
        if rid in self._active_requests:
            logger.warning(
                f"Request {rid} is already scheduled. Ignoring duplicate schedule_job call."
            )
            return

        # New request - store it and initialize position from task
        # (may not be 0 with trie_prefix_matching)
        self._active_requests[rid] = task
        self._request_positions[rid] = task.start_position

    def should_execute(self, strobe) -> List[List[LlmTaskInput]]:
        """Determine which tasks should be executed now.

        Dynamically chunks active requests based on the number of requests and token budget.
        Each request gets a page-aligned chunk that fits within the budget.
        """
        if not self._active_requests:
            return []

        # Calculate dynamic chunk size based on active requests that are NOT in-flight
        available_requests = {
            rid: task
            for rid, task in self._active_requests.items()
            if rid not in self._in_flight
        }

        if not available_requests:
            # All requests are currently executing
            return []

        num_active = len(available_requests)
        tokens_per_request = self._token_budget // num_active
        # Align to page boundaries
        chunk_size = (
            tokens_per_request // self._block_seq_stride
        ) * self._block_seq_stride

        if chunk_size == 0:
            # Too many requests for the budget - shouldn't happen but handle gracefully
            chunk_size = self._block_seq_stride

        # Create chunks for this batch
        batch = []
        for rid, full_task in available_requests.items():
            position = self._request_positions[rid]
            all_tokens = full_task.input_tokens

            # Determine how many tokens to take
            remaining_tokens = len(all_tokens) - position
            tokens_to_take = min(chunk_size, remaining_tokens)

            if tokens_to_take <= 0:
                continue

            # Create a chunk from current position
            chunk_tokens = all_tokens[position : position + tokens_to_take]

            logger.info(
                f"ExtendAttentionScheduler: rid={rid}, position={position}, tokens_to_take={tokens_to_take}, chunk_tokens={chunk_tokens}, all_tokens={all_tokens}"
            )

            # Calculate cumulative seq_len and block_count
            cumulative_seq_len = position + len(chunk_tokens)
            chunk_block_count = math.ceil(cumulative_seq_len / self._block_seq_stride)

            # Get page_ids up to the current block count
            chunk_page_ids = full_task.page_ids[:chunk_block_count]

            # Store the actual chunk size used for this request
            self._last_chunk_sizes[rid] = tokens_to_take

            # Mark this request as in-flight
            self._in_flight.add(rid)

            # Create the chunk task input
            chunk_task = LlmTaskInput(
                rid=rid,
                instance_id=full_task.instance_id,
                block_count=chunk_block_count,
                seq_len=cumulative_seq_len,
                input_tokens=chunk_tokens,
                page_ids=chunk_page_ids,
                start_position=position,
            )
            batch.append(chunk_task)

        return [batch] if batch else []

    def handle_scheduler(self, msg) -> bool:
        # Handle scheduler messages
        return False

    def reserve_workload(self, *, batcher, count, rid):
        # Handle workload reservation
        pass

    def handle_completed(self, rid: str) -> bool:
        """Handle completion of a chunk.

        Updates the position for this request and determines if more tokens remain.

        Returns True if the request is fully complete (no more tokens).
        Returns False if there are more tokens to process.
        """
        assert (
            rid in self._active_requests
        ), f"Request {rid} not found in active requests"

        # Remove from in-flight set to allow next chunk to be scheduled
        self._in_flight.discard(rid)

        full_task = self._active_requests[rid]
        current_position = self._request_positions[rid]

        # Get the chunk size that was actually used for this request in the last execution
        tokens_processed = self._last_chunk_sizes.get(rid, 0)
        new_position = current_position + tokens_processed

        # Update position
        self._request_positions[rid] = new_position

        # Check if we've processed all tokens
        if new_position >= len(full_task.input_tokens):
            # Request complete - remove from active requests
            del self._active_requests[rid]
            del self._request_positions[rid]
            del self._last_chunk_sizes[rid]
            return True  # Request fully complete

        # More tokens to process
        return False
