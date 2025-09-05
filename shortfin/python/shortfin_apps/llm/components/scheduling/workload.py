import itertools

import shortfin as sf


class UpdateWorkload(sf.Message):
    def __init__(self, *, count: int, rid: str):
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

    def schedule(self, *, pending):
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
