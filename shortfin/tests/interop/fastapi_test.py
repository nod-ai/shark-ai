import asyncio
import pytest
import threading
from shortfin.interop.fastapi import RequestStatusTracker


class FakeRequest:
    def __init__(self):
        self._disconnected = False
        self._lock = threading.Lock()
        pass

    async def set_disconnected(self, disconnected):
        with self._lock:
            self._disconnected = disconnected

    async def is_disconnected(self):
        with self._lock:
            return self._disconnected


class FakeCancellable:
    def __init__(self):
        self._cancelled = False

    def is_cancelled(self):
        return self._cancelled

    def cancel(self):
        self._cancelled = True


@pytest.mark.asyncio
async def test_status_tracker_cancelled():
    request = FakeRequest()
    cancellable = FakeCancellable()
    tracker = RequestStatusTracker(request)
    tracker.add_cancellable(cancellable)

    assert not cancellable.is_cancelled()

    await request.set_disconnected(True)
    await asyncio.sleep(2)
    assert cancellable.is_cancelled()
    tracker.close()


@pytest.mark.asyncio
async def test_status_tracker_finished():
    request = FakeRequest()
    cancellable = FakeCancellable()
    tracker = RequestStatusTracker(request)
    await request.set_disconnected(True)
    await asyncio.sleep(2)
    tracker.add_cancellable(cancellable)
    assert cancellable.is_cancelled()
    tracker.close()


@pytest.mark.asyncio
async def test_status_tracker_nocancel():
    request = FakeRequest()
    cancellable = FakeCancellable()
    tracker = RequestStatusTracker(request)
    tracker.add_cancellable(cancellable)
    tracker.close()
    await asyncio.sleep(2)
    assert not cancellable.is_cancelled()
