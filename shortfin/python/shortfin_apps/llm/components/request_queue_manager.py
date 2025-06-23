import threading
import logging

logger = logging.getLogger(__name__)

class RequestQueueManager:
    def __init__(self, model_params):
        self.model_params = model_params
        self._lock = threading.Lock()
        self.current_queue_size = 0
        self.max_queue_size = 0
        self._initialize_queues()

    def _initialize_queues(self):
        """Initialize request and response queues"""
        if self.model_params.decode_batch_sizes:
            self.max_queue_size = max(self.model_params.decode_batch_sizes)
            logger.debug(f"Max queue size: {self.max_queue_size}")

    def add_to_queue(self, num_beams: int) -> bool:
        """Try to add a request to the queue. Returns True if successful, False if queue is full."""
        with self._lock:
            if self.current_queue_size >= self.max_queue_size:
                return False
            self.current_queue_size += num_beams
            logger.debug(f"Adding to queue, queue size: {self.current_queue_size}")
            return True

    def remove_from_queue(self, num_beams: int):
        """Remove a request from the queue."""
        with self._lock:
            if self.current_queue_size >= num_beams:
                self.current_queue_size -= num_beams
                logger.debug(f"Removing from queue, queue size: {self.current_queue_size}")
