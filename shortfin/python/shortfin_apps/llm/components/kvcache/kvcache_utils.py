from dataclasses import dataclass


@dataclass
class RefCount:
    """
    A reference counter to replace simple int.
    """

    count: int = 0

    def increment(self) -> int:
        self.count += 1
        return self.count

    def decrement(self) -> int:
        self.count -= 1
        return self.count

    def is_empty(self) -> bool:
        return self.count <= 0
