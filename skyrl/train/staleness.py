"""Consumer-side oldest-policy fencing, independent of training backends."""

from enum import Enum


class GroupDecision(str, Enum):
    KEEP = "keep"
    STALE = "stale"
    DUPLICATE = "duplicate"


class StalenessFence:
    """Decide once per UID per epoch, before any group reaches the optimizer.

    The oldest version covers every continuation in a partial trajectory. Discarded
    groups are terminal for this epoch, so there is no stale retry starvation loop.
    Persist terminal UIDs in the caller's checkpoint, not this ephemeral registry.
    """

    def __init__(self, max_age: int):
        if isinstance(max_age, bool) or not isinstance(max_age, int) or max_age < 0:
            raise ValueError("max_age must be a nonnegative integer")
        self.max_age = max_age
        self._seen: set[str] = set()

    def decide(self, uid: str, oldest_version: int, current_version: int) -> GroupDecision:
        if not isinstance(uid, str) or not uid:
            raise ValueError("uid must be a nonempty string")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (oldest_version, current_version)):
            raise ValueError("policy versions must be nonnegative integers")
        if oldest_version > current_version:
            raise ValueError("oldest policy version cannot exceed the consumer version")
        if uid in self._seen:
            return GroupDecision.DUPLICATE
        self._seen.add(uid)
        if current_version - oldest_version > self.max_age:
            return GroupDecision.STALE
        return GroupDecision.KEEP
