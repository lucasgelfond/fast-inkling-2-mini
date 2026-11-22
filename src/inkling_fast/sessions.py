"""Several cached sessions in mlx-vlm's prefix cache (APC).

Inkling's sliding-window KV and conv states can't be paged in blocks, so APC
keeps whole-cache snapshots ("checkpoints"): one at the end of each prompt
plus intermediate ones it can roll back to. Upstream, the snapshot budget is
shared: one request may take every slot, so each agent's turn evicted the
other agents' sessions.

Here the total budget is `2 * sessions` snapshots, each request still takes at
most two, and a snapshot is dropped once two longer snapshots of the same
conversation extend it (it can no longer be the best restore point). A
snapshot that several conversations extend, such as a shared system prompt,
is kept.
"""

from itertools import combinations

CHECKPOINTS_PER_REQUEST = 2


def _extends(short, long):
    return len(short) < len(long) and long[: len(short)] == short


def _superseded(entries):
    stale = []
    for key, entry in entries.items():
        longer = [
            other.token_ids
            for other in entries.values()
            if other.extra_hash == entry.extra_hash and _extends(entry.token_ids, other.token_ids)
        ]
        one_conversation = all(_extends(a, b) or _extends(b, a) for a, b in combinations(longer, 2))
        if len(longer) >= 2 and one_conversation:
            stale.append(key)
    return stale


def install():
    from mlx_vlm.apc import APCManager
    from mlx_vlm.apc_coordinator import APCCoordinator

    checkpoint_lengths = APCCoordinator.checkpoint_lengths

    def per_request_checkpoint_lengths(self, *args, **kwargs):
        manager = self.manager
        with manager.lock:  # reentrant; stores and lookups see the total budget again on exit
            total = manager._exact_cache_max
            manager._exact_cache_max = min(total, CHECKPOINTS_PER_REQUEST)
            try:
                return checkpoint_lengths(self, *args, **kwargs)
            finally:
                manager._exact_cache_max = total

    store_exact_cache = APCManager.store_exact_cache

    def store_and_prune(self, *args, **kwargs):
        stored = store_exact_cache(self, *args, **kwargs)
        with self.lock:
            for key in _superseded(self._exact_cache):
                del self._exact_cache[key]
        return stored

    APCCoordinator.checkpoint_lengths = per_request_checkpoint_lengths
    APCManager.store_exact_cache = store_and_prune
