from __future__ import annotations
from time import process_time
from typing import Callable, Any
from functools import reduce
import pickle

"""LazyTree: A class for managing hierarchical, memoized, and dependency-aware computations."""

type LazyTreeSpec = dict[str, (Callable | LazyTreeSpec | "LazyTree")]
type LazyTreeBuilt = dict[str, Any]

_SENTINEL = object()   # used to detect "not previously memoized"


class LazyTree:
    def __init__(self, d: LazyTreeSpec = dict(), id: str = "", memoize: bool = True) -> None:
        self.id = id
        self.d = dict()
        self.memo = dict()
        self.force_recompute = False
        self.times_changed = dict()
        self.dependencies = dict()
        self._localget_cache = dict()
        self._computing: set[str] = set()   # FIX 8: cycle detection
        self._calldepth = 0
        self.report = True
        self.spec = d

        def _flatten(D: dict):
            res = dict()
            for k, v in D.items():
                if isinstance(v, dict):
                    for l, val in _flatten(v).items():
                        res[(k, *l)] = val
                elif isinstance(v, LazyTree):
                    for l, val in v.d.items():
                        res[(k, v.globalid(l))] = val
                else:
                    res[(k,)] = v
            return res

        def generate_id(pieces: tuple[str]):
            return reduce(lambda a, b: a + "-" + b, pieces[1:], pieces[0])

        for k, v in _flatten(d).items():
            gid = self.globalid(generate_id(k))
            self.d[gid] = v
            self.dependencies[gid] = dict()
            self.times_changed[gid] = 0
            self._localget_cache[gid] = dict()

    def new(self):
        """Returns a fresh instance of the tree without any ulterior modifications."""
        return LazyTree(self.spec)

    # ------------------------------------------------------------------
    # Identity helpers
    # ------------------------------------------------------------------

    def globalid(self, localid: str) -> str:
        return (self.id + "-" + localid) if self.id else localid

    def _strip_global(self, gid: str) -> str:
        """Return the local portion of a global id (inverse of globalid)."""
        if self.id:
            prefix = self.id + "-"
            if gid.startswith(prefix):
                return gid[len(prefix):]
        return gid

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def merge(self, locationid: str, tree: LazyTree) -> None:
        if locationid in self.d:
            del self.d[locationid]
        tree.id = locationid
        for k, v in tree.d.items():
            self.d[locationid + "-" + k] = v

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(
                {"memo": self.memo, "times_changed": self.times_changed, "dependencies": self.dependencies},
                file=f,
            )

    def load(self, filepath: str):
        with open(filepath, "rb") as f:
            res = pickle.load(file=f)
            self.memo.update(res["memo"])
            self.times_changed.update(res["times_changed"])
            self.dependencies.update(res["dependencies"])

    # ------------------------------------------------------------------
    # Core read / write
    # ------------------------------------------------------------------

    def get(self, id: str, recompute: bool = False, force_global_recompute: bool | None = None) -> Any:
        self._calldepth += 1
        depth = self._calldepth
        indent = "│   " * (depth - 1)
        prefix = "├── "
        start = process_time()

        gid = self.globalid(id)

        if self.report:
            print(
                f"{indent}{prefix}get({id}, recompute={recompute}, "
                f"force_global_recompute={force_global_recompute}, "
                f"force_recompute={self.force_recompute})"
            )

        try:
            if gid not in self.d:
                raise KeyError(f"Identifier not found in self.get({id!r})")

            # FIX 8: cycle detection
            if gid in self._computing:
                raise RecursionError(
                    f"Circular dependency detected: '{gid}' is already being computed. "
                    f"Current chain: {self._computing}"
                )

            prev_force = self.force_recompute
            if force_global_recompute is not None:
                self.force_recompute = force_global_recompute

            outofdate = False
            need_recompute = (
                recompute
                or self.force_recompute
                or gid not in self.memo
                or (outofdate := self.is_outofdate(gid))  # FIX 1: pass gid directly
            )

            if need_recompute:
                reason = []
                if gid not in self.memo:
                    reason.append("not in memo")
                if recompute:
                    reason.append("recompute=True")
                if self.force_recompute:
                    reason.append("force_recompute=True")
                if outofdate:
                    reason.append("dependencies out of date")
                if self.report:
                    print(f"{indent}│   ↳ recomputing ({', '.join(reason)})")

                # FIX 8: mark as computing before calling, clear after
                self._computing.add(gid)
                try:
                    res = self.d[gid](self, gid)
                finally:
                    self._computing.discard(gid)

                # FIX 9: only increment times_changed when the value actually changed
                old_val = self.memo.get(gid, _SENTINEL)
                self.memo[gid] = res
                try:
                    value_changed = bool((old_val is _SENTINEL) or (res != old_val))
                except Exception:
                    value_changed = True  # incomparable types → assume changed
                if value_changed:
                    self.times_changed[gid] += 1
            else:
                if self.report:
                    print(f"{indent}│   ↳ using memo")
                res = self.memo[gid]

            self.force_recompute = prev_force
            if self.report:
                elapsed = process_time() - start
                print(f"{indent}└── done in {elapsed:.6f}s")
            return res

        finally:
            self._calldepth -= 1

    def set(self, id: str, value: Callable | Any, isCallable=False, update_memo=True) -> None:
        gid = self.globalid(id)

        # Ensure the node is registered (supports adding new nodes via set)
        if gid not in self.d:
            self.dependencies[gid] = dict()
            self.times_changed[gid] = 0
            self._localget_cache[gid] = dict()

        if isCallable:
            self.d[gid] = value
        else:
            self.d[gid] = lambda tree, i: value

        # FIX 4: invalidate _localget_cache for any caller that resolved to this gid
        self._invalidate_localget_cache(gid)

        if update_memo:
            # FIX 2: pass local id so get() can globalise correctly (was passing gid → double prefix)
            self.get(id, recompute=True)
        else:
            # Still bump times_changed so dependents know the node changed
            self.times_changed[gid] += 1

    def setmemo(self, id: str, value: Any, invalidate:bool = True) -> None:
        gid = self.globalid(id)
        old_val = self.memo.get(gid, _SENTINEL)
        self.memo[gid] = value
        # FIX 9: only increment when value changed
        try:
            value_changed = bool((old_val is _SENTINEL) or (value != old_val))
        except Exception:
            value_changed = True
        if value_changed:
            if invalidate :
                self.times_changed[gid] += 1
        # FIX 4: invalidate cache entries pointing at this gid
        if invalidate:
            self._invalidate_localget_cache(gid)

    def _invalidate_localget_cache(self, gid: str) -> None:
        """Remove every _localget_cache entry whose resolved global id is *gid*."""
        for cache in self._localget_cache.values():
            stale = [local_key for local_key, resolved in cache.items() if resolved == gid]
            for local_key in stale:
                del cache[local_key]

    # ------------------------------------------------------------------
    # Dependency validation
    # ------------------------------------------------------------------

    def is_outofdate(self, gid: str) -> bool:
        # FIX 1: caller already passes a global id; do NOT call globalid() again.
        # FIX 7: guard against dependencies that no longer exist in the tree.
        deps = self.dependencies.get(gid)
        if not deps:
            return False
        for dep, recorded_count in deps.items():
            if dep not in self.times_changed:
                # Dependency was removed → conservatively treat as out-of-date
                return True
            if self.times_changed[dep] != recorded_count:
                return True
            if self.is_outofdate(dep):
                return True
        return False

    def validate_dependencies(self) -> dict[str, list[str]]:
        """
        Audit every recorded dependency.
        Returns a dict mapping node-id → list of problem descriptions.
        An empty dict means everything is consistent.
        """
        problems: dict[str, list[str]] = {}

        for gid in self.d:
            node_problems: list[str] = []

            # 1. Node has no entry in times_changed
            if gid not in self.times_changed:
                node_problems.append("missing from times_changed")

            # 2. Node has no entry in dependencies
            if gid not in self.dependencies:
                node_problems.append("missing from dependencies")
            else:
                for dep, recorded in self.dependencies[gid].items():
                    # 3. Dependency refers to an unknown node
                    if dep not in self.d:
                        node_problems.append(f"dependency '{dep}' not in self.d")
                    # 4. Recorded count is ahead of the dep's actual count (corruption)
                    elif dep not in self.times_changed:
                        node_problems.append(f"dependency '{dep}' missing from times_changed")
                    elif recorded > self.times_changed[dep]:
                        node_problems.append(
                            f"dependency '{dep}': recorded count {recorded} > "
                            f"actual count {self.times_changed[dep]} (corrupt)"
                        )

            # 5. Memo exists for a node that has no callable
            if gid in self.memo and gid not in self.d:
                node_problems.append("memo entry exists but node is not in self.d")

            if node_problems:
                problems[gid] = node_problems

        # 6. Dangling memo entries (node was removed)
        for gid in self.memo:
            if gid not in self.d:
                problems.setdefault(gid, []).append("memo entry has no corresponding node in self.d")

        return problems

    def detect_cycles(self) -> list[list[str]]:
        """
        Return a list of cycles found in the *static* dependency graph.
        Each cycle is represented as an ordered list of node ids.
        Uses DFS with colouring (white / grey / black).
        Note: static deps are only populated after nodes have been computed at
        least once, so this is most useful after rebuild_all().
        """
        WHITE, GREY, BLACK = 0, 1, 2
        colour = {gid: WHITE for gid in self.d}
        cycles: list[list[str]] = []
        path: list[str] = []

        def dfs(node: str):
            colour[node] = GREY
            path.append(node)
            for dep in self.dependencies.get(node, {}):
                if dep not in colour:
                    continue   # unknown dep, validate_dependencies() will catch it
                if colour[dep] == GREY:
                    # Found a cycle – extract the loop portion
                    cycle_start = path.index(dep)
                    cycles.append(path[cycle_start:] + [dep])
                elif colour[dep] == WHITE:
                    dfs(dep)
            path.pop()
            colour[node] = BLACK

        for gid in self.d:
            if colour[gid] == WHITE:
                dfs(gid)

        return cycles

    # ------------------------------------------------------------------
    # Relative-path lookup
    # ------------------------------------------------------------------

    def localget(
        self,
        callerid: str,
        localid: str,
        recompute: bool = False,
        force_global_recompute: bool | None = None,
    ) -> Any:
        """Relative-path lookup with caching and dependency recording."""

        # Ensure caller has a cache bucket
        if callerid not in self._localget_cache:
            self._localget_cache[callerid] = dict()

        # --- cache hit ---
        if localid in self._localget_cache[callerid]:
            cached_gid = self._localget_cache[callerid][localid]
            if cached_gid not in self.d:
                # FIX 5: cached target was removed → evict and fall through
                del self._localget_cache[callerid][localid]
            else:
                # FIX 4 (cache path): strip global prefix before calling get()
                res = self.get(
                    self._strip_global(cached_gid),
                    recompute=recompute,
                    force_global_recompute=force_global_recompute,
                )
                self.dependencies[callerid][cached_gid] = self.times_changed[cached_gid]
                return res

        # --- cache miss ---
        if localid.startswith("-"):
            # Backtrack through every ancestor level until a match is found
            callersplit = callerid.split("-")
            for k in range(len(callersplit)):
                fetchloc = callersplit[: -(k + 2)] if (k + 2) <= len(callersplit) else []
                bare = localid.removeprefix("-")
                fetchid = ("-".join(fetchloc) + "-" if fetchloc else "") + bare
                if fetchid in self.d:
                    res = self.get(
                        self._strip_global(fetchid),
                        recompute=recompute,
                        force_global_recompute=force_global_recompute,
                    )
                    self.dependencies[callerid][fetchid] = self.times_changed[fetchid]
                    self._localget_cache[callerid][localid] = fetchid
                    return res
            # FIX 3: raise AFTER the loop, not inside it
            raise KeyError(f"ID not found via backtrack: '{localid}' from caller '{callerid}'")
        else:
            # One level up only
            # FIX 6: use _strip_global (with dash) so split is always clean
            callerid_local = self._strip_global(callerid)
            callersplit = callerid_local.split("-")
            fetchloc = callersplit[: len(callersplit) - 1]
            fetchid_local = "-".join(fetchloc) + ("-" if fetchloc else "") + localid
            fetchid_global = self.globalid(fetchid_local)
            if fetchid_global not in self.d:
                raise KeyError(
                    f"ID not found: '{localid}' resolved to '{fetchid_global}' "
                    f"(caller '{callerid}')"
                )
            res = self.get(fetchid_local, recompute=recompute, force_global_recompute=force_global_recompute)
            self.dependencies[callerid][fetchid_global] = self.times_changed[fetchid_global]
            self._localget_cache[callerid][localid] = fetchid_global
            return res

    # ------------------------------------------------------------------
    # Subtree helpers
    # ------------------------------------------------------------------

    def getsubtree(
        self,
        id: str,
        recompute: bool = False,
        force_global_recompute: bool | None = None,
    ) -> dict:
        res = dict()
        prefix = self.globalid(id)
        for k in self.d:
            if k.startswith(prefix):
                key = "-".join(k.split("-")[1:])
                res[key] = self.get(
                    self._strip_global(k),
                    recompute=recompute,
                    force_global_recompute=force_global_recompute,
                )
        return res

    def rebuild_all(self) -> LazyTreeBuilt:
        """Wipe memo and dependencies, then rebuild every node fresh."""
        self.memo = dict()
        # Reset dependency snapshots so is_outofdate starts clean
        self.dependencies = {k: {} for k in self.d}
        for k in self.d:
            self.get(self._strip_global(k))
        return self.memo
