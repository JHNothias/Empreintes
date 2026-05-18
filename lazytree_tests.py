"""Unit tests for LazyTree."""
from __future__ import annotations

import pickle
import tempfile
import os
import pytest

# ---------------------------------------------------------------------------
# Make the module importable regardless of working directory
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, os.path.dirname(__file__))
from LazyTree import LazyTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quiet(t: LazyTree) -> LazyTree:
    """Disable console output on a tree and return it."""
    t.report = False
    return t


def simple_tree(**kwargs) -> LazyTree:
    """
    Build a small tree:
        root
        ├── a  = 1
        └── b  = lambda: root.a + 1  (depends on a)
    Extra kwargs are merged into the spec.
    """
    spec = {
        "a": lambda tree, i: 1,
        "b": lambda tree, i: tree.localget(i, "a") + 10,
    }
    spec.update(kwargs)
    return quiet(LazyTree(spec))


# ===========================================================================
# 1.  __init__
# ===========================================================================

class TestInit:
    def test_empty_spec(self):
        t = quiet(LazyTree())
        assert t.d == {}
        assert t.memo == {}
        assert t.times_changed == {}
        assert t.dependencies == {}

    def test_flat_spec_registers_nodes(self):
        t = quiet(LazyTree({"x": lambda *_: 42, "y": lambda *_: 99}))
        assert "x" in t.d
        assert "y" in t.d

    def test_flat_spec_with_tree_id(self):
        t = quiet(LazyTree({"x": lambda *_: 7}, id="root"))
        assert "root-x" in t.d
        assert "x" not in t.d

    def test_nested_spec_flattened(self):
        t = quiet(LazyTree({"a": {"b": lambda *_: 5}}))
        assert "a-b" in t.d

    def test_deeply_nested_spec(self):
        t = quiet(LazyTree({"a": {"b": {"c": lambda *_: 1}}}))
        assert "a-b-c" in t.d

    def test_embedded_lazytree_merged(self):
        inner = quiet(LazyTree({"x": lambda *_: 3}, id="inner"))
        outer = quiet(LazyTree({"sub": inner}))
        # inner's node "inner-x" should appear under "sub"
        assert any("sub" in k for k in outer.d)

    def test_times_changed_initialised_to_zero(self):
        t = quiet(LazyTree({"n": lambda *_: 0}))
        assert t.times_changed["n"] == 0

    def test_dependencies_initialised_empty(self):
        t = quiet(LazyTree({"n": lambda *_: 0}))
        assert t.dependencies["n"] == {}

    def test_computing_set_empty(self):
        t = quiet(LazyTree())
        assert t._computing == set()


# ===========================================================================
# 2.  globalid / _strip_global
# ===========================================================================

class TestIdHelpers:
    def test_globalid_no_prefix(self):
        t = quiet(LazyTree(id=""))
        assert t.globalid("foo") == "foo"

    def test_globalid_with_prefix(self):
        t = quiet(LazyTree(id="root"))
        assert t.globalid("foo") == "root-foo"

    def test_strip_global_no_prefix(self):
        t = quiet(LazyTree(id=""))
        assert t._strip_global("foo") == "foo"

    def test_strip_global_with_prefix(self):
        t = quiet(LazyTree(id="root"))
        assert t._strip_global("root-foo") == "foo"

    def test_strip_global_non_matching(self):
        t = quiet(LazyTree(id="root"))
        assert t._strip_global("other-foo") == "other-foo"

    def test_globalid_strip_roundtrip(self):
        t = quiet(LazyTree(id="myid"))
        local = "some-node"
        assert t._strip_global(t.globalid(local)) == local


# ===========================================================================
# 3.  get
# ===========================================================================

class TestGet:
    def test_basic_get(self):
        t = simple_tree()
        assert t.get("a") == 1

    def test_memoises_on_first_call(self):
        calls = []
        t = quiet(LazyTree({"a": lambda tree, i: calls.append(1) or 42}))
        t.get("a")
        t.get("a")
        assert len(calls) == 1  # computed only once

    def test_recompute_flag_forces_rerun(self):
        calls = []
        t = quiet(LazyTree({"a": lambda tree, i: calls.append(1) or 42}))
        t.get("a")
        t.get("a", recompute=True)
        assert len(calls) == 2

    def test_force_global_recompute_true(self):
        calls = []
        t = quiet(LazyTree({"a": lambda tree, i: calls.append(1) or 7}))
        t.get("a")
        t.get("a", force_global_recompute=True)
        assert len(calls) == 2

    def test_force_global_recompute_false_overrides_force_recompute(self):
        calls = []
        t = quiet(LazyTree({"a": lambda tree, i: calls.append(1) or 7}))
        t.get("a")
        t.force_recompute = True
        t.get("a", force_global_recompute=False)
        # force_global_recompute=False should prevent recompute even though force_recompute=True
        assert len(calls) == 1

    def test_force_recompute_flag_on_tree(self):
        calls = []
        t = quiet(LazyTree({"a": lambda tree, i: calls.append(1) or 5}))
        t.get("a")
        t.force_recompute = True
        t.get("a")
        assert len(calls) == 2

    def test_unknown_id_raises_key_error(self):
        t = quiet(LazyTree({"a": lambda *_: 1}))
        with pytest.raises(KeyError):
            t.get("zzz")

    def test_cycle_raises_recursion_error(self):
        # a calls b, b calls a
        t = quiet(LazyTree({
            "a": lambda tree, i: tree.localget(i, "b"),
            "b": lambda tree, i: tree.localget(i, "a"),
        }))
        with pytest.raises(RecursionError):
            t.get("a")

    def test_times_changed_increments_on_first_compute(self):
        t = simple_tree()
        assert t.times_changed["a"] == 0
        t.get("a")
        assert t.times_changed["a"] == 1

    def test_times_changed_no_increment_same_value(self):
        t = quiet(LazyTree({"a": lambda *_: 42}))
        t.get("a")
        count_before = t.times_changed["a"]
        t.get("a", recompute=True)
        assert t.times_changed["a"] == count_before  # same value → no increment

    def test_times_changed_increments_different_value(self):
        counter = [0]
        def fn(tree, i):
            counter[0] += 1
            return counter[0]
        t = quiet(LazyTree({"a": fn}))
        t.get("a")
        t.get("a", recompute=True)
        assert t.times_changed["a"] == 2

    def test_numpy_array_does_not_raise(self):
        """Regression: bool(arr != arr) raises ValueError; must be caught."""
        import numpy as np
        arr = np.array([1, 2, 3])
        t = quiet(LazyTree({"a": lambda *_: arr}))
        t.get("a")
        t.get("a", recompute=True)  # should not raise

    def test_with_tree_id(self):
        t = quiet(LazyTree({"a": lambda *_: 99}, id="root"))
        assert t.get("a") == 99

    def test_calldepth_restored_after_exception(self):
        t = quiet(LazyTree({"a": lambda *_: 1}))
        try:
            t.get("missing")
        except KeyError:
            pass
        assert t._calldepth == 0


# ===========================================================================
# 4.  set
# ===========================================================================

class TestSet:
    def test_set_scalar_value(self):
        t = simple_tree()
        t.get("a")
        t.set("a", 99)
        assert t.get("a") == 99

    def test_set_callable(self):
        t = simple_tree()
        t.set("a", lambda tree, i: 77, isCallable=True)
        assert t.get("a") == 77

    def test_set_increments_times_changed(self):
        t = simple_tree()
        before = t.times_changed["a"]
        t.set("a", 5)
        assert t.times_changed["a"] > before

    def test_set_invalidates_localget_cache(self):
        t = simple_tree()
        t.get("b")  # populates cache: b → a
        # "a" should be in b's localget cache
        assert any("a" in cache for cache in t._localget_cache.values())
        t.set("a", 999)
        # after set, every cache entry pointing at "a" must be gone
        for cache in t._localget_cache.values():
            assert "a" not in cache.values()

    def test_set_new_node(self):
        t = simple_tree()
        t.set("c", 55)
        assert t.get("c") == 55

    def test_set_update_memo_false(self):
        """With update_memo=False the callable is updated but get() is not called."""
        calls = []
        t = quiet(LazyTree({"a": lambda *_: 1}))
        t.get("a")
        t.set("a", lambda tree, i: calls.append(1) or 2, isCallable=True, update_memo=False)
        assert len(calls) == 0

    def test_set_update_memo_true_calls_get(self):
        calls = []
        t = quiet(LazyTree({"a": lambda *_: 1}))
        t.get("a")
        t.set("a", lambda tree, i: calls.append(1) or 2, isCallable=True, update_memo=True)
        assert len(calls) == 1

    def test_set_with_tree_id(self):
        t = quiet(LazyTree({"a": lambda *_: 1}, id="root"))
        t.set("a", 42)
        assert t.get("a") == 42


# ===========================================================================
# 5.  setmemo
# ===========================================================================

class TestSetMemo:
    def test_setmemo_stores_value(self):
        t = simple_tree()
        t.setmemo("a", 123)
        assert t.memo["a"] == 123

    def test_setmemo_increments_times_changed_on_new_value(self):
        t = simple_tree()
        t.get("a")
        before = t.times_changed["a"]
        t.setmemo("a", 999)
        assert t.times_changed["a"] == before + 1

    def test_setmemo_no_increment_same_value(self):
        t = simple_tree()
        t.get("a")
        t.setmemo("a", 1)          # same as current memo
        count = t.times_changed["a"]
        t.setmemo("a", 1)
        assert t.times_changed["a"] == count

    def test_setmemo_numpy_no_raise(self):
        """Regression test for ValueError on numpy array equality."""
        import numpy as np
        t = simple_tree()
        arr = np.array([1, 2, 3])
        t.setmemo("a", arr)        # must not raise ValueError
        arr2 = np.array([4, 5, 6])
        t.setmemo("a", arr2)       # different array, also must not raise

    def test_setmemo_invalidates_localget_cache(self):
        t = simple_tree()
        t.get("b")  # populates b's cache pointing at "a"
        t.setmemo("a", 500)
        for cache in t._localget_cache.values():
            assert "a" not in cache.values()

    def test_setmemo_with_tree_id(self):
        t = quiet(LazyTree({"a": lambda *_: 1}, id="root"))
        t.setmemo("a", 77)
        assert t.memo["root-a"] == 77


# ===========================================================================
# 6.  is_outofdate
# ===========================================================================

class TestIsOutofdate:
    def test_fresh_node_not_outofdate(self):
        t = simple_tree()
        t.get("b")
        assert not t.is_outofdate("b")

    def test_dep_changed_marks_outofdate(self):
        t = simple_tree()
        t.get("b")
        t.set("a", 99)   # bumps times_changed["a"]
        assert t.is_outofdate("b")

    def test_no_deps_never_outofdate(self):
        t = simple_tree()
        t.get("a")
        assert not t.is_outofdate("a")

    def test_missing_dep_treated_as_outofdate(self):
        t = simple_tree()
        t.get("b")
        # Manually inject a phantom dependency
        t.dependencies["b"]["nonexistent-node"] = 0
        assert t.is_outofdate("b")

    def test_unknown_gid_returns_false(self):
        t = simple_tree()
        assert not t.is_outofdate("totally-unknown")


# ===========================================================================
# 7.  localget
# ===========================================================================

class TestLocalget:
    def test_sibling_lookup(self):
        """b should be able to find sibling 'a' via plain localget."""
        t = simple_tree()
        assert t.get("b") == 11   # a=1, b=a+10

    def test_localget_caches_resolution(self):
        t = simple_tree()
        t.get("b")
        # After first get, "a" should be cached inside b's localget cache
        b_cache = t._localget_cache.get("b", {})
        assert "a" in b_cache

    def test_localget_cache_hit_returns_same_value(self):
        t = simple_tree()
        first = t.get("b")
        second = t.get("b")
        assert first == second

    def test_localget_backtrack_prefix(self):
        """Node c can find ancestor-level 'a' using '-a' syntax."""
        t = quiet(LazyTree({
            "a": lambda *_: 5,
            "sub": {
                "c": lambda tree, i: tree.localget(i, "-a") * 2,
            }
        }))
        assert t.get("sub-c") == 10

    def test_localget_backtrack_not_found_raises(self):
        t = quiet(LazyTree({
            "sub": {
                "c": lambda tree, i: tree.localget(i, "-missing"),
            }
        }))
        with pytest.raises(KeyError):
            t.get("sub-c")

    def test_localget_plain_not_found_raises(self):
        t = quiet(LazyTree({
            "sub": {
                "c": lambda tree, i: tree.localget(i, "ghost"),
            }
        }))
        with pytest.raises(KeyError):
            t.get("sub-c")

    def test_localget_cache_evicted_when_target_removed(self):
        t = simple_tree()
        t.get("b")  # primes cache
        # Manually remove "a" from self.d (simulates node removal)
        del t.d["a"]
        # Next call to localget must not use stale cache
        # (will raise KeyError since "a" is gone, but must not silently return stale data)
        with pytest.raises((KeyError, Exception)):
            t.get("b", recompute=True)

    def test_dependency_recorded_by_localget(self):
        t = simple_tree()
        t.get("b")
        assert "a" in t.dependencies["b"]

    def test_localget_records_times_changed_snapshot(self):
        t = simple_tree()
        t.get("b")
        # The snapshot should equal current times_changed["a"]
        assert t.dependencies["b"]["a"] == t.times_changed["a"]


# ===========================================================================
# 8.  validate_dependencies
# ===========================================================================

class TestValidateDependencies:
    def test_clean_tree_no_problems(self):
        t = simple_tree()
        t.rebuild_all()
        assert t.validate_dependencies() == {}

    def test_detects_missing_times_changed(self):
        t = simple_tree()
        del t.times_changed["a"]
        problems = t.validate_dependencies()
        assert "a" in problems

    def test_detects_missing_dependencies_entry(self):
        t = simple_tree()
        del t.dependencies["a"]
        problems = t.validate_dependencies()
        assert "a" in problems

    def test_detects_unknown_dep(self):
        t = simple_tree()
        t.get("b")
        t.dependencies["b"]["ghost-node"] = 0
        problems = t.validate_dependencies()
        assert "b" in problems
        assert any("ghost-node" in p for p in problems["b"])

    def test_detects_corrupt_count(self):
        t = simple_tree()
        t.get("b")
        # Record a count higher than actual
        t.dependencies["b"]["a"] = t.times_changed["a"] + 100
        problems = t.validate_dependencies()
        assert "b" in problems

    def test_detects_dangling_memo(self):
        t = simple_tree()
        t.get("a")
        t.memo["phantom"] = 42    # memo entry with no matching node
        problems = t.validate_dependencies()
        assert "phantom" in problems


# ===========================================================================
# 9.  detect_cycles
# ===========================================================================

class TestDetectCycles:
    def test_no_cycles_in_dag(self):
        t = simple_tree()
        t.rebuild_all()
        assert t.detect_cycles() == []

    def test_detects_direct_cycle(self):
        """Force a cycle into the static dependency dict and check detection."""
        t = simple_tree()
        t.rebuild_all()
        # Manually insert a cycle: a depends on b, b depends on a
        t.dependencies["a"]["b"] = 0
        t.dependencies["b"]["a"] = 0
        cycles = t.detect_cycles()
        assert len(cycles) > 0

    def test_self_loop(self):
        t = simple_tree()
        t.rebuild_all()
        t.dependencies["a"]["a"] = 0
        cycles = t.detect_cycles()
        assert len(cycles) > 0


# ===========================================================================
# 10.  rebuild_all
# ===========================================================================

class TestRebuildAll:
    def test_rebuild_all_returns_memo(self):
        t = simple_tree()
        result = t.rebuild_all()
        assert isinstance(result, dict)
        assert "a" in result
        assert "b" in result

    def test_rebuild_all_clears_memo_first(self):
        t = simple_tree()
        t.get("a")
        t.memo["a"] = 999   # corrupt memo
        t.rebuild_all()
        assert t.memo["a"] == 1   # recomputed correctly

    def test_rebuild_all_clears_dependencies(self):
        t = simple_tree()
        t.get("b")
        t.dependencies["a"]["ghost"] = 0   # inject stale dep
        t.rebuild_all()
        assert "ghost" not in t.dependencies.get("a", {})

    def test_validate_after_rebuild(self):
        t = simple_tree()
        t.rebuild_all()
        assert t.validate_dependencies() == {}


# ===========================================================================
# 11.  getsubtree
# ===========================================================================

class TestGetsubtree:
    def test_getsubtree_returns_children(self):
        t = quiet(LazyTree({
            "parent": {
                "x": lambda *_: 1,
                "y": lambda *_: 2,
            },
            "other": lambda *_: 99,
        }))
        sub = t.getsubtree("parent")
        assert all(v in (1, 2) for v in sub.values())
        assert len(sub) == 2

    def test_getsubtree_excludes_unrelated_nodes(self):
        t = quiet(LazyTree({
            "parent": {"x": lambda *_: 1},
            "sibling": lambda *_: 9,
        }))
        sub = t.getsubtree("parent")
        assert not any("sibling" in k for k in sub)


# ===========================================================================
# 12.  save / load
# ===========================================================================

class TestSaveLoad:
    def test_save_load_roundtrip(self):
        t = simple_tree()
        t.get("a")
        t.get("b")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            path = f.name
        try:
            t.save(path)
            t2 = simple_tree()
            t2.load(path)
            assert t2.memo == t.memo
            assert t2.times_changed == t.times_changed
        finally:
            os.unlink(path)

    def test_load_merges_into_existing_state(self):
        t = simple_tree()
        t.get("a")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            path = f.name
        try:
            t.save(path)
            t2 = simple_tree()
            t2.memo["extra"] = "kept"
            t2.load(path)
            assert "extra" in t2.memo   # existing keys preserved
            assert "a" in t2.memo       # loaded keys present
        finally:
            os.unlink(path)

    def test_save_load_with_tree_id(self):
        t = quiet(LazyTree({"a": lambda *_: 5}, id="root"))
        t.get("a")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            path = f.name
        try:
            t.save(path)
            t2 = quiet(LazyTree({"a": lambda *_: 5}, id="root"))
            t2.load(path)
            assert t2.memo.get("root-a") == 5
        finally:
            os.unlink(path)


# ===========================================================================
# 13.  merge
# ===========================================================================

class TestMerge:
    def test_merge_adds_subtree_nodes(self):
        inner = quiet(LazyTree({"x": lambda *_: 7}))
        outer = quiet(LazyTree({"placeholder": lambda *_: 0}))
        outer.merge("placeholder", inner)
        assert any("placeholder" in k for k in outer.d)

    def test_merge_removes_original_key(self):
        inner = quiet(LazyTree({"x": lambda *_: 7}))
        outer = quiet(LazyTree({"placeholder": lambda *_: 0}))
        outer.merge("placeholder", inner)
        # Original leaf "placeholder" replaced by subtree nodes
        assert "placeholder" not in outer.d or any(k != "placeholder" for k in outer.d if "placeholder" in k)

    def test_merge_sets_tree_id(self):
        inner = quiet(LazyTree({"x": lambda *_: 7}))
        outer = quiet(LazyTree({}))
        outer.merge("section", inner)
        assert inner.id == "section"


# ===========================================================================
# 14.  Integration
# ===========================================================================

class TestIntegration:
    def test_chain_invalidation(self):
        """Changing 'a' should cause 'b' (which depends on 'a') to recompute."""
        t = simple_tree()
        assert t.get("b") == 11    # a=1, b=a+10
        t.set("a", 5)
        assert t.get("b") == 15    # a=5, b=a+10

    #@pytest.mark.xfail(
    #    strict=True,
    #    reason=(
    #        "Known limitation: is_outofdate only checks direct deps one level deep. "
    #        "d depends on b and c; b and c depend on a. When a changes, "
    #        "b.times_changed and c.times_changed haven't bumped yet, so d sees "
    #        "its direct deps as up-to-date and returns the stale memo value. "
    #        "Fix: make is_outofdate recurse transitively."
    #    ),
    #)
    def test_diamond_dependency(self):
        """
        a = 1
        b = a + 1
        c = a + 2
        d = b + c   (depends on both b and c, which both depend on a)
        """
        t = quiet(LazyTree({
            "a": lambda tree, i: 1,
            "b": lambda tree, i: tree.localget(i, "a") + 1,
            "c": lambda tree, i: tree.localget(i, "a") + 2,
            "d": lambda tree, i: tree.localget(i, "b") + tree.localget(i, "c"),
        }))
        assert t.get("d") == 5   # (1+1) + (1+2)
        t.set("a", 10)
        assert t.get("d") == 23  # (10+1) + (10+2)

    def test_independent_nodes_not_recomputed(self):
        calls_x = []
        calls_y = []
        t = quiet(LazyTree({
            "x": lambda tree, i: calls_x.append(1) or 1,
            "y": lambda tree, i: calls_y.append(1) or 2,
        }))
        t.get("x")
        t.get("y")
        t.set("x", 99)
        t.get("y")   # y does not depend on x
        assert len(calls_y) == 1   # y not recomputed

    def test_validate_after_chain_update(self):
        t = simple_tree()
        t.rebuild_all()
        t.set("a", 5)
        t.get("b")
        assert t.validate_dependencies() == {}

    def test_detect_no_cycle_in_chain(self):
        t = quiet(LazyTree({
            "a": lambda *_: 1,
            "b": lambda tree, i: tree.localget(i, "a") + 1,
            "c": lambda tree, i: tree.localget(i, "b") + 1,
        }))
        t.rebuild_all()
        assert t.detect_cycles() == []
