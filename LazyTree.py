from __future__ import annotations
from time import process_time
from typing import Callable, Any
from functools import reduce
#import inspect, asyncio
#from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle

"""LazyTree: A class for managing hierarchical, memoized, and dependency-aware computations.
This class allows you to define a tree-like structure where each node can represent a computation or a value. 
It supports memoization, dependency tracking, and hierarchical ID management. The tree can also be serialized 
and deserialized for persistence.
Classes:
    LazyTree: Represents the main tree structure.
Type Aliases:
    LazyTreeSpec: A dictionary specifying the structure of the LazyTree.
    LazyTreeBuilt: A dictionary representing the fully built LazyTree.
Methods:
    __init__(d: LazyTreeSpec, id: str, memoize: bool) -> None:
        Initializes a LazyTree instance with the given structure, ID, and memoization setting.
    merge(locationid: str, tree: LazyTree) -> None:
        Merges another LazyTree into the current tree at the specified location.
    globalid(localid: str) -> str:
        Converts a local ID into a global ID by prefixing it with the tree's ID.
    save(filepath: str) -> None:
        Saves the current state of the tree (memo, times_changed, dependencies) to a file.
    load(filepath: str) -> None:
        Loads the state of the tree (memo, times_changed, dependencies) from a file.
    localget(callerid: str, localid: str, recompute: bool, force_global_recompute: bool | None) -> Any:
        Retrieves a value using relative path search, with caching and dependency tracking.
    get(id: str, recompute: bool, force_global_recompute: bool | None) -> Any:
        Retrieves the value associated with the given ID, with optional recomputation and global recompute control.
    is_outofdate(id: str) -> bool:
        Checks if a node's dependencies are out of date.
    set(id: str, value: Callable | Any, isCallable: bool) -> None:
        Sets a value or callable for the given ID in the tree.
    setmemo(id: str, value: Any) -> None:
        Directly sets a memoized value for the given ID.
    rebuild_all() -> LazyTreeBuilt:
        Clears the memo and rebuilds all nodes in the tree.
Attributes:
    id (str): The unique identifier for the tree.
    d (dict): The dictionary representing the tree structure.
    memo (dict): A dictionary for memoized values.
    force_recompute (bool): A flag to force recomputation of all nodes.
    times_changed (dict): Tracks the number of times each node has changed.
    dependencies (dict): Tracks dependencies between nodes.
    _localget_cache (dict): Caches local ID lookups for performance.
    report (bool): Flag to enable or disable reporting of operations.
Dependencies:
    - time.process_time: Used for performance measurement.
    - inspect: Used for introspection (not currently utilized in the code).
    - asyncio: Used for asynchronous operations (not currently utilized in the code).
    - concurrent.futures: Provides thread and process pool executors (not currently utilized in the code).
    - pickle: Used for serialization and deserialization of the tree state."""



type LazyTreeSpec = dict[str, (Callable | LazyTreeSpec | LazyTree)]
type LazyTreeBuilt = dict[str, Any]

class LazyTree:
    def __init__(self, d:LazyTreeSpec = dict(), id:str = "", memoize:bool = True) -> None:
        self.id = id
        self.d = dict()
        self.memo = dict()
        self.force_recompute = False
        self.times_changed = dict()
        self.dependencies = dict()
        self._localget_cache = dict()
        #self.notify = dict()
        #self.react = dict()
        self.report = True
        def _flatten(D: dict):
            res = dict()
            for k, v in D.items():
                if isinstance(v, dict):
                    flattened = _flatten(v)
                    for l, val in flattened.items():
                        res[(k, *l)] = val
                elif isinstance(v, LazyTree):
                    flattened = v.d
                    for l, val in flattened.items():
                        res[(k, v.globalid(l))] = val
                else:
                    res[(k,)] = v
            return res

        def generate_id(pieces:tuple[str]):
            return reduce(lambda a, b: a + '-' + b, pieces[1:], pieces[0])
    
        for k, v in _flatten(d).items():
            id = generate_id(k)
            self.d[self.globalid(id)] = v
            self.dependencies[self.globalid(id)] = dict()
            self.times_changed[self.globalid(id)] = 0
            self._localget_cache[self.globalid(id)] = dict()

    def merge(self, locationid:str, tree:LazyTree) -> None:
        if locationid in self.d:
            del self.d[locationid]
        tree.id = locationid
        for k, v in tree.d.items():
            self.d[locationid + '-' + k] = v

    def globalid(self, localid:str) -> str:
        if self.id != '':
            return self.id + '-' + localid
        else:
            return localid
    
    def save(self, filepath:str):
        with open(filepath, 'wb') as f:
            pickle.dump({'memo' : self.memo, "times_changed" : self.times_changed, "dependencies" : self.dependencies}, file=f)
    
    def load(self, filepath:str):
        with open(filepath, 'rb') as f:
            res = pickle.load(file=f)
            self.memo.update(res["memo"])
            self.times_changed.update(res["times_changed"])
            self.dependencies.update(res["dependencies"])
    
    def localget(self, callerid:str, localid:str, recompute:bool = False, force_global_recompute : (bool | None) = None) -> Any:
        """ Implements relative path search.
            If localid is prefixed with "-", it will backtrack the caller path until it finds a match for localid.
            Otherwise, it will look for a match to localid by backtracking only once.
            Caches the access for performance reasons.
        """
        if localid in self._localget_cache[callerid]:   # caching logic
            res = self.get(self._localget_cache[callerid][localid], recompute=recompute) # get re-adds self.id.
            self.dependencies[callerid][self._localget_cache[callerid][localid]] = self.times_changed[self._localget_cache[callerid][localid]]
            return res
    
        if localid.startswith("-"):                             # -lid
            callersplit = callerid.split("-")                   # callersplit = [cid1, cid2, cid3] (cid1 is top level)
            for k in range(len(callersplit)):                   # k = 0, 1, 2
                fetchloc = callersplit[:-k-2]                   # fetchloc = [cid1, cid2, cid3], [cid1, cid2], [cid1]
                fetchid = ("-".join(fetchloc) + "-" if fetchloc else "") + localid.removeprefix('-')    # fetchid = cid1-cid2-cid3-lid, cid1-cid2-lid, cid1-lid
                if fetchid in self.d:                           # fetchid found
                    res = self.get(fetchid.removeprefix(self.id), recompute=recompute) # get re-adds self.id.
                    self.dependencies[callerid][fetchid] = self.times_changed[fetchid]
                    self._localget_cache[callerid][localid] = fetchid
                    return res
                raise(KeyError(f'ID not found : {fetchid}'))
        else:                                              # lid
            callerid2 = callerid.removeprefix(self.id)           # callerid = cid2-cid3 (cid1 added back later in the get)
            callersplit = callerid2.split("-")                   # callersplit = [cid2, cid3]
            fetchloc = callersplit[:len(callersplit) - 1]       # fetchloc = [cid2] (go one level up)
            fetchid = "-".join(fetchloc) + ("-" if fetchloc else '') + localid        # fetchid cid2-lidz
            res = self.get(fetchid, recompute=recompute, force_global_recompute=force_global_recompute) # get cid1-cid2-lid
            self.dependencies[callerid][fetchid] = self.times_changed[fetchid]
            self._localget_cache[callerid][localid] = fetchid
            return res

    def get(self, id:str, recompute:bool = False, force_global_recompute : (bool | None) = None) -> Any:
        """ gets the value with the given id.
            `force_global_recompute`:   - None : No effect
                                        - True : forces every subsequent get to recompute its return value
                                        - False : forces every subsequent get to check its return value
            It recomputes anyway if :   - id is not recorded in memo
                                        - the memo is out of date (a dependency in the previous call has changed)"""
        
        if self.report : print(f"get({id}, recompute = {recompute}, force_global_recompute = {force_global_recompute}) | force_recompute = {self.force_recompute} | ", end=''); start = process_time()

        assert self.globalid(id) in self.d, f"Identifier could not be found in self.get({id}, recompute = {recompute}, force_global_recompute = {force_global_recompute})."
        recomp = self.force_recompute

        if force_global_recompute is not None:
            self.force_recompute = force_global_recompute
    
        outofdate = False
        if recompute or self.force_recompute or self.globalid(id) not in self.memo or (outofdate := self.is_outofdate(self.globalid(id))):
            if self.report:
                if not (recompute or self.force_recompute or outofdate): print(f"id NOT FOUND in memo | ", end='');
                elif outofdate : print(f"dependencies out of date | ", end='')
    
            res = self.d[self.globalid(id)](self.globalid(id))
            self.memo[self.globalid(id)] = res
            self.times_changed[self.globalid(id)] += 1
        else:
            if self.report: print(f"id FOUND in memo | ", end='')

            res = self.memo[self.globalid(id)]

        self.force_recompute = recomp

        if self.report : end = process_time(); print(f"finished in {end - start:3f} seconds") # type: ignore
        return res
    
    def is_outofdate(self, id):
        return any([self.times_changed[dep] != n for dep, n in self.dependencies[self.globalid(id)].items()])
    
    #def trace(self, id):
    #    if not self.dependencies[self.globalid(id)]:
    #        return []
    #    else:
    #        res = []
    #        for dep in self.dependencies[self.globalid(id)]:
    #            if dep == self.globalid(id):
    #                res.append(self.globalid(id))
    #            else:
    #                res.append(self.trace(dep))
    #        return res

    def set(self, id:str, value:(Callable | Any), isCallable = False, update_memo=True) -> None:
        if isCallable:
            self.d[self.globalid(id)] = value
        else:
            self.d[self.globalid(id)] = lambda i : value
        self.times_changed[self.globalid(id)] += 1
        if update_memo:
            self.get(self.globalid(id), recompute=True)
    
    def setmemo(self, id:str, value:Any) -> None:
        self.memo[self.globalid(id)] = value
        self.times_changed[self.globalid(id)] += 1
    
    def getsubtree(self, id:str, recompute:bool = False, force_global_recompute : (bool | None) = None) -> Any:
        res = dict()
        for k in self.d:
            if k.startswith(self.globalid(id)):
                res[k] = self.get(k, recompute = recompute, force_global_recompute = force_global_recompute)
        return res
    
    def rebuild_all(self) -> LazyTreeBuilt:
        """Whipes the memo and builds it fresh."""
        self.memo = dict()
        for k in self.d:
            self.get(k)
        return self.memo