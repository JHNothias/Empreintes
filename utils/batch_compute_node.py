from empreintes.LazyTree import LazyTree
from typing import Callable, Any
import json

def settings_key(d):
    return json.dumps(d, sort_keys=True)

def batch_compute_node(compute_fn:Callable, inputs:list[str], settings_location:str, map_location:str):
    def node(tree, i):
        dependents = tree.localget(i, map_location)
        seen = dict()
        flag = False
        if i[0] == "-" : flag = True
        for d in dependents:
            s = tree.localget(i, f"{'-' if flag else ''}{settings_location}-{d}")
            k = settings_key(s)
            if k not in seen:
                seen[k] = s
        inp = [tree.localget(i, j) for j in inputs]
        return {k: compute_fn(*inp, s) for k, s in seen.items()}
    return node

def batch_item_node(compute_id:str, settings_location:str):
    def node(tree, i):
        flag = False
        if i[0] == "-" : flag = True
        desc_name = i.split('-')[-1]
        s = tree.localget(i, f"{'-' if flag else ''}{settings_location}-{desc_name}")
        return tree.localget(i, compute_id)[settings_key(s)]
    return node
