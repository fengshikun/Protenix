# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import torch
import torch.distributed as dist


def distributed_available() -> bool:
    return dist.is_available() and dist.is_initialized()


def _sanitize_obj(obj, max_list_len: int = 200, max_array_elems: int = 256):
    """
    Make obj safe for dist.all_gather_object():
    - scalar tensor -> float
    - non-scalar tensor -> {shape, dtype}
    - ndarray -> small sample or {shape,dtype,size}
    - list/tuple -> cap length, recurse
    - dict -> cap keys, recurse
    """
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return float(obj.detach().cpu().item())
        return {"__tensor__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}

    if isinstance(obj, np.ndarray):
        n = int(obj.size)
        if n <= max_array_elems:
            return obj.reshape(-1).tolist()
        return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype), "size": n}

    if isinstance(obj, dict):
        if len(obj) > max_list_len:
            keys = list(obj.keys())[:max_list_len]
            obj = {k: obj[k] for k in keys}
            obj["__truncated_dict__"] = True
        return {k: _sanitize_obj(v, max_list_len, max_array_elems) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        lst = list(obj)
        if len(lst) > max_list_len:
            lst = lst[:max_list_len]
            lst.append({"__truncated_list__": True, "orig_len": len(obj)})
        return [_sanitize_obj(v, max_list_len, max_array_elems) for v in lst]

    return obj


class DistWrapper:
    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.num_nodes = int(self.world_size // max(self.local_world_size, 1))
        self.node_rank = int(self.rank // max(self.local_world_size, 1))

        # lazy-created group for object gather on CPU
        self._obj_group = None

    def _get_obj_group(self):
        # Create a GLOO group lazily (must be called by all ranks).
        if self._obj_group is None:
            try:
                self._obj_group = dist.new_group(backend="gloo")
            except Exception:
                # fallback: use default group
                self._obj_group = None
        return self._obj_group

    def all_gather_object(self, obj, group=None):
        """
        Gather python objects safely.
        - sanitize payload to avoid pathological huge buffers
        - prefer a GLOO group so object collectives do NOT allocate on GPU/NCCL
        """
        if self.world_size > 1 and distributed_available():
            obj = _sanitize_obj(obj)
            obj_list = [None for _ in range(self.world_size)]
            use_group = group if group is not None else self._get_obj_group()
            dist.all_gather_object(obj_list, obj, group=use_group)
            return obj_list
        else:
            return [obj]


DIST_WRAPPER = DistWrapper()


def traverse_and_aggregate(dict_list, aggregation_func=None):
    merged_dict = {}
    all_keys = set().union(*dict_list)
    for key in all_keys:
        agg_value = [m[key] for m in dict_list if key in m]
        if isinstance(agg_value[0], dict):
            merged_dict[key] = traverse_and_aggregate(agg_value, aggregation_func=aggregation_func)
        else:
            if aggregation_func is not None:
                agg_value = aggregation_func(agg_value)
            merged_dict[key] = agg_value
    return merged_dict


def gather_and_merge(metrics, aggregation_func=None):
    gathered_metrics = DIST_WRAPPER.all_gather_object(metrics)
    merged_metrics = traverse_and_aggregate(gathered_metrics, aggregation_func)
    return merged_metrics
