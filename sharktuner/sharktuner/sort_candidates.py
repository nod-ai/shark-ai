from enum import Enum
from typing import Optional
from . import common
import random
import math
import logging


class SortMethods(str, Enum):
    no_sort = "no-sort"
    shuffle = "shuffle"
    heuristic = "heuristic"


def LLVMGPUVectorDistributeContractionSortKey(
    knob: common.LLVMGPUVectorDistributeContractionKnobs,
):
    # 0 if is power of 2.
    is_pow2 = lambda x: 0 if (x > 0 and (x & (x - 1)) == 0) else 1
    # 0 if is a multiple of 4 (number of SIMDs in a CU).
    is_mult_simd_num = lambda x, simd_num=4: 0 if (x % simd_num == 0) else 1
    num_flops = lambda x, y, z: 2 * x * y * z
    num_byte_access = lambda x, y, z: 2 * (x * y + y * z + x * z)
    arith_intensity = lambda x, y, z: num_flops(x, y, z) / num_byte_access(x, y, z)
    # WG = M/m * N/n.
    wg = lambda knob: (knob.m / knob.tile_m) * (knob.n / knob.tile_n)
    # quantization Inefficency = [ceil(WG/CU) - WG/CU] / ceil(WG/CU), ~0 is good.
    quantization_inefficiency = lambda knob, cu_num=304: (
        math.ceil(wg(knob) / cu_num) - wg(knob) / cu_num
    ) / math.ceil(wg(knob) / cu_num)

    return (
        is_pow2(knob.tile_k),
        is_mult_simd_num(knob.subgroup_m_cnt * knob.subgroup_n_cnt),
        arith_intensity(
            knob.intrinsic_mn, knob.intrinsic_mn, knob.intrinsic_k
        ),  # Lower is better.
        quantization_inefficiency(knob),  # Lower is better.
    )


SORT_KEY_MAP = {
    common.LLVMGPUVectorDistributeContractionKnobs: LLVMGPUVectorDistributeContractionSortKey,
    # TODO: Add key() for conv and atten and other knobs.
}


def sorting_handler(
    l: list[common.KnobAssignment], sorting: SortMethods, key_fn: callable = None
) -> list[int]:
    """
    Returns a list of indices representing the new order relative to the original list.
    Example: ['a', 'b', 'c'] -> ['b', 'a', 'c'], return [1, 0, 2]
    """
    logging.debug(f"Selected sorting method: {sorting}")

    def return_same_order():
        logging.debug("Sorting will be skipped.")
        return list(range(len(l)))  # Identity mapping.

    if sorting == SortMethods.no_sort or not l:
        return_same_order()

    if sorting == SortMethods.shuffle:
        indices = list(range(len(l)))
        random.shuffle(indices)
        l[:] = [l[i] for i in indices]
        return indices

    if sorting == SortMethods.heuristic:
        # Auto set a sort key function based on the knob type.
        knob_type = type(l[0])
        key_fn = key_fn if key_fn else SORT_KEY_MAP.get(knob_type)
        if key_fn is None:
            logging.warning(f"No sort key defined for knob type {knob_type.__name__}.")
            return_same_order()
        logging.debug(f"Selected sort key: {key_fn.__name__}")

        indexed_list = list(enumerate(l))
        indexed_list.sort(key=lambda pair: key_fn(pair[1]))
        indices = [i for i, _ in indexed_list]
        # Reorder l in place.
        l[:] = [trace for _, trace in indexed_list]
        return indices

    return_same_order()
