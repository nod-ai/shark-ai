from enum import Enum
from typing import Optional, Callable
import random
import math
import logging

from . import common


class SortMethods(str, Enum):
    no_sort = "no-sort"
    shuffle = "shuffle"
    heuristic = "heuristic"


def is_pow2(x) -> bool:
    # Return True if is power of 2.
    return True if (x > 0 and (x & (x - 1)) == 0) else False


def is_mult_simd_num(x, simd_num=4) -> bool:
    # Return True if is a multiple of 4 (number of SIMDs in a CU).
    return True if (x % simd_num == 0) else False


def arith_intensity(x, y, z) -> float:
    num_flops = 2 * x * y * z
    num_byte_access = 2 * (x * y + y * z + x * z)
    return num_flops / num_byte_access


def quantization_inefficiency(knob, cu_num=304) -> float:
    # WG = M/m * N/n.
    wg = lambda knob: (knob.M / knob.tile_m) * (knob.N / knob.tile_n)
    # Quantization Inefficency = [ceil(WG/CU) - WG/CU] / ceil(WG/CU), ~0 is good.
    quantization_inefficiency = (
        math.ceil(wg(knob) / cu_num) - wg(knob) / cu_num
    ) / math.ceil(wg(knob) / cu_num)
    return quantization_inefficiency


def llvm_gpu_vector_distribute_contraction_sort_key(
    knob: common.LLVMGPUVectorDistributeContractionKnobs,
):
    pow2_rank = 0 if is_pow2(knob.tile_k) else 1
    mult_simd_num_rank = (
        0 if is_mult_simd_num(knob.subgroup_m_cnt * knob.subgroup_n_cnt) else 1
    )

    return (
        pow2_rank,
        mult_simd_num_rank,
        arith_intensity(
            knob.intrinsic_mn, knob.intrinsic_mn, knob.intrinsic_k
        ),  # Lower is better.
        quantization_inefficiency(knob),  # Lower is better.
    )


SORT_KEY_MAP: dict[type[common.KnobAssignment], Callable] = {
    common.LLVMGPUVectorDistributeContractionKnobs: llvm_gpu_vector_distribute_contraction_sort_key,
    # TODO: Add key() for conv and atten and other knobs.
}


def sorting_handler(
    knobs: Optional[list[common.KnobAssignment]],
    sorting: SortMethods,
    key_fn: Optional[Callable] = None,
) -> list[int]:
    """
    Returns a list of indices representing the new order relative to the original list.
    Example: ['a', 'b', 'c'] -> ['b', 'a', 'c'], return [1, 0, 2]
    """
    logging.debug(f"Selected sorting method: {sorting}")

    if not knobs:
        return []

    original_order = list(range(len(knobs)))  # Identity mapping.

    if sorting == SortMethods.no_sort:
        return original_order

    if sorting == SortMethods.shuffle:
        indices = list(range(len(knobs)))
        random.shuffle(indices)
        return indices

    if sorting == SortMethods.heuristic:
        # Auto set a sort key function based on the knob type.
        knob_type = type(knobs[0])
        key_fn = key_fn if key_fn else SORT_KEY_MAP.get(knob_type)
        if key_fn is None:
            logging.warning(f"No sort key defined for knob type {knob_type.__name__}.")
            return original_order
        logging.debug(f"Selected sort key: {key_fn.__name__}")

        indexed_list = list(enumerate(knobs))
        sorted_list = sorted(indexed_list, key=lambda pair: key_fn(pair[1]))
        indices = [i for i, _ in sorted_list]
        return indices

    logging.warning(f"Unknown sort method {sorting}, skip sorting.")
    return original_order
