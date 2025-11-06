from enum import Enum
from typing import Optional, Callable
import random
import logging

from . import common
from iree.compiler.dialects import iree_gpu  # type: ignore


class CandidateOrderKind(str, Enum):
    no_sort = "no-sort"
    shuffle = "shuffle"
    heuristic = "heuristic"


def is_pow2(x: int) -> bool:
    # Return True if is power of 2.
    return x > 0 and (x & (x - 1)) == 0


def is_mult_simd_num(x: int, simd_num: int) -> bool:
    # Return True if is a multiple of 4 (number of SIMDs in a CU).
    return x % simd_num == 0


def arith_intensity(x: int, y: int, z: int) -> float:
    num_flops = 2 * x * y * z
    num_byte_access = 2 * (x * y + y * z + x * z)
    return num_flops / num_byte_access


def llvm_gpu_vector_distribute_contraction_sort_key(
    target_info: iree_gpu.TargetInfo,
    knob: common.LLVMGPUVectorDistributeContractionKnobs,
) -> tuple[bool, bool, float]:
    return (
        not is_pow2(knob.tile_k),
        not is_mult_simd_num(
            knob.subgroup_m_cnt * knob.subgroup_n_cnt, target_info.simds_per_workgroup
        ),
        arith_intensity(
            knob.intrinsic_mn, knob.intrinsic_mn, knob.intrinsic_k
        ),  # Lower is better.
    )


SORT_KEY_MAP: dict[type[common.KnobAssignment | None], Callable | None] = {
    common.LLVMGPUVectorDistributeContractionKnobs: llvm_gpu_vector_distribute_contraction_sort_key,
    type(None): None,
    # TODO: Add key() for conv, attention, and other dispatch kinds.
}


def reorder_assignments(
    knobs: list[Optional[common.KnobAssignment]],
    strategy: CandidateOrderKind,
    key_fn: Optional[Callable] = None,
    target_info: Optional[iree_gpu.TargetInfo] = None,
) -> list[int]:
    """
    Returns a list of indices representing the new order relative to the original list.
    Example: ['a', 'b', 'c'] -> ['b', 'a', 'c'], return [1, 0, 2]
    """
    logging.debug(f"Selected candidate ordering strategy: {strategy}")

    if not knobs:
        return []

    original_order = list(range(len(knobs)))  # Identity mapping.

    key_fn_to_use: Optional[Callable] = None
    match strategy:
        case CandidateOrderKind.no_sort:
            return original_order
        case CandidateOrderKind.shuffle:
            indices = list(range(len(knobs)))
            random.shuffle(indices)
            return indices
        case CandidateOrderKind.heuristic:
            # Auto set a sort key function based on the knob type.
            knob_type = type(knobs[0])
            key_fn_to_use = key_fn if key_fn else SORT_KEY_MAP.get(knob_type)
            if key_fn_to_use is None:
                logging.warning(
                    f"No sort key defined for knob type {knob_type.__name__}."
                )
                return original_order
            logging.debug(f"Selected sort key: {key_fn_to_use.__name__}")

            indexed_list = list(enumerate(knobs))
            # Good candidates are sorted to the front of the list.
            if not key_fn:
                assert target_info, "Failed to query target info."
                sorted_list = sorted(
                    indexed_list, key=lambda pair: key_fn_to_use(target_info, pair[1])
                )
            else:
                sorted_list = sorted(
                    indexed_list, key=lambda pair: key_fn_to_use(pair[1])
                )

            indices = [i for i, _ in sorted_list]
            return indices
        case _:
            assert False
