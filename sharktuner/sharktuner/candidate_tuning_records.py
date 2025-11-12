from typing import Optional
from dataclasses import dataclass

from . import common, candidate_tuning_records


@dataclass
class TuningRecord:
    gen_id: int
    candidate_id: int
    knob: Optional[common.KnobAssignment] = None
    to_compile: bool = False
    compile_status: bool = False
    to_benchmark: bool = False
    benchmark_device_id: Optional[str] = None
    benchmark_queue_position: Optional[int] = None
    benchmark_status: bool = False
    baseline_benchmark_time_us: Optional[float] = None
    benchmark_time_us: Optional[float] = None
    benchmark_speedup: Optional[float] = None
    benchmark_rank_order: Optional[int] = None

def init_tuning_records(knobs: list[Optional[common.KnobAssignment]], sorted_order: list[int]) -> list[TuningRecord]:
    tuning_records: list[TuningRecord] = []
    tuning_records.append(TuningRecord(gen_id=0, candidate_id=0, to_compile=True, to_benchmark=True))

    for can_idx, gen_idx in enumerate(sorted_order, start=1):
        tr = TuningRecord(
            gen_id=gen_idx,
            candidate_id=can_idx,
            knob=knobs[gen_idx],
        )
        tuning_records.append(tr)

    return tuning_records