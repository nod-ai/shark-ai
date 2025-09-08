from dataclasses import dataclass
from dataclasses_json import dataclass_json, Undefined

# TODO (stbaione): Extend for chunked prefill
@dataclass_json(undefined=Undefined.RAISE)
@dataclass(kw_only=True)
class PrefillConfig:
    has_prefill_position: bool
