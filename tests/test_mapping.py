from __future__ import annotations

import numpy as np

from trt_profiler.mapping.input_mapper import InputMapper
from trt_profiler.mapping.output_mapper import OutputMapper


def test_input_mapper_renames_common_inputs() -> None:
    mapper = InputMapper({"runner": {"actual_input": "image"}})
    image = np.array([1.0], dtype=np.float32)

    mapped = mapper.map("runner", {"image": image})
    assert mapped["actual_input"] is image


def test_output_mapper_renames_runner_outputs() -> None:
    mapper = OutputMapper({"runner": {"logits": "output_0"}})
    output = np.array([1.0], dtype=np.float32)

    mapped = mapper.map("runner", {"output_0": output})
    assert mapped["logits"] is output
