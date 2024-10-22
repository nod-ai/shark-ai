# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Given an input dispatch, this code modifies the hyperparameters
# in the code and runs it.

"""
Generate candidates by tweaking op configuration for tuning.

It can be invoked in two ways:
    1. From another python script, import and call `tune()`
    2. Run this script directly from the command

Usage: ./candidate_gen.py 121.mlir -o "tuning/candidates" -l 1024 --lhs-dims=mk --rhs-dims=nk --tile-dims=mnk

"""

import argparse
import logging
import math
import pickle
import re
import z3
from dataclasses import astuple, dataclass
from enum import Enum
from os import mkdir, path, makedirs
from typing import Optional, Generator
from textwrap import indent
from abc import ABC, abstractmethod

import iree.compiler as ireec
from iree.compiler import ir
from iree.compiler.dialects import _linalg_ops_gen, _util_ops_gen


tune_logger = logging.getLogger("tune")


class DispatchKind(Enum):
    conv = 1
    mmt = 2
    contraction = 3
    batch_mmt = 4
    batch_matmul = 5
    broadcast_rhs_mmt = 6
    mmt4d = 7


class ElementType(Enum):
    i8 = 1
    i32 = 2
    f8 = 3
    f16 = 4
    f32 = 5

    @property
    def bitwidth(self) -> int:
        match self:
            case ElementType.i8 | ElementType.f8:
                return 8
            case ElementType.f16:
                return 16
            case ElementType.i32 | ElementType.f32:
                return 32
            case _:
                assert False, "unhandled case"

    def __str__(self) -> str:
        return self.name


@dataclass
class ShapedType:
    shape: list[int]
    element_type: ElementType

    def rank(self) -> int:
        return len(self.shape)

    @property
    def bitwidth(self) -> int:
        return self.element_type.bitwidth

    def __str__(self) -> str:
        dim_to_str = lambda dim: str(dim) if dim != -1 else "?"
        return "x".join(map(dim_to_str, self.shape)) + "x" + str(self.element_type)


@dataclass
class MatmulSize:
    M: int
    N: int
    K: int
    B: int = 1


@dataclass
class ProblemSize:
    matmul_size: MatmulSize
    lhs_type: ShapedType
    rhs_type: ShapedType
    res_type: ShapedType
    dispatch_kind: DispatchKind

    @property
    def MNK(self) -> tuple[int, int, int]:
        return (self.matmul_size.M, self.matmul_size.N, self.matmul_size.K)


@dataclass
class MfmaIntrinsic:
    output_type: ElementType
    m: int
    n: int
    k: int
    input_type: ElementType

    def __str__(self) -> str:
        input = str(self.input_type).upper()
        output = str(self.output_type).upper()
        return f"MFMA_{output}_{self.m}x{self.n}x{self.k}_{input}"

    @staticmethod
    def mfma_f32_16x16x16_f16():
        return MfmaIntrinsic(ElementType.f32, 16, 16, 16, ElementType.f16)

    @staticmethod
    def mfma_f32_32x32x8_f16():
        return MfmaIntrinsic(ElementType.f32, 32, 32, 8, ElementType.f16)

    @staticmethod
    def mfma_i32_16x16x32_i8():
        return MfmaIntrinsic(ElementType.i32, 16, 16, 32, ElementType.i8)

    @staticmethod
    def mfma_i32_32x32x16_i8():
        return MfmaIntrinsic(ElementType.i32, 32, 32, 16, ElementType.i8)

    @staticmethod
    def all():
        return [
            MfmaIntrinsic.mfma_f32_16x16x16_f16(),
            MfmaIntrinsic.mfma_f32_32x32x8_f16(),
            MfmaIntrinsic.mfma_i32_16x16x32_i8(),
            MfmaIntrinsic.mfma_i32_32x32x16_i8(),
        ]


class ReorderWorkgroupsStrategy(Enum):
    NONE = 0
    SWIZZLE = 1
    TRANSPOSE = 2

    def __str__(self) -> str:
        return self.name.title()


@dataclass
class GpuPipelineOptions:
    """Represents the `iree_gpu.pipeline_options` attribute"""

    prefetch_shared_memory: Optional[bool] = None
    no_reduce_shared_memory_bank_conflicts: Optional[bool] = None
    reorder_workgroups_strategy: Optional[ReorderWorkgroupsStrategy] = None

    def all_default(self) -> bool:
        return all(x is None for x in astuple(self))

    def __str__(self) -> str:
        options: list[str] = []
        if self.prefetch_shared_memory is not None:
            options.append(
                f"prefetch_shared_memory = {str(self.prefetch_shared_memory).lower()}"
            )
        if self.no_reduce_shared_memory_bank_conflicts is not None:
            options.append(
                f"no_reduce_shared_memory_bank_conflicts = {str(self.no_reduce_shared_memory_bank_conflicts).lower()}"
            )
        if self.reorder_workgroups_strategy is not None:
            options.append(
                f"reorder_workgroups_strategy = {self.reorder_workgroups_strategy}"
            )

        return f"#iree_gpu.pipeline_options<{', '.join(options)}>"


@dataclass
class BaseConfiguration:
    tile_sizes: list[int]

    @abstractmethod
    def get_mmt_tile_sizes(self) -> list[int]:
        return self.tile_sizes

    @abstractmethod
    def get_batch_mmt_tile_sizes(self) -> list[int]:
        return [1] + self.tile_sizes

    def get_contract_tile_sizes(self, tile_dims: list[str]) -> list[int]:
        m, n, k = self.tile_sizes
        tile_size = [1] * len(tile_dims)
        for idx, dim in enumerate(tile_dims):
            if dim == "m":
                tile_size[idx] = m
            if dim == "n":
                tile_size[idx] = n
            if dim == "k":
                tile_size[idx] = k
        return tile_size

    @abstractmethod
    def get_pipeline_config(self) -> str:
        pass

    @abstractmethod
    def get_intrinsic_config(self) -> str:
        pass

    def get_base_mlir_config(self, tile_sizes: list[int]) -> str:
        tile_sizes_str = ", ".join(map(str, tile_sizes))
        return f"""
    %config = transform.param.constant #iree_codegen.compilation_info<
        lowering_config = #iree_codegen.lowering_config<tile_sizes = [[{tile_sizes_str}]]>,
    """

    @abstractmethod
    def get_mlir_config(self, tile_sizes: list[int]) -> str:
        base_config = self.get_base_mlir_config(tile_sizes)
        full_config = f"""
        translation_info = #iree_codegen.translation_info<NONE>
        > -> !transform.any_param
    """

        return base_config + full_config

    @abstractmethod
    def apply_configuration(self, template: list[str], tile_sizes: list[int]) -> str:
        pass


@dataclass
class LLVMGPUConfiguration(BaseConfiguration):
    subgroup_size: int
    workgroup_size: list[int]
    intrinsic: MfmaIntrinsic
    subgroup_m_count: int
    subgroup_n_count: int
    gpu_pipeline_options: GpuPipelineOptions
    waves_per_eu: int

    def get_pipeline_config(self) -> str:
        extra_config = ""
        if not self.gpu_pipeline_options.all_default():
            extra_config += f", gpu_pipeline_options = {self.gpu_pipeline_options}"
        if self.waves_per_eu != 2:
            extra_config += (
                f', llvm_func_attrs = {{"amdgpu-waves-per-eu" = "{self.waves_per_eu}"}}'
            )

        return extra_config

    def get_intrinsic_config(self) -> str:
        return str(self.intrinsic)

    def get_mlir_config(self, tile_sizes: list[int]) -> str:
        base_config = self.get_base_mlir_config(tile_sizes)
        wg_x, wg_y, wg_z = self.workgroup_size

        backend_config = f"""
        translation_info = #iree_codegen.translation_info<LLVMGPUVectorDistribute
            workgroup_size = [{wg_x}, {wg_y}, {wg_z}] subgroup_size = {self.subgroup_size},
            {{mma_schedule = #iree_gpu.mma_schedule<
                intrinsic = #iree_gpu.mma_layout<{self.get_intrinsic_config()}>,
                subgroup_m_count = {self.subgroup_m_count}, subgroup_n_count = {self.subgroup_n_count}>
            {self.get_pipeline_config()}}}>
        > -> !transform.any_param
    """

        return base_config + backend_config

    def apply_configuration(self, template: list[str], tile_sizes: list[int]) -> str:
        tune_logger.info(f"Applying: {self}")

        expr0 = re.compile(
            r"<intrinsic = #iree_gpu\.mma_layout<(.+)>, subgroup_m_count = ([0-9]+), subgroup_n_count = ([0-9]+)>"
        )
        expr1 = re.compile(
            r"LLVMGPUVectorDistribute workgroup_size = \[.+\] subgroup_size = ([0-9]+),"
        )
        expr2 = re.compile(r"tile_sizes = \[\[([0-9]+)(, ([0-9]+))+\]\]")
        expr3 = re.compile(
            r"gpu_pipeline_options = #iree_gpu\.pipeline_options<([^>]*)>"
        )
        expr4 = re.compile(r"\"amdgpu-waves-per-eu\" = \"([0-9])\"")

        repl0 = f"<intrinsic = #iree_gpu.mma_layout<{self.intrinsic}>, subgroup_m_count = {self.subgroup_m_count}, subgroup_n_count = {self.subgroup_n_count}>"
        repl1 = f'LLVMGPUVectorDistribute workgroup_size = [{", ".join(map(str, self.workgroup_size))}] subgroup_size = {self.subgroup_size},'
        repl2 = f'tile_sizes = [[{", ".join(map(str, tile_sizes))}]]'
        repl3 = f"gpu_pipeline_options = {self.gpu_pipeline_options}"
        repl4 = f'"amdgpu-waves-per-eu" = "{self.waves_per_eu}"'

        new_mlir = ""
        for line in template:
            if "intrinsic =" in line:
                line = re.sub(expr0, repl0, line)
            if "LLVMGPUVectorDistribute " in line:
                line = re.sub(expr1, repl1, line)
            if "tile_sizes" in line:
                line = re.sub(expr2, repl2, line)
            if "gpu_pipeline_options =" in line:
                line = re.sub(expr3, repl3, line)
            if "amdgpu-waves-per-eu" in line:
                line = re.sub(expr4, repl4, line)
            new_mlir += line

        return new_mlir


@dataclass
class LLVMCPUConfiguration(BaseConfiguration):
    def get_mlir_config(self, tile_sizes: list[int]) -> str:
        base_config = self.get_base_mlir_config(tile_sizes)

        backend_config = f"""
        translation_info = #iree_codegen.translation_info<Mmt4dTilingExpert>
        > -> !transform.any_param
    """

        return base_config + backend_config

    def apply_configuration(self, template: list[str], tile_sizes: list[int]) -> str:
        tune_logger.info(f"Applying: {self}")

        expr0 = re.compile(r"tile_sizes = \[\[([0-9]+)(, ([0-9]+))+\]\]")

        repl0 = f'tile_sizes = [[{", ".join(map(str, tile_sizes))}]]'

        new_mlir = ""
        for line in template:
            if "tile_sizes" in line:
                line = re.sub(expr0, repl0, line)
            new_mlir += line

        return new_mlir


class MlirRegex(Enum):
    ssa_value = r"%[a-zA-Z0-9-_]+"
    tensor_type = r"tensor<(([0-9]+x)+((f|i)[0-9]+))>"
    device_target = r'#hal\.device\.target<"(?P<target>[a-zA-Z0-9-_]+)"'

    def __str__(self) -> str:
        return self.value

    @staticmethod
    def dps_ins_two_args() -> str:
        return rf"ins\({MlirRegex.ssa_value}, {MlirRegex.ssa_value} : (?P<LHS>{MlirRegex.tensor_type}), (?P<RHS>{MlirRegex.tensor_type})\)"

    @staticmethod
    def dps_outs_one_arg() -> str:
        return rf"outs\({MlirRegex.ssa_value} : (?P<RES>{MlirRegex.tensor_type})\)"


def read_input_mlir(filename: str) -> list[str]:
    with open(filename, "r") as f:
        return f.readlines()


@dataclass
class ConvDimInfo:
    n: int
    oh: int
    ow: int
    oc: int
    fh: int
    fw: int
    ic: int

    @staticmethod
    def from_rhs_res(rhs_shaped_type: ShapedType, res_shaped_type: ShapedType):
        n, oh, ow, oc = res_shaped_type.shape
        fh, fw, ic, _ = rhs_shaped_type.shape
        return ConvDimInfo(n, oh, ow, oc, fh, fw, ic)

    @staticmethod
    def from_problem_size(problem_size: ProblemSize):
        return ConvDimInfo.from_rhs_res(problem_size.rhs_type, problem_size.res_type)


def parse_tensor_type(tensor_type: str) -> ShapedType:
    shape_match = re.search(str(MlirRegex.tensor_type), tensor_type)
    assert shape_match

    shape_str = shape_match.group(1)
    dims_and_elem = shape_str.split("x")
    dims = [int(x) for x in dims_and_elem[:-1]]
    elem = dims_and_elem[-1]
    str_to_elem_ty = {x.name: x for x in ElementType}
    return ShapedType(dims, str_to_elem_ty[elem])


def get_compatible_mfma_intrinsics(problem_size: ProblemSize) -> list[MfmaIntrinsic]:
    def is_compatible(intrinsic: MfmaIntrinsic) -> bool:
        if problem_size.res_type.element_type != intrinsic.output_type:
            return False
        if problem_size.dispatch_kind != DispatchKind.batch_matmul:
            if problem_size.lhs_type.element_type != intrinsic.input_type:
                return False
            if problem_size.rhs_type.element_type != intrinsic.input_type:
                return False
        return True

    return list(filter(is_compatible, MfmaIntrinsic.all()))


def get_mfma_intrinsic_constraints(
    problem_size: ProblemSize,
    intrinsic_m: z3.ArithRef,
    intrinsic_n: z3.ArithRef,
    intrinsic_k: z3.ArithRef,
) -> z3.BoolRef:
    compatible_intrinsics = get_compatible_mfma_intrinsics(problem_size)
    assert len(compatible_intrinsics) > 0, "No compatible intrinsics found"
    return z3.Or(
        *(
            z3.And(intrinsic_m == mfma.m, intrinsic_n == mfma.n, intrinsic_k == mfma.k)
            for mfma in compatible_intrinsics
        )
    )


def get_dispatch_constraints(
    problem_size: ProblemSize,
    tile_m: z3.ArithRef,
    tile_n: z3.ArithRef,
    tile_k: z3.ArithRef,
) -> list[z3.BoolRef]:
    if problem_size.dispatch_kind != DispatchKind.conv:
        return []

    dim_info = ConvDimInfo.from_problem_size(problem_size)
    conv_constraints = []
    # WARNING: This sometimes makes the constraints UNSAT for some reason.
    conv_constraints += [tile_m <= dim_info.ow]
    conv_constraints += [tile_n <= dim_info.oc]
    conv_constraints += [tile_k <= dim_info.ic]
    return conv_constraints


def calculate_shared_memory_usage_in_bytes(
    problem_size: ProblemSize,
    m: int | z3.ArithRef,
    n: int | z3.ArithRef,
    k: int | z3.ArithRef,
) -> int | z3.ArithRef:
    lhs_memory = m * k * (problem_size.lhs_type.bitwidth // 8)
    rhs_memory = k * n * (problem_size.rhs_type.bitwidth // 8)
    return lhs_memory + rhs_memory


class SolutionGenerationStrategy(ABC):
    @abstractmethod
    def generate_solutions(
        self, problem_size: ProblemSize, num_subgroups: int
    ) -> Generator[BaseConfiguration, None, None]:
        pass


class LLVMGPUSolutionStrategy(SolutionGenerationStrategy):
    def generate_solutions(
        self,
        problem_size: ProblemSize,
        num_subgroups: int,
    ):
        M, N, K = problem_size.MNK
        tune_logger.info(f"{M},{N},{K}")
        m, n, k = z3.Int("m"), z3.Int("n"), z3.Int("k")
        subgroup_size = z3.Int("subgroup_size")
        intrinsic_mn = z3.Int("intrinsic_mn")
        intrinsic_k = z3.Int("intrinsic_k")
        wg_x, wg_y, wg_z = z3.Int("wg_x"), z3.Int("wg_y"), z3.Int("wg_z")
        sg_m_cnt = z3.Int("sg_m_cnt")
        sg_n_cnt = z3.Int("sg_n_cnt")
        waves_per_eu = z3.Int("waves_per_eu")
        all_vars = [
            m,
            n,
            k,
            subgroup_size,
            intrinsic_mn,
            intrinsic_k,
            wg_x,
            wg_y,
            wg_z,
            sg_m_cnt,
            sg_n_cnt,
            waves_per_eu,
        ]

        solver = z3.Solver()
        constraints = self.generate_constraints(
            problem_size,
            [m, n, k],
            num_subgroups,
            subgroup_size,
            [intrinsic_mn, intrinsic_k],
            [wg_x, wg_y, wg_z],
            sg_m_cnt,
            sg_n_cnt,
            waves_per_eu,
        )
        solver.add(z3.simplify(z3.And(constraints)))
        tune_logger.debug(f"Initial constraints: {solver}")
        i = 0
        while solver.check() == z3.sat:
            model = solver.model()
            lookup = lambda var: model[var].as_long()

            config = LLVMGPUConfiguration(
                subgroup_size=lookup(subgroup_size),
                workgroup_size=[lookup(wg_x), lookup(wg_y), lookup(wg_z)],
                tile_sizes=[lookup(m), lookup(n), lookup(k)],
                subgroup_m_count=lookup(sg_m_cnt),
                subgroup_n_count=lookup(sg_n_cnt),
                gpu_pipeline_options=GpuPipelineOptions(),
                waves_per_eu=lookup(waves_per_eu),
                intrinsic=MfmaIntrinsic(
                    problem_size.res_type.element_type,
                    lookup(intrinsic_mn),
                    lookup(intrinsic_mn),
                    lookup(intrinsic_k),
                    problem_size.lhs_type.element_type,
                ),
            )

            solver.add(
                z3.simplify(z3.Not(z3.And(list(x == model[x] for x in all_vars))))
            )
            i += 1
            yield config

    def generate_constraints(
        self,
        problem_size: ProblemSize,
        tile_sizes,
        num_subgroups,
        subgroup_size,
        intrinsic_size,
        workgroup_size,
        subgroup_m_count,
        subgroup_n_count,
        waves_per_eu,
    ):
        M, N, K = (
            problem_size.matmul_size.M,
            problem_size.matmul_size.N,
            problem_size.matmul_size.K,
        )
        m, n, k = tile_sizes
        intrinsic_mn, intrinsic_k = intrinsic_size
        wg_x, wg_y, wg_z = workgroup_size
        wg_threads = z3.Int("wg_threads")
        constraints = [wg_threads == wg_x * wg_y * wg_z]
        constraints += [subgroup_size == 64, wg_threads <= 1024]
        constraints += [
            get_mfma_intrinsic_constraints(
                problem_size, intrinsic_mn, intrinsic_mn, intrinsic_k
            )
        ]
        subgroup_k_count = 1
        constraints += [
            m >= intrinsic_mn,
            m <= 512,
            m <= M,
        ]
        constraints += [n >= intrinsic_mn, n <= 512, n <= N, N % n == 0]
        constraints += [k >= intrinsic_k, k <= 512, k <= K, K % k == 0]
        for x in (subgroup_m_count, subgroup_n_count):
            constraints += [x >= 1, x <= 32]

        subgroup_m_tile_count = z3.Int("sg_m_tcnt")
        subgroup_n_tile_count = z3.Int("sg_n_tcnt")
        subgroup_k_tile_count = z3.Int("sg_k_tcnt")
        for x in (subgroup_m_tile_count, subgroup_n_tile_count, subgroup_k_tile_count):
            constraints += [x >= 1, x <= 32]

        constraints += [m == subgroup_m_count * subgroup_m_tile_count * intrinsic_mn]
        constraints += [n == subgroup_n_count * subgroup_n_tile_count * intrinsic_mn]
        constraints += [k == subgroup_k_count * subgroup_k_tile_count * intrinsic_k]
        constraints += [wg_x == subgroup_size * subgroup_n_count]
        constraints += [wg_y == subgroup_m_count]
        constraints += [wg_z == subgroup_k_count]
        constraints += [z3.Or(wg_x <= n, wg_x <= m)]
        constraints += [k % intrinsic_mn == 0]
        constraints += [(k * n) % wg_threads == 0]
        constraints += [(k * m) % wg_threads == 0]
        subgroups = subgroup_m_count * subgroup_n_count
        if num_subgroups > 0:
            constraints += [subgroups == num_subgroups]
        else:
            constraints += [subgroups >= 1, subgroups <= 10]

        constraints += [waves_per_eu == 2]
        # constraints += [z3.Or(waves_per_eu == 2, waves_per_eu == 3, waves_per_eu == 4)]

        shared_memory = calculate_shared_memory_usage_in_bytes(problem_size, m, n, k)
        constraints += [shared_memory <= 65536]

        constraints += get_dispatch_constraints(problem_size, m, n, k)

        return constraints


class LLVMCPUSolutionStrategy(SolutionGenerationStrategy):
    def generate_solutions(
        self,
        problem_size: ProblemSize,
        num_subgroups: int,
    ):
        M, N, K = problem_size.MNK
        tune_logger.info(f"{M},{N},{K}")
        m, n, k = z3.Int("m"), z3.Int("n"), z3.Int("k")
        all_vars = [
            m,
            n,
            k,
        ]

        solver = z3.Solver()
        constraints = self.generate_constraints(
            problem_size,
            [m, n, k],
        )
        solver.add(z3.simplify(z3.And(constraints)))
        tune_logger.debug(f"Initial constraints: {solver}")
        i = 0
        while solver.check() == z3.sat:
            model = solver.model()
            lookup = lambda var: model[var].as_long()

            config = LLVMCPUConfiguration(
                tile_sizes=[lookup(m), lookup(n), lookup(k)],
            )

            solver.add(
                z3.simplify(z3.Not(z3.And(list(x == model[x] for x in all_vars))))
            )
            i += 1
            yield config

    def generate_constraints(
        self,
        problem_size: ProblemSize,
        tile_sizes,
    ) -> list:
        M, N, K = (
            problem_size.matmul_size.M,
            problem_size.matmul_size.N,
            problem_size.matmul_size.K,
        )
        m, n, k = tile_sizes

        constraints = []
        constraints += [m >= 1, m <= 512, m <= M]
        constraints += [n >= 1, n <= 512, n <= N, N % n == 0]
        constraints += [n >= 1, k <= 512, k <= K, K % k == 0]

        shared_memory = calculate_shared_memory_usage_in_bytes(problem_size, m, n, k)
        constraints += [shared_memory <= 65536]

        constraints += get_dispatch_constraints(problem_size, m, n, k)

        return constraints


@dataclass
class SolutionStrategyFactory:
    @staticmethod
    def create_strategy(mlir_text: str) -> SolutionGenerationStrategy:
        match = re.search(MlirRegex.device_target.value, mlir_text)

        if not match:
            raise ValueError("No target found")

        target = match.group("target")
        return SolutionStrategyFactory.get_strategy(target)

    @staticmethod
    def get_strategy(target: str) -> SolutionGenerationStrategy:
        if target == "local":
            return LLVMCPUSolutionStrategy()
        else:
            return LLVMGPUSolutionStrategy()


def get_default_output_dir() -> str:
    from datetime import datetime

    return "tuning_" + datetime.now().strftime("%Y_%m_%d_%H_%M")


def parse_mlir(mlir_text: str) -> ir.Module:
    mlir_module = None
    with ireec.ir.Context() as context:
        try:
            mlir_module = ireec.ir.Module.parse(mlir_text)
            tune_logger.info("MLIR parsing successful!")
        except ireec.ir.MLIRError as e:
            tune_logger.error(f"Error parsing MLIR: {e}")
            raise RuntimeError(f"Error parsing MLIR: {e}")

    return mlir_module


@dataclass
class MLIRTransformation:
    """Transformation of MLIR context"""

    template: str
    modified: str
    embeddable: str


class DispatchTuner(ABC):
    @abstractmethod
    def supports(self, op_name: str) -> bool:
        """Check if the tuner can handle the type of operation represented by the input string."""
        pass

    @abstractmethod
    def get_shapes(self, template: list[str]) -> ProblemSize:
        """Extract problem size of thge operation."""
        pass

    @abstractmethod
    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        """Apply parameter transformations to the operation."""
        pass


@dataclass
class OpWalkResult:
    was_interrupted: bool = False
    dispatch_tuner: Optional[DispatchTuner] = None


class DispatchTunerRegistry:
    def __init__(self):
        self.registry = set()

    def register(self, dispatch_tuners: list[DispatchTuner]) -> None:
        for dispatch_tuner in dispatch_tuners:
            self.registry.add(dispatch_tuner)

    def validate_translation(self, attrs: list[ir.NamedAttribute]) -> bool:
        for attr in attrs:
            if (attr.name == "translation_info") and (
                "LLVMGPUVectorDistribute" in str(attr.attr)
            ):
                return True

            if (attr.name == "translation_info") and (
                "Mmt4dTilingExpert" in str(attr.attr)
            ):
                return True

        assert False, "Translation info not supported"

    def find_handler(self, op_name: str) -> DispatchTuner:
        for dispatch_tuner in self.registry:
            if dispatch_tuner.supports(op_name):
                return dispatch_tuner
        assert False, "Dispatch kind not supported"


class MmtTuner(DispatchTuner):
    def supports(self, op_name: str) -> bool:
        return "matmul_transpose_b" in op_name

    def get_shapes(self, template: list[str]) -> ProblemSize:
        mmt_re = None
        dps = None
        for line in template:

            if "linalg.generic" not in line:
                continue
            if r'iterator_types = ["parallel", "parallel", "reduction"]' not in line:
                continue
            # ins(%13, %14 : tensor<2048x1280xf16>, tensor<1280x1280xf16>) outs(%19 : tensor<2048x1280xf32>)
            mmt_re = rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
            dps = re.search(mmt_re, line)
            if dps is None:
                continue

            lhs_tensor_type = dps.group("LHS")
            rhs_tensor_type = dps.group("RHS")
            lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
            assert lhs_shaped_type.rank() == 2
            lhs_M, lhs_K = lhs_shaped_type.shape

            rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
            assert rhs_shaped_type.rank() == 2
            rhs_N, rhs_K = rhs_shaped_type.shape

            assert lhs_shaped_type.element_type == rhs_shaped_type.element_type
            assert lhs_K == rhs_K

            res_tensor_type = dps.group("RES")
            res_shaped_type = parse_tensor_type(res_tensor_type)
            assert res_shaped_type.rank() == 2
            res_M, res_N = res_shaped_type.shape

            assert lhs_M == res_M
            assert rhs_N == res_N

            matmul_size = MatmulSize(
                lhs_shaped_type.shape[0],
                rhs_shaped_type.shape[0],
                lhs_shaped_type.shape[1],
            )
            return ProblemSize(
                matmul_size,
                lhs_type=lhs_shaped_type,
                rhs_type=rhs_shaped_type,
                res_type=res_shaped_type,
                dispatch_kind=DispatchKind.mmt,
            )

        assert mmt_re
        assert dps, f"'{mmt_re}' not found in given context"

    def get_transform_function_mmt(
        self,
        problem_size: ProblemSize,
        functionName: str,
        configuration: BaseConfiguration,
    ) -> str:
        tile_sizes = configuration.get_mmt_tile_sizes()
        config_mlir = configuration.get_mlir_config(tile_sizes)

        return f"""
    transform.named_sequence @{functionName}(%matmul: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
    %mmt = transform.include @match_mmt_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<{problem_size.lhs_type}> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
    {config_mlir}
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
    }}
    """

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        modified = indent(
            self.get_transform_function_mmt(
                problem_size, f"match_mmt_{M}x{N}x{K}", configuration
            ),
            "//   ",
        )
        modified += configuration.apply_configuration(
            template, configuration.get_mmt_tile_sizes()
        )
        embeddable = indent(
            self.get_transform_function_mmt(problem_size, f"match_op", configuration),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)


class Mmt4dTuner(DispatchTuner):
    def supports(self, op_name: str) -> bool:
        return "mmt4d" in op_name

    def get_shapes(self, template: list[str]) -> ProblemSize:
        mmt_re = None
        dps = None
        for line in template:

            if "linalg.mmt4d" not in line:
                continue
            # ins(%3, %4 : tensor<256x1280x8x1xf16>, tensor<1280x1280x8x1xf16>) outs(%6 : tensor<256x1280x8x8xf32>)
            mmt_re = rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"

            dps = re.search(mmt_re, line)
            if dps is None:
                continue

            lhs_tensor_type = dps.group("LHS")
            rhs_tensor_type = dps.group("RHS")
            lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
            assert lhs_shaped_type.rank() == 4
            lhs_M = lhs_shaped_type.shape[0] * lhs_shaped_type.shape[2]
            lhs_K = lhs_shaped_type.shape[1] * lhs_shaped_type.shape[3]

            rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
            assert rhs_shaped_type.rank() == 4
            rhs_N = rhs_shaped_type.shape[0] * rhs_shaped_type.shape[2]
            rhs_K = rhs_shaped_type.shape[1] * rhs_shaped_type.shape[3]
            assert lhs_shaped_type.element_type == rhs_shaped_type.element_type
            assert lhs_K == rhs_K

            res_tensor_type = dps.group("RES")
            res_shaped_type = parse_tensor_type(res_tensor_type)
            assert res_shaped_type.rank() == 4
            res_M = res_shaped_type.shape[0] * res_shaped_type.shape[2]
            res_N = res_shaped_type.shape[1] * res_shaped_type.shape[3]

            assert lhs_K == rhs_K
            assert lhs_M == res_M
            assert rhs_N == res_N

            matmul_size = MatmulSize(
                lhs_M,
                rhs_N,
                lhs_K,
            )
            return ProblemSize(
                matmul_size,
                lhs_type=lhs_shaped_type,
                rhs_type=rhs_shaped_type,
                res_type=res_shaped_type,
                dispatch_kind=DispatchKind.mmt4d,
            )

        assert mmt_re
        assert dps, f"'{mmt_re}' not found in given context"

    def get_transform_function_mmt(
        self,
        problem_size: ProblemSize,
        functionName: str,
        configuration: BaseConfiguration,
    ) -> str:
        tile_sizes = configuration.get_mmt_tile_sizes()
        config_mlir = configuration.get_mlir_config(tile_sizes)

        return f"""
    transform.named_sequence @{functionName}(%matmul: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
    %mmt = transform.include @match_mmt4d_f16_f16_f32 failures(propagate) (%matmul) : (!transform.any_op) -> !transform.any_op
    %lhs = transform.get_operand %matmul[0] : (!transform.any_op) -> !transform.any_value
    %rhs = transform.get_operand %matmul[1] : (!transform.any_op) -> !transform.any_value
    transform.iree.match.cast_compatible_type %lhs = tensor<{problem_size.lhs_type}> : !transform.any_value
    transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
    {config_mlir}
    transform.yield %matmul, %config : !transform.any_op, !transform.any_param
    }}
    """

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        modified = indent(
            self.get_transform_function_mmt(
                problem_size, f"match_mmt4d_{M}x{N}x{K}", configuration
            ),
            "//   ",
        )
        modified += configuration.apply_configuration(
            template, configuration.get_mmt_tile_sizes()
        )
        embeddable = indent(
            self.get_transform_function_mmt(problem_size, f"match_op", configuration),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)


class ConvTuner(DispatchTuner):
    def supports(self, op_name: str) -> bool:
        return "conv_2d_nhwc_hwcf" in op_name

    def get_conv_tile_sizes(self, configuration: BaseConfiguration) -> list[int]:
        m, n, k = configuration.tile_sizes
        batch = 1
        fh = 1
        fw = 1

        oh = 1

        oc = n
        ow = m
        ic = k
        return [batch, oh, ow, oc, fh, fw, ic]

    def get_shapes(self, template: list[str]) -> ProblemSize:
        for line in template:
            if "linalg.conv_2d_nhwc_hwcf" not in line:
                continue

            # ins(%19, %20 : tensor<2x34x34x1280xf16>, tensor<3x3x1280x1280xf16>) outs (%27 : tensor<2x32x32x1280xf32>)
            conv_re = (
                rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
            )
            dps = re.search(conv_re, line)
            if dps is None:
                continue

            lhs_tensor_type = dps.group("LHS")
            rhs_tensor_type = dps.group("RHS")
            lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
            assert lhs_shaped_type.rank() == 4

            rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
            assert rhs_shaped_type.rank() == 4

            res_tensor_type = dps.group("RES")
            res_shaped_type = parse_tensor_type(res_tensor_type)
            assert res_shaped_type.rank() == 4

            # int64_t n = outputShape[0];
            # int64_t oh = outputShape[1];
            # int64_t ow = outputShape[2];
            # int64_t oc = outputShape[3];
            # int64_t fh = filterShape[0];
            # int64_t fw = filterShape[1];
            # int64_t ic = filterShape[2];
            dim_info = ConvDimInfo.from_rhs_res(rhs_shaped_type, res_shaped_type)
            return ProblemSize(
                MatmulSize(
                    M=dim_info.oh * dim_info.ow,
                    N=dim_info.oc,
                    K=dim_info.fh * dim_info.fw * dim_info.ic,
                    B=dim_info.n,
                ),
                lhs_shaped_type,
                rhs_shaped_type,
                res_shaped_type,
                DispatchKind.conv,
            )

        assert False, "Shape not found"

    # int64_t n = outputShape[0];
    # int64_t oh = outputShape[1];
    # int64_t ow = outputShape[2];
    # int64_t oc = outputShape[3];
    # int64_t fh = filterShape[0];
    # int64_t fw = filterShape[1];
    # int64_t ic = filterShape[2];
    def get_transform_function_conv(
        self,
        problem_size: ProblemSize,
        functionName: str,
        configuration: BaseConfiguration,
    ) -> str:
        dynamic_batch_input_ty = problem_size.lhs_type
        dynamic_batch_input_ty.shape = dynamic_batch_input_ty.shape.copy()
        dynamic_batch_input_ty.shape[0] = -1

        dynamic_batch_output_ty = problem_size.res_type
        dynamic_batch_output_ty.shape = dynamic_batch_output_ty.shape.copy()
        dynamic_batch_output_ty.shape[0] - 1

        input = f"tensor<{dynamic_batch_input_ty}>"
        filter = f"tensor<{problem_size.rhs_type}>"
        output = f"tensor<{dynamic_batch_output_ty}>"

        tile_sizes = self.get_conv_tile_sizes(configuration)
        config_mlir = configuration.get_mlir_config(tile_sizes)

        return f"""
    transform.named_sequence @{functionName}(%conv: !transform.any_op {{transform.readonly}})
    -> (!transform.any_op, !transform.any_param) {{
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %conv {{
    ^bb0(%lhs: {input}, %rhs: {filter}, %out: {output}):
        %13 = linalg.conv_2d_nhwc_hwcf {{dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}}
        ins(%lhs, %rhs : {input}, {filter})
        outs(%out : {output}) -> {output}
    }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
        {config_mlir}
    transform.yield %conv, %config : !transform.any_op, !transform.any_param
    }}
    """

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        conv_dims = ConvDimInfo.from_problem_size(problem_size)
        modified = indent(
            self.get_transform_function_conv(
                problem_size,
                f"match_conv_2d_nhwc_hwcf_Bx{conv_dims.oh}x{conv_dims.ow}x{conv_dims.oc}x{conv_dims.fh}x{conv_dims.fw}x{conv_dims.ic}",
                configuration,
            ),
            "//   ",
        )
        modified += configuration.apply_configuration(
            template, self.get_conv_tile_sizes(configuration)
        )
        embeddable = indent(
            self.get_transform_function_conv(problem_size, f"match_op", configuration),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)


class ContractionTuner(DispatchTuner):
    def __init__(self, lhs_dims: str, rhs_dims: str, tile_dims: str):
        self.lhs_dims = lhs_dims
        self.rhs_dims = rhs_dims
        self.tile_dims = tile_dims

    def supports(self, op_name: str) -> bool:
        return "matmul_like" in op_name

    def is_broadcast_rhs_mmt_op(self, line: str) -> bool:
        if "linalg.generic" not in line:
            return False
        if (
            r'iterator_types = ["parallel", "parallel", "parallel", "reduction"]'
            not in line
        ):
            return False
        if (
            r"indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>"
            not in line
        ):
            return False
        return True

    def is_broadcast_rhs_mmt(self, template: list[str]) -> bool:
        return any(self.is_broadcast_rhs_mmt_op(line) for line in template)

    def get_shapes_broadcast_rhs_mmt(self, template: list[str]) -> ProblemSize:
        for line in template:
            if not self.is_broadcast_rhs_mmt_op(line):
                continue

            # ins(%11, %12 : tensor<2x1024x1280xi8>, tensor<10240x1280xi8>) outs(%19 : tensor<2x1024x10240xi32>)
            bmmt_re = (
                rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
            )
            dps = re.search(bmmt_re, line)
            if dps is None:
                continue

            lhs_tensor_type = dps.group("LHS")
            rhs_tensor_type = dps.group("RHS")
            lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
            assert lhs_shaped_type.rank() == 3

            rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
            assert rhs_shaped_type.rank() == 2

            res_tensor_type = dps.group("RES")
            res_shaped_type = parse_tensor_type(res_tensor_type)
            assert res_shaped_type.rank() == 3

            B0, M0, K0 = lhs_shaped_type.shape
            N1, K1 = rhs_shaped_type.shape
            B2, M2, N2 = res_shaped_type.shape
            assert B0 == B2
            assert M0 == M2
            assert N1 == N2
            assert K0 == K1
            return ProblemSize(
                MatmulSize(M0, N1, K0, B0),
                lhs_shaped_type,
                rhs_shaped_type,
                res_shaped_type,
                DispatchKind.broadcast_rhs_mmt,
            )

        assert False, "Shape not found"

    def get_shapes(self, template: list[str]) -> ProblemSize:
        if self.is_broadcast_rhs_mmt(template):
            return self.get_shapes_broadcast_rhs_mmt(template)

        for line in template:
            if "linalg.generic" not in line:
                continue
            if "lowering_config =" not in line:
                continue
            if '"reduction"' not in line:
                continue

            # ins(%7, %8 : tensor<2x1024x1280xf16>, tensor<20x64x1280xf16>)
            cont_re = (
                rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
            )
            dps = re.search(cont_re, line)
            if dps is None:
                continue

            lhs_tensor_type = dps.group("LHS")
            rhs_tensor_type = dps.group("RHS")
            lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
            assert lhs_shaped_type.rank() == len(self.lhs_dims)

            rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
            assert rhs_shaped_type.rank() == len(self.rhs_dims)

            res_tensor_type = dps.group("RES")
            res_shaped_type = parse_tensor_type(res_tensor_type)
            assert res_shaped_type.rank() >= 2

            M = math.prod(
                val if dim == "m" else 1
                for dim, val in zip(self.lhs_dims, lhs_shaped_type.shape)
            )
            N = math.prod(
                val if dim == "n" else 1
                for dim, val in zip(self.rhs_dims, rhs_shaped_type.shape)
            )
            K0 = math.prod(
                val if dim == "k" else 1
                for dim, val in zip(self.lhs_dims, lhs_shaped_type.shape)
            )
            K1 = math.prod(
                val if dim == "k" else 1
                for dim, val in zip(self.rhs_dims, rhs_shaped_type.shape)
            )
            assert K0 == K1

            return ProblemSize(
                MatmulSize(M, N, K0),
                lhs_type=lhs_shaped_type,
                rhs_type=rhs_shaped_type,
                res_type=res_shaped_type,
                dispatch_kind=DispatchKind.contraction,
            )

        assert False, "Shape not found"

    def get_transform_function_broadcast_rhs_mmt(
        self,
        problem_size: ProblemSize,
        functionName: str,
        configuration: BaseConfiguration,
    ) -> str:
        tile_sizes = configuration.get_batch_mmt_tile_sizes()
        config_mlir = configuration.get_mlir_config(tile_sizes)

        lhs_dynamic_batch = problem_size.lhs_type
        lhs_dynamic_batch.shape = lhs_dynamic_batch.shape.copy()
        lhs_dynamic_batch.shape[0] = -1

        return f"""
transform.named_sequence @{functionName}(%generic: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
%mmt = transform.include @match_broadcast_rhs_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
%lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
%rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
transform.iree.match.cast_compatible_type %lhs = tensor<{lhs_dynamic_batch}> : !transform.any_value
transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
{config_mlir}
transform.yield %generic, %config : !transform.any_op, !transform.any_param
}}
"""

    def apply_params_broadcast_rhs_mmt(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        modified = indent(
            self.get_transform_function_broadcast_rhs_mmt(
                problem_size, f"match_broadcast_rhs_mmt_Bx{M}x{N}x{K}", configuration
            ),
            "//   ",
        )
        modified += configuration.apply_configuration(
            template, configuration.get_batch_mmt_tile_sizes()
        )

        embeddable = indent(
            self.get_transform_function_broadcast_rhs_mmt(
                problem_size, f"match_op", configuration
            ),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        if self.is_broadcast_rhs_mmt(template):
            return self.apply_params_broadcast_rhs_mmt(
                problem_size, template, configuration
            )

        # TODO: Generate transform function.
        return MLIRTransformation(
            template,
            configuration.apply_configuration(
                template,
                configuration.get_contract_tile_sizes(self.tile_dims),
            ),
            "",
        )


class BatchMmtTuner(DispatchTuner):
    def supports(self, op_name: str) -> bool:
        return "batch_matmul_transpose_b" in op_name

    def get_shapes(self, template: list[str]) -> ProblemSize:
        for line in template:
            if "linalg.generic" not in line:
                continue
            if (
                r'iterator_types = ["parallel", "parallel", "parallel", "reduction"]'
                not in line
            ):
                continue
            # ins(%11, %12 : tensor<2x4096x640xi8>, tensor<2x640x640xi8>) outs(%19 : tensor<2x4096x640xi32>)
            bmmt_re = (
                rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
            )
            dps = re.search(bmmt_re, line)
            if dps is None:
                continue

            lhs_tensor_type = dps.group("LHS")
            rhs_tensor_type = dps.group("RHS")
            lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
            assert lhs_shaped_type.rank() == 3

            rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
            assert rhs_shaped_type.rank() == 3

            res_tensor_type = dps.group("RES")
            res_shaped_type = parse_tensor_type(res_tensor_type)
            assert res_shaped_type.rank() == 3

            B0, M0, K0 = lhs_shaped_type.shape
            B1, N1, K1 = rhs_shaped_type.shape
            B2, M2, N2 = res_shaped_type.shape
            assert B0 == B1
            assert B0 == B2
            assert M0 == M2
            assert N1 == N2
            assert K0 == K1
            return ProblemSize(
                MatmulSize(M0, N1, K0, B0),
                lhs_shaped_type,
                rhs_shaped_type,
                res_shaped_type,
                DispatchKind.batch_mmt,
            )

        assert False, "Shape not found"

    def get_transform_function_batch_mmt(
        self,
        problem_size: ProblemSize,
        functionName: str,
        configuration: BaseConfiguration,
    ) -> str:
        tile_sizes = configuration.get_batch_mmt_tile_sizes()
        config_mlir = configuration.get_mlir_config(tile_sizes)

        return f"""
transform.named_sequence @{functionName}(%generic: !transform.any_op {{transform.readonly}}) -> (!transform.any_op, !transform.any_param) {{
%mmt = transform.include @match_batch_mmt_i8_i8_i32 failures(propagate) (%generic) : (!transform.any_op) -> !transform.any_op
%lhs = transform.get_operand %generic[0] : (!transform.any_op) -> !transform.any_value
%rhs = transform.get_operand %generic[1] : (!transform.any_op) -> !transform.any_value
transform.iree.match.cast_compatible_type %lhs = tensor<{problem_size.lhs_type}> : !transform.any_value
transform.iree.match.cast_compatible_type %rhs = tensor<{problem_size.rhs_type}> : !transform.any_value
{config_mlir}
transform.yield %generic, %config : !transform.any_op, !transform.any_param
}}
"""

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        B = problem_size.matmul_size.B
        modified = indent(
            self.get_transform_function_batch_mmt(
                problem_size, f"match_batch_mmt_{B}x{M}x{N}x{K}", configuration
            ),
            "//   ",
        )
        modified += configuration.apply_configuration(
            template, configuration.get_batch_mmt_tile_sizes()
        )

        embeddable = indent(
            self.get_transform_function_batch_mmt(
                problem_size, f"match_op", configuration
            ),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)


class BatchMatmulTuner(DispatchTuner):
    def __init__(self, lhs_dims: str, rhs_dims: str, tile_dims: str):
        self.lhs_dims = lhs_dims
        self.rhs_dims = rhs_dims
        self.tile_dims = tile_dims

    def supports(self, op_name: str) -> bool:
        return "batch_matmul" in op_name

    def get_shapes(self, template: list[str]) -> ProblemSize:
        for line in template:
            if "linalg.batch_matmul" not in line:
                continue
            # ins(%9, %10 : tensor<64x72x1280xf16>, tensor<64x1280x1280xf16>)
            # outs(%12 : tensor<64x72x1280xf32>)
            cont_re = (
                rf"{MlirRegex.dps_ins_two_args()}\s+{MlirRegex.dps_outs_one_arg()}"
            )
            dps = re.search(cont_re, line)
            if dps is None:
                continue

            lhs_tensor_type = dps.group("LHS")
            rhs_tensor_type = dps.group("RHS")
            lhs_shaped_type = parse_tensor_type(lhs_tensor_type)
            assert lhs_shaped_type.rank() == len(self.lhs_dims)

            rhs_shaped_type = parse_tensor_type(rhs_tensor_type)
            assert rhs_shaped_type.rank() == len(self.rhs_dims)

            res_tensor_type = dps.group("RES")
            res_shaped_type = parse_tensor_type(res_tensor_type)
            assert res_shaped_type.rank() == lhs_shaped_type.rank()

            LHS = lhs_shaped_type.shape
            RHS = rhs_shaped_type.shape
            RES = res_shaped_type.shape

            B = math.prod(
                val if dim == "b" else 1 for dim, val in zip(self.lhs_dims, LHS)
            )
            B0 = math.prod(
                val if dim == "b" else 1 for dim, val in zip(self.lhs_dims, RHS)
            )
            B1 = math.prod(
                val if dim == "b" else 1 for dim, val in zip(self.lhs_dims, RES)
            )
            M = math.prod(
                val if dim == "m" else 1 for dim, val in zip(self.lhs_dims, LHS)
            )
            N = math.prod(
                val if dim == "n" else 1 for dim, val in zip(self.rhs_dims, RHS)
            )
            K0 = math.prod(
                val if dim == "k" else 1 for dim, val in zip(self.lhs_dims, LHS)
            )
            K1 = math.prod(
                val if dim == "k" else 1 for dim, val in zip(self.rhs_dims, RHS)
            )
            assert B == B0 and B == B1
            assert K0 == K1

            return ProblemSize(
                MatmulSize(M, N, K0, B),
                lhs_type=lhs_shaped_type,
                rhs_type=rhs_shaped_type,
                res_type=res_shaped_type,
                dispatch_kind=DispatchKind.batch_matmul,
            )

        assert False, "Shape not found"

    def get_transform_function_batch_matmul(
        self,
        problem_size: ProblemSize,
        tile_dims: str,
        functionName: str,
        configuration: BaseConfiguration,
    ) -> str:
        input0 = f"tensor<{problem_size.lhs_type}>"
        input1 = f"tensor<{problem_size.rhs_type}>"
        output = f"tensor<{problem_size.res_type}>"

        tile_sizes = configuration.get_contract_tile_sizes(tile_dims)
        config_mlir = configuration.get_mlir_config(tile_sizes)

        return f"""
    transform.named_sequence @{functionName}(%batch_matmul: !transform.any_op {{transform.readonly}})
    -> (!transform.any_op, !transform.any_param) {{
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %batch_matmul {{
    ^bb0(%lhs: {input0}, %rhs: {input1}, %out: {output}):
        %13 = linalg.batch_matmul
        ins(%lhs, %rhs : {input0}, {input1})
        outs(%out : {output}) -> {output}
    }} : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
        {config_mlir}
    transform.yield %batch_matmul, %config : !transform.any_op, !transform.any_param
    }}
    """

    def apply_params(
        self,
        problem_size: ProblemSize,
        template: list[str],
        configuration: BaseConfiguration,
    ) -> MLIRTransformation:
        M, N, K = problem_size.MNK
        modified = indent(
            self.get_transform_function_batch_matmul(
                problem_size,
                self.tile_dims,
                f"match_batch_matmul_{problem_size.matmul_size.B}x{M}x{N}x{K}",
                configuration,
            ),
            "//   ",
        )
        modified += configuration.apply_configuration(
            template,
            configuration.get_contract_tile_sizes(self.tile_dims),
        )

        embeddable = indent(
            self.get_transform_function_batch_matmul(
                problem_size, self.tile_dims, f"match_op", configuration
            ),
            "  ",
        )
        return MLIRTransformation(template, modified, embeddable)


def walk_callback_get_fn(
    op: ir.Operation,
    walk_result: OpWalkResult,
    dispatch_tuner_registry: DispatchTunerRegistry,
) -> ir.WalkResult:
    if op.name == "func.func":
        dispatch_tuner_registry.validate_translation([a for a in op.opview.attributes])
    if op.name == "util.func":
        func_name = str(op.opview.sym_name)
        walk_result.was_interrupted = True
        walk_result.dispatch_tuner = dispatch_tuner_registry.find_handler(func_name)
        return ir.WalkResult.INTERRUPT
    return ir.WalkResult.ADVANCE


def walk_mlir_op(
    mlir_module: ir.Module,
    dispatch_tuner_registry: DispatchTunerRegistry,
) -> OpWalkResult:
    walk_result = OpWalkResult()
    for op in mlir_module.body.operations:
        op.walk(
            lambda op: walk_callback_get_fn(op, walk_result, dispatch_tuner_registry),
            ir.WalkOrder.POST_ORDER,
        )
        if walk_result.was_interrupted:
            break
    return walk_result


def tune(
    input: str,  # Path to the mlir file to be tuned
    output: str = "",  # Path to the output directory, auto creates one if not given
    limit: int = 4096,  # Max candidates to be generated
    num_subgroups: int = 4,  # GPU spec, used to determine candidate generation constraints
    lhs_dims: str = "mk",  # Dimensions for the left-hand side operand in matrix operations
    rhs_dims: str = "nk",  # Dimensions for the right-hand side operand in matrix operations
    tile_dims: str = "mnk",  # Dimensions for the tile size
):
    input_file = str(input)

    if not output:
        output = get_default_output_dir()

    # Create the directory if it does not exist
    makedirs(str(output), exist_ok=True)

    tune_logger.debug(f"Output directory {output}")
    tune_logger.debug(f"Processing {input_file}")
    mlir_template = read_input_mlir(input_file)
    mlir_text = "".join(mlir_template)

    mlir_module = parse_mlir(mlir_text)
    # Save the input file as the first candidate.
    with open(path.join(output, f"0.mlir"), "w") as f:
        f.write(mlir_text)

    dispatch_tuner_registry = DispatchTunerRegistry()
    dispatch_tuner_registry.register(
        [
            MmtTuner(),
            Mmt4dTuner(),
            ConvTuner(),
            ContractionTuner(lhs_dims, rhs_dims, tile_dims),
            BatchMmtTuner(),
            BatchMatmulTuner(lhs_dims, rhs_dims, tile_dims),
        ]
    )

    walk_result = walk_mlir_op(mlir_module, dispatch_tuner_registry)

    dispatch_tuner = walk_result.dispatch_tuner
    problem_size = dispatch_tuner.get_shapes(mlir_template)
    tune_logger.debug(str(problem_size))

    solutions_strategy = SolutionStrategyFactory.create_strategy(mlir_text)
    configs = []
    for i, config in enumerate(
        solutions_strategy.generate_solutions(problem_size, num_subgroups)
    ):
        if i >= limit:
            break
        tune_logger.info(f"Solution #{i+1}: {config}")
        configs.append(config)
        tf_mlir = dispatch_tuner.apply_params(problem_size, mlir_template, config)

        with open(path.join(output, f"{i+1}.mlir"), "w") as f:
            f.write(tf_mlir.modified)
        with open(path.join(output, f"{i+1}_config.mlir"), "w") as f:
            f.write(tf_mlir.embeddable)

    with open(path.join(output, "configs.pkl"), "wb") as file:
        pickle.dump(configs, file)

    tune_logger.info(f"Generated {len(configs)} candidates")
    tune_logger.info(f"Configurations .pkl is stored in {output}/configs.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input mlir file", type=str)
    parser.add_argument(
        "-o", "--output", help="Output dir", type=str, default=get_default_output_dir()
    )
    parser.add_argument(
        "-l",
        "--limit",
        help="Max number of candidates generated",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--num-subgroups",
        help="Number of subgroups per workgroup to use. (-1 == unconstrained)",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--lhs-dims", help="Map of LHS matmul dims", type=str, default="mk"
    )
    parser.add_argument(
        "--rhs-dims", help="Map of RHS matmul dims", type=str, default="nk"
    )
    parser.add_argument(
        "--tile-dims", help="Map of tile size matmul dims", type=str, default="mnk"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output to stdout"
    )

    args = parser.parse_args()
    tune_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # Create printing formatter for logging info
    formatter = logging.Formatter("%(message)s")

    # Create a handler to print to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    tune_logger.addHandler(console_handler)

    # # Optionally, add a file handler to log to a file
    # file_handler = logging.FileHandler("tune.log")
    # file_handler.setFormatter(formatter)
    # tune_logger.addHandler(file_handler)

    tune(
        args.input,
        args.output,
        args.limit,
        args.num_subgroups,
        args.lhs_dims,
        args.rhs_dims,
        args.tile_dims,
    )


if __name__ == "__main__":
    args = main()
