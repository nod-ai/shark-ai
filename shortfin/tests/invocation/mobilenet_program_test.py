# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import array
import asyncio
import time
import functools
import pytest

import shortfin as sf
import shortfin.array as sfnp


@pytest.fixture
def lsys():
    sc = sf.SystemBuilder()
    lsys = sc.create_system()
    yield lsys
    lsys.shutdown()


@pytest.fixture
def fiber0(lsys):
    return lsys.create_fiber()


@pytest.fixture
def device(fiber0):
    return fiber0.device(0)


@pytest.fixture
def mobilenet_program_function(
    lsys, mobilenet_compiled_path
) -> tuple[sf.ProgramFunction]:
    program_module = lsys.load_module(mobilenet_compiled_path)
    program = sf.Program([program_module], devices=lsys.devices)
    main_function = program["module.torch-jit-export"]
    return main_function


@pytest.fixture
def mobilenet_program_function_per_call(
    lsys, mobilenet_compiled_path
) -> tuple[sf.ProgramFunction]:
    program_module = lsys.load_module(mobilenet_compiled_path)
    program = sf.Program(
        [program_module], devices=lsys.devices, isolation=sf.ProgramIsolation.PER_CALL
    )
    main_function = program["module.torch-jit-export"]
    return main_function


def get_mobilenet_ref_input(device) -> sfnp.device_array:
    dummy_data = array.array(
        "f", ([0.2] * (224 * 224)) + ([0.4] * (224 * 224)) + ([-0.2] * (224 * 224))
    )
    device_input = sfnp.device_array(device, [1, 3, 224, 224], sfnp.float32)
    staging_input = device_input.for_transfer()
    with staging_input.map(discard=True) as m:
        m.fill(dummy_data)
    device_input.copy_from(staging_input)
    return device_input


async def assert_mobilenet_ref_output(device, device_output):
    host_output = device_output.for_transfer()
    host_output.copy_from(device_output)
    await device
    flat_output = host_output.items
    absmean = functools.reduce(
        lambda x, y: x + abs(y) / len(flat_output), flat_output, 0.0
    )
    # Note: this value was just copied from a sample run of the test.
    # Comparison against a reference backend for this model is tested upstream
    # in https://github.com/iree-org/iree-test-suites/tree/main/onnx_models.
    assert absmean == pytest.approx(0.81196929)


# Tests that a single invocation on a single fiber works.
def test_invoke_mobilenet_single_per_fiber(lsys, fiber0, mobilenet_program_function):
    assert mobilenet_program_function.isolation == sf.ProgramIsolation.PER_FIBER
    device = fiber0.device(0)

    async def main():
        device_input = get_mobilenet_ref_input(device)
        (device_output,) = await mobilenet_program_function(device_input, fiber=fiber0)
        await assert_mobilenet_ref_output(device, device_output)
        del device_output

    lsys.run(main())


# Tests that a single invocation on a single fiber in per_call mode works.
def test_invoke_mobilenet_single_per_call(
    lsys, fiber0, mobilenet_program_function_per_call
):
    assert mobilenet_program_function_per_call.isolation == sf.ProgramIsolation.PER_CALL
    device = fiber0.device(0)

    async def main():
        device_input = get_mobilenet_ref_input(device)
        (device_output,) = await mobilenet_program_function_per_call(
            device_input, fiber=fiber0
        )
        await assert_mobilenet_ref_output(device, device_output)
        del device_output

    lsys.run(main())


# Tests that chained back to back invocations on the same fiber work correctly.
# Does an async gather/assert with all results at the end.
def test_invoke_mobilenet_chained_per_fiber(lsys, fiber0, mobilenet_program_function):
    assert mobilenet_program_function.isolation == sf.ProgramIsolation.PER_FIBER
    device = fiber0.device(0)

    async def main():
        device_input = get_mobilenet_ref_input(device)
        results = [
            await mobilenet_program_function(device_input, fiber=fiber0)
            for _ in range(5)
        ]

        gather = asyncio.gather(
            *[
                assert_mobilenet_ref_output(device, device_output)
                for (device_output,) in results
            ]
        )
        del results
        await gather

    lsys.run(main())


# Tests that parallel invocations on a single fiber with a program in PER_CALL
# isolation functions properly. Note that in this variant, the await is done
# on all invocations vs serially per invocation (as in
# test_invoke_mobilenet_chained_per_fiber). This would be illegal if done on the
# same fiber without PER_CALL isolation managing forks.
#
# Note that since these are all operating on the same fiber, they are added to
# the device-side work graph with a one-after-the-other dependency, but the
# host side schedules concurrently.
def test_invoke_mobilenet_parallel_per_call(
    lsys, fiber0, mobilenet_program_function_per_call
):
    assert mobilenet_program_function_per_call.isolation == sf.ProgramIsolation.PER_CALL
    device = fiber0.device(0)

    async def main():
        device_input = get_mobilenet_ref_input(device)
        results = await asyncio.gather(
            *[
                mobilenet_program_function_per_call(device_input, fiber=fiber0)
                for _ in range(5)
            ]
        )

        gather = asyncio.gather(
            *[
                assert_mobilenet_ref_output(device, device_output)
                for (device_output,) in results
            ]
        )
        del results
        await gather

    lsys.run(main())


# Same as above but uses explicit isolation controls on the function vs as the
# program level. If this constraint were violated, shortfin makes a best effort
# attempt to detect the situation and raise an exception, but there are a subset
# of programs which are purely async and would make detection of this exception
# lossy in the synchronous completion case.
def test_invoke_mobilenet_parallel_per_call_explicit(
    lsys, fiber0, mobilenet_program_function
):
    assert mobilenet_program_function.isolation == sf.ProgramIsolation.PER_FIBER
    device = fiber0.device(0)

    async def main():
        device_input = get_mobilenet_ref_input(device)
        results = await asyncio.gather(
            *[
                mobilenet_program_function(
                    device_input, fiber=fiber0, isolation=sf.ProgramIsolation.PER_CALL
                )
                for _ in range(50)
            ]
        )

        gather = asyncio.gather(
            *[
                assert_mobilenet_ref_output(device, device_output)
                for (device_output,) in results
            ]
        )
        del results
        await gather

    lsys.run(main())


# Tests that independent executions on multiple fibers all run concurrently.
# All fibers share the same host thread but schedule concurrently. Since
# each fiber has its own timeline, device side graphs have no dependency on
# each other and also schedule concurrently.
def test_invoke_mobilenet_multi_fiber_per_fiber(lsys, mobilenet_program_function):
    assert mobilenet_program_function.isolation == sf.ProgramIsolation.PER_FIBER

    class InferProcess(sf.Process):
        async def run(self):
            start_time = time.time()

            def duration():
                return round((time.time() - start_time) * 1000.0)

            print(f"{self}: Start")
            device = self.fiber.device(0)
            device_input = get_mobilenet_ref_input(device)
            (device_output,) = await mobilenet_program_function(
                device_input, fiber=self.fiber
            )
            print(f"{self}: Program complete (+{duration()}ms)")
            await assert_mobilenet_ref_output(device, device_output)
            del device_output
            print(f"{self} End (+{duration()}ms)")

    async def main():
        start_time = time.time()

        def duration():
            return round((time.time() - start_time) * 1000.0)

        fibers = [lsys.create_fiber() for _ in range(5)]
        print("Fibers:", fibers)
        processes = [InferProcess(fiber=f).launch() for f in fibers]
        print("Waiting for processes:", processes)
        await asyncio.gather(*processes)
        print(f"All processes complete: (+{duration()}ms)")

    lsys.run(main())
