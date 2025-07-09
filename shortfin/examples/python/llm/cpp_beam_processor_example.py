#!/usr/bin/env python3
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Example showing how to use the C++ BeamProcessor implementation from Python.

This demonstrates the C++ implementation of the process_beams function
that was originally written in Python.
"""

import shortfin as sf


def create_mock_selection_callback(eos_token_id=2):
    """Create a mock beam selection callback for testing."""
    token_counter = [10]  # Use list for mutable counter

    def selection_callback(active_beams, completed_beams):
        selections = []
        for beam in active_beams:
            # Create a new beam state with the next token
            new_beam = sf.llm.BeamState()
            new_beam.exec_req = beam.exec_req
            new_beam.score = beam.score
            new_beam.accumulated_normalization = beam.accumulated_normalization

            # Simulate different token selection logic
            if len(selections) == 0 and len(active_beams) > 1:
                # First beam in a multi-beam scenario - give it EOS token sometimes
                new_beam.last_token = (
                    eos_token_id if token_counter[0] > 15 else token_counter[0]
                )
            else:
                new_beam.last_token = token_counter[0]

            token_counter[0] += 1
            selections.append(new_beam)

        return selections

    return selection_callback


def create_mock_inference_request(instance_id, input_tokens):
    """Create a mock inference request for testing."""
    req = sf.llm.InferenceRequest()
    req.instance_id = instance_id
    req.input_token_ids = input_tokens.copy()
    req.prompt_length = (
        len(input_tokens) - 1
    )  # Assume last token is the start of generation
    req.start_position = len(input_tokens) - 1
    return req


def main():
    """Main example function."""
    print("C++ BeamProcessor Example")
    print("=" * 40)

    # Configuration
    eos_token_id = 2
    num_beams = 3

    # Create the beam processor with a mock selection callback
    selection_callback = create_mock_selection_callback(eos_token_id)
    processor = sf.llm.BeamProcessor(eos_token_id, num_beams, selection_callback)

    print(f"Created BeamProcessor:")
    print(f"  EOS Token ID: {processor.eos_token_id}")
    print(f"  Number of Beams: {processor.num_beams}")
    print()

    # Create mock inference requests
    req1 = create_mock_inference_request("request_1", [1, 3, 5, 7])
    req2 = create_mock_inference_request("request_2", [1, 4, 6, 8])
    req3 = create_mock_inference_request("request_3", [1, 2, 9, 11])

    # Create initial beam states
    beam1 = sf.llm.BeamState(req1)
    beam1.score = 0.9

    beam2 = sf.llm.BeamState(req2)
    beam2.score = 0.85

    beam3 = sf.llm.BeamState(req3)
    beam3.score = 0.8

    active_beams = [beam1, beam2, beam3]
    completed_beams = []

    print("Initial state:")
    print(f"  Active beams: {len(active_beams)}")
    print(f"  Completed beams: {len(completed_beams)}")
    for i, beam in enumerate(active_beams):
        print(f"    Beam {i+1}: {beam.exec_req.input_token_ids}, score: {beam.score}")
    print()

    # Simulate multiple rounds of beam processing
    for round_num in range(1, 6):
        print(f"Round {round_num}:")
        print("-" * 20)

        # Process beams using C++ implementation
        result = processor.process_beams(active_beams, completed_beams)

        if not result.success:
            print(f"Error in processing: {result.error_message}")
            break

        # Update our beam lists
        active_beams = result.active_beams
        completed_beams = result.completed_beams

        print(f"  Result: {len(active_beams)} active, {len(completed_beams)} completed")

        # Show details
        if active_beams:
            print("  Active beams:")
            for i, beam in enumerate(active_beams):
                print(
                    f"    Beam {i+1}: {beam.exec_req.input_token_ids}, last_token: {beam.last_token}"
                )

        if completed_beams:
            print("  Completed beams:")
            for i, beam in enumerate(completed_beams):
                print(
                    f"    Beam {i+1}: {beam.exec_req.input_token_ids}, last_token: {beam.last_token}"
                )

        print()

        # Stop if all beams are completed
        if not active_beams:
            print("All beams completed!")
            break

    # Final summary
    print("Final Results:")
    print("=" * 20)
    print(f"Total completed beams: {len(completed_beams)}")
    for i, beam in enumerate(completed_beams):
        generated_tokens = beam.exec_req.input_token_ids[beam.exec_req.prompt_length :]
        print(f"  Beam {i+1}: Generated tokens = {generated_tokens}")


def demo_parallel_processing():
    """Demonstrate parallel processing of multiple beam groups."""
    print("\nParallel Processing Demo")
    print("=" * 40)

    # This would demonstrate the parallel processing capabilities
    # For now, we'll just show the API structure
    print("Note: Parallel processing requires multiple BeamProcessor instances")
    print("and would typically be used for processing multiple independent")
    print("beam groups simultaneously.")

    # Example setup (not fully functional without proper threading setup)
    try:
        processors = []
        active_beams_batch = []
        completed_beams_batch = []

        # This is the API that would be called for parallel processing
        # results = sf.llm.parallel.process_beam_groups_parallel(
        #     processors, active_beams_batch, completed_beams_batch
        # )
        print(
            "Parallel processing API available via sf.llm.parallel.process_beam_groups_parallel()"
        )
    except Exception as e:
        print(f"Parallel processing demo setup: {e}")


if __name__ == "__main__":
    main()
    demo_parallel_processing()
