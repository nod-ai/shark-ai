# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#!/usr/bin/env bash

set -o pipefail
shopt -s extglob

# The script accepts a number of command line arguments from the user.
#   1. -h -- prints the help string [same as no flag given at all].
#   2. -o -- specifies the export directory, where the MLIR and the VMFB are placed.
#   3. -w -- specifies the location of the weight file.
#   4. -a -- specifies where to place the artifacts (such as sharded weights).
#            if this flag is unset, the artifact dir defaults to <export_dir>/artifacts/.
#   5. -b -- specifies the batch sizes for the export.
#   6. -i -- path to the IREE build dir (for dev purposes).
#   7. -v -- enable verbose logging.
#   8. -s -- Whether to shard the weights or not. Defaults to tp1 if not set, else tp8.
#   9. -c -- If set, compilation is enabled as a part of the pipeline.
#  10. -e -- If set, export is called. This allows sharding to be run separately from export.
#  11. -o -- The value for EXPORT_DIR, where the exported model will be placed


print_help_string() {
  printf "%s\n" "USAGE: run_shark_ai [OPTIONS] ... -o [EXPORT_DIR]
Exports the model using shark-ai and compiles the IR to produce the VMFB module.

Some options are required to be specified.
    -h            prints the help string. The same output is emitted when no arguments are given.
    -v            enables verbose logging. When not specified, only errors are logged.
    -w            the location of the GGUF/IRPA file(s) that contain the parameters. (required)
    -a            the location of the artifacts (sharded weights). (defaults to EXPORT_DIR/artifacts)
    -b            batch sizes for the export. Multiple batch sizes should be separated by ','.
                    defaults to (1,4).
    -i            the directory containing iree-compile, if from local build, to use for compilation.
                    defaults to the PATH value if not set.
    -s            use to enable sharding. shards to tp8 if set, otherwise tp1.
    -c            whether to also compile the exported model and produce a VMFB file.
    -e            calls export first if set. This allows the export and sharding commands to be
                    run independently.
    -o            the location for the exported model. (required)"
}

print_help_string_and_exit() {
  print_help_string && exit $1
}

unset -v VERBOSE_LOGGING
unset -v WEIGHT_FILE_LOC
unset -v ARTIFACT_LOC
unset -v EXPORT_BATCH_SIZES
unset -v EXPORT
unset -v ROCR_VISIBLE_DEVICES
unset -v MODEL_CONFIG_LOC
unset -v IREE_BUILD_DIR
unset -v SHARD_WEIGHTS
unset -v COMPILE
unset -v EXPORT_DIR

VERBOSE_LOGGING=false
EXPORT_BATCH_SIZES=1,4
EXPORT=0
ROCR_VISIBLE_DEVICES=1
MODEL_CONFIG_LOC=0
SHARD_WEIGHTS=0
COMPILE=0

err() { printf "[ERROR] %s\n" "$*" >&2 && exit 1; }
log() { "$VERBOSE_LOGGING" && printf "%s\n" "$*" >&2; }
info() { "$VERBOSE_LOGGING" && log "[INFO] $*"; }
warn() { "$VERBOSE_LOGGING" && log "[WARNING] $*"; }

parse_and_handle_args() {
  while getopts ":ehvscw:t:a:b:i:o:" flag; do
    case ""$flag"" in
      h)
        print_help_string_and_exit 0
        ;;
      v)
        VERBOSE_LOGGING=true
        ;;
      w)
        WEIGHT_FILE_LOC="$OPTARG"
        ;;
      a)
        ARTIFACT_LOC=$(realpath "$OPTARG")
        if [[ ! -d "$ARTIFACT_LOC" ]]; then
            warn ""$ARTIFACT_LOC" does not exist, creating..."
            mkdir -p "$ARTIFACT_LOC"
        fi
        ;;
      b)
        EXPORT_BATCH_SIZES="$OPTARG"
        ;;
      i)
        IREE_BUILD_DIR=$(realpath "$OPTARG")
        ;;
      s)
        SHARD_WEIGHTS=1
        info "Weights will be sharded with tp 8"
        ;;
      c)
        COMPILE=1
        info "Compilation is enabled"
        ;;
      e)
        EXPORT=1
        info "Export is enabled"
        ;;
      o)
        EXPORT_DIR=$(realpath "$OPTARG")
        info "Exported model will be saved at $EXPORT_DIR"
        ;;
      :)
        err "option -$OPTARG expected an argument"
        ;;
      \?)   print_help_string && err "INVALID OPTION -$OPTARG"
        ;;
    esac
  done
  shift $(( "$OPTIND"-1 ))
}

check_valid() {

  for arg in EXPORT_DIR WEIGHT_FILE_LOC; do
    if [[ -z "${!arg}" ]]; then
      print_help_string
      err "Missing required argument for ${arg@Q}"
    fi
  done

  if [[ ! -d "$EXPORT_DIR" ]]; then
    warn "$EXPORT_DIR does not exist, creating..."
    mkdir -p "$EXPORT_DIR"
  fi

  MODEL_CONFIG_LOC="$EXPORT_DIR"/config.json

  [[ -f "$WEIGHT_FILE_LOC" ]] || (err "Missing argument to -w or invalid path to weight file" && exit 1)
  info "Using weight file from $WEIGHT_FILE_LOC"
}

sharktank_export() {
  info "Starting Export..."
  MLIR_PATH="$EXPORT_DIR"/model.mlir
  info "Model MLIR will be saved at $MLIR_PATH"
  info "Generated config.json will be placed in $MODEL_CONFIG_LOC"

  python3 -m sharktank.examples.export_paged_llm_v1 \
    --gguf-file="$WEIGHT_FILE_LOC" \
    --output-mlir="$MLIR_PATH" \
    --output-config="$MODEL_CONFIG_LOC" \
    --bs="$EXPORT_BATCH_SIZES" 2>&1
  if [[ $? -ne 0 ]]; then
    err "Failed to run export"
  fi

  info "Successfully exported model to $EXPORT_DIR"

}

shard() {
    info "Sharding enabled. Sharding weights and saving to $ARTIFACT_LOC"

    if [[ -z "$ARTIFACT_LOC" ]] || [[ ! -d "$ARTIFACT_LOC" ]]; then
      ARTIFACT_LOC="$EXPORT_DIR"/artifacts
      mkdir -p "$ARTIFACT_LOC"
      warn "Invalid or missing path to artifacts dir; Defaulting to $EXPORT_DIR/artifacts"
    fi

    info "Setting artifacts dir to $ARTIFACT_LOC"

    weight_file_stem=$(echo "$WEIGHT_FILE_LOC" | rev | cut -d / -f 1 | cut --complement -d . -f 1 | rev)
    shard_file_suffix="_tp8.irpa"
    SHARD_OUTPUT_FILE="$ARTIFACT_LOC"/"$weight_file_stem$shard_file_suffix"
    python3 -m sharktank.examples.sharding.shard_llm_dataset \
      --irpa-file "$WEIGHT_FILE_LOC" \
      --output-irpa "$SHARD_OUTPUT_FILE" \
      --tensor-parallelism-size 8 2>&1

    if [[ $? -ne 0 ]]; then
      err "Failed to run export"
    fi
    info "Successfully sharded weights; unranked shard file located at $SHARD_OUTPUT_FILE"

    info "Generating sharded IR from unranked shard file"
    TP8_MODEL_CONFIG_LOC="$EXPORT_DIR"/config_tp8_nondecomposed.json
    python3 -m sharktank.examples.export_paged_llm_v1 \
            --bs=4 \
            --irpa-file="$SHARD_OUTPUT_FILE" \
            --output-mlir="$EXPORT_DIR"/model_tp8_nondecomposed.mlir \
            --output-config="$TP8_MODEL_CONFIG_LOC" 2>&1

    if [[ $? -ne 0 ]]; then
      err "Failed to run generate tp8 IR"
    fi

    info "Successfully saved tp8 IR at $EXPORT_DIR/model_tp8_nondecomposed.mlir"
    info "TP8 model config file saved to $TP8_MODEL_CONFIG_LOC"
}

compile_tp8() {
  info "Starting compilation with tp 8"
  "$IREE_BUILD_DIR"/iree-compile --compile-to=input  \
        "$EXPORT_DIR"/model_tp8_nondecomposed.mlir  \
        -o "$EXPORT_DIR"/model_tp8_nondecomposed.iree.mlir

  "$IREE_BUILD_DIR"/iree-compile  \
        "$EXPORT_DIR"/model_tp8_nondecomposed.iree.mlir  \
        --iree-hip-target=gfx942  \
        --iree-hal-target-device=hip[0]  \
        --iree-hal-target-device=hip[1]  \
        --iree-hal-target-device=hip[2]  \
        --iree-hal-target-device=hip[3]  \
        --iree-hal-target-device=hip[4]  \
        --iree-hal-target-device=hip[5]  \
        --iree-hal-target-device=hip[6]  \
        --iree-hal-target-device=hip[7]  \
        --iree-dispatch-creation-enable-aggressive-fusion=true     \
        --iree-global-opt-propagate-transposes=true  \
        --iree-opt-aggressively-propagate-transposes=true     \
        --iree-opt-data-tiling=false  \
        --iree-preprocessing-pass-pipeline='builtin.module(util.func(iree-preprocessing-generalize-linalg-matmul-experimental))'     \
        --iree-hal-indirect-command-buffers=true  \
        --iree-stream-resource-memory-model=discrete  \
        --iree-hip-legacy-sync=false    \
        --iree-hal-memoization=true  \
        --iree-opt-strip-assertions \
        -o="$EXPORT_DIR"/model_tp8.vmfb 2>&1

  if [[ $? -ne 0 ]]; then
    err "Failed to compile model"
  fi

  info "Successfully compiled the model and saved VMFB at $EXPORT_DIR/model_tp8.vmfb"
}

compile_tp1() {
  info "Starting compilation"
  "$IREE_BUILD_DIR"/iree-compile "$EXPORT_DIR"/model.mlir \
               --iree-hal-target-backends=rocm \
               --iree-hip-target=gfx942 \
               -o "$EXPORT_DIR"/model.vmfb 2>&1
  if [[ $? -ne 0 ]]; then
    err "Failed to compile model"
  fi

  info "Successfully compiled the model and saved VMFB at $EXPORT_DIR/model.vmfb"
}

main() {
  if [[ $# -eq 0 ]]; then
    print_help_string_and_exit 1
  fi

  parse_and_handle_args "$@"
  check_valid

  if [[ "$EXPORT" -eq 0 ]] && [[ "$SHARD_WEIGHTS" -eq 0 ]]; then
    err "At least one of export (-e) or shard (-s) must be specified"
  fi

  if [[ "$EXPORT"  -eq 1 ]]; then
    sharktank_export
  fi

  if [[ "$SHARD_WEIGHTS" -eq 1 ]]; then
    shard
  fi

  if [[ "$COMPILE" -eq 1 ]]; then
    if [[ -z "$IREE_BUILD_DIR" ]]; then
      compiler_loc=$(which iree-compile)
      [[ ! -f "$compiler_loc" ]] && err "iree-compile not in PATH, and IREE_BUILD_DIR was not provided through '-i'"
      IREE_BUILD_DIR=$(dirname "$compiler_loc")
      warn "Missing argument to IREE build dir. Defaulting to installation found in PATH"
      info "Using iree-compile from $IREE_BUILD_DIR"
    fi

    if [[ ! -f "$IREE_BUILD_DIR/iree-compile" ]]; then
      err "Could not find iree-compile. Please add to PATH or provide local binary through -i"
    fi

    if [[ "$SHARD_WEIGHTS" -eq 1 ]]; then
      compile_tp8
    else
      compile_tp1
    fi
  fi

}

main "$@"
